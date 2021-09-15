import argparse, os
import logging
import random
import json
import multiprocessing

import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

try:
    import cv2
except ImportError:
    pass

from proxies.database import ObjectDatabase
import proxies.shapenet as shapenet
import proxies.errors as errors
import proxies.logger as logger
import proxies.eval as eval
import proxies.analysis as analysis


def image_results(
    odb,
    results,
    reorder,
    include_annotations,
    class_filter=None,
    stop=50,
    top_k=3,
    resolution=tuple((480, 480)),
):
    class_filter = [shapenet.string2class(c) for c in class_filter]

    # Evaluate
    image_row_stacker, get_eval_results = [], []
    overall_idx = 0
    for idx, (res, obj) in enumerate(zip(results, odb.query_objects)):

        # Only display objects in the reorder
        if idx not in set(reorder):
            continue

        # Filter by class (optionally)
        if (class_filter is not None) and (obj.class_id not in class_filter):
            continue

        # Early stop
        overall_idx += 1
        if (stop is not None) and (overall_idx > stop):
            break

        if overall_idx != 97:
            continue

        # Get image of the broken object
        col_stacker = []

        # Get the image
        img = cv2.resize(obj.load_render("b", angle=45), resolution)

        # Add the class
        if include_annotations:
            cv2.putText(
                img,
                shapenet.class2string(obj.true_class),
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 0),
                2,
            )
            # Add the name
            cv2.putText(
                img,
                obj.object_id,
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )
            # Add the index
            cv2.putText(
                img,
                str(overall_idx),
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )
        col_stacker.append(img)

        get_eval_results.append([])
        for (obj_idx, score) in res[:top_k]:
            # print(len(odb.database_objects), obj_idx)
            robj = odb.database_objects[int(obj_idx)]

            # Get the image
            img = cv2.resize(robj.load_render("c", angle=45), resolution)

            # Add the class
            if include_annotations:
                cv2.putText(
                    img,
                    shapenet.class2string(robj.true_class),
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 0),
                    2,
                )
                # Add the name
                cv2.putText(
                    img,
                    robj.object_id,
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2,
                )
                # Add score
                cv2.putText(
                    img,
                    str(round(score, 3)),
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2,
                )
            col_stacker.append(img)

        # Concat the result
        image_row_stacker.append(np.hstack(col_stacker))
    img = Image.fromarray(np.vstack(image_row_stacker))

    return img


def image_class(odb, class_id, stop=50):
    """save one big list of all objects in that class"""

    stacker = []
    counter = 0
    for idx, o in enumerate(odb.database_objects):

        # Filter by class
        if o.class_id == shapenet.string2class(class_id):
            counter += 1
            img = cv2.resize(o.load_render("c", angle=45), (1000, 1000))
            # Add score
            cv2.putText(
                img,
                str(idx),
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )
            stacker.append(img)

        # Early stop
        if counter > stop:
            break

    class_name = shapenet.class2string(class_id)
    Image.fromarray(np.vstack(stacker)).save("{}.png".format(class_name))


def eval_wrapper(fn, *args):
    # This prevents the evaluator from crashing
    try:
        return fn(*args)
    except errors.MeshNotClosedError:
        return None


def get_eval_results(odb, results, nthreads, eval_fn, topk):
    # Start the threadpool
    main_pool = multiprocessing.Pool(nthreads)

    pbar = tqdm.tqdm(total=len(results) * topk)
    # Chamfer distance
    col_stacker = []
    for k in range(topk):
        pbar.write("Computing top {} results".format(k + 1))

        # Submit to threadpool
        row_futures = [
            main_pool.apply_async(
                eval_wrapper,
                args=(obj.eval, eval_fn, odb.database_objects[int(r[k, 0])]),
            )
            for obj, r in zip(odb.query_objects, results)
        ]

        # Get results
        row_stacker = []
        for f in row_futures:
            result = f.get()
            if result is None:
                row_stacker.append(np.nan)
            else:
                row_stacker.append(result)
            pbar.update(1)

        # Stack
        col_stacker.append(row_stacker)
    pbar.close()

    # Transpose
    return np.array(col_stacker).T


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--feat_mask",
        type=int,
        nargs="+",
        help="A binary mask indicating which features to use.",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        help="Directory that stores the data on disk. In the case of ShapeNet "
        + "this is '/home/.../ShapeNetCore.v2'. This does not have to be passed "
        + "if using fully populated database.",
    )
    parser.add_argument(
        "--load",
        type=str,
        default="index",
        help="Path to directory containing saved database. This directory will "
        + "contain several files. Same directory that was passed during databse "
        + "creation.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="How many query objects to pass at a time to the database query "
        + "engine. Note that passing many at once will see a significant "
        + "speedup.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="__temp",
        help="Results will be dumped to temporary files so that evaluation can "
        + "be performed more quickly. These temporary files will live in this "
        + "directory.",
    )
    parser.add_argument(
        "--render",
        type=str,
        default=None,
        help="If passed, will output a summary render with 50 images, saved to "
        + "this path.",
    )
    parser.add_argument(
        "--chamfer",
        type=str,
        default=None,
        help="If passed, will compute chamfer distance for each result, saved to "
        + "this path.",
    )
    parser.add_argument(
        "--norm_const",
        type=str,
        default=None,
        help="If passed, will compute normal consistency for each result, saved to "
        + "this path.",
    )
    parser.add_argument(
        "--values",
        type=str,
        default=None,
        help="If passed, will extract the values for each result, saved to this "
        + "path.",
    )
    parser.add_argument(
        "--pr_curve",
        type=str,
        default=None,
        help="If passed, will compute precision recall curve.",
    )
    parser.add_argument(
        "--map",
        type=str,
        default=None,
        help="If passed, will compute mean average precision.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of threads to use during evaluation.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Number of results to return from the query.",
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default=None,
        help="If passed, will dump output of log to this file.",
    )
    parser.add_argument(
        "--use_gpu",
        default=False,
        action="store_true",
        help="If passed, will perform query on the gpu.",
    )
    parser.add_argument(
        "--annotations",
        default=False,
        action="store_true",
        help="If passed, will render images with annotations.",
    )
    parser.add_argument(
        "--allowlist",
        type=str,
        default=None,
        help="Use to make sure results are consistent between different approaches.",
    )
    logger.add_logger_args(parser)
    args = parser.parse_args()
    logger.configure_logging(args)

    assert os.path.isdir(args.cache_dir), "Cachedir not found: {}".format(
        args.cache_dir
    )
    assert args.batch_size > 0
    assert args.topk > 0
    # assert args.allowlist is not None

    # Add log file output
    if args.logfile is not None:
        rootLogger = logging.getLogger()
        fileHandler = logging.FileHandler(args.logfile)
        rootLogger.addHandler(fileHandler)

    # Load the database
    odb = ObjectDatabase(
        load_from=args.load, root_dir=args.root_dir, use_gpu=args.use_gpu
    )
    print(odb)
    print("{} total objects".format(len(odb.database_objects) + len(odb.query_objects)))
    print("{} database objects".format(len(odb.database_objects)))
    print("{} query objects".format(len(odb.query_objects)))

    allow_list_file = args.allowlist

    if allow_list_file is not None:
        logging.info("Loading from allowlist {}".format(allow_list_file))
        # Apply allowlist during evaluation
        allow_dict = json.load(open(allow_list_file, "r"))
        database_excl_list = []
        for o in odb.database_objects:
            o_full_id = o.get_id_str
            if o_full_id not in allow_dict["database"]:
                database_excl_list.append(o_full_id)

        new_query_objects = []
        for o in odb.query_objects:
            o_full_id = o.get_id_str
            if o_full_id in allow_dict["query"]:
                new_query_objects.append(o)

        # This is okay to apply directly
        odb._query_object_id_list = new_query_objects
    else:
        database_excl_list = []

    print("After allowlist removal:")
    print(
        "{} database objects".format(
            len(odb.database_objects) - len(database_excl_list)
        )
    )
    print("{} query objects".format(len(odb.query_objects)))

    # Double check the feature mask
    if args.feat_mask is None:
        args.feat_mask = [True for _ in range(len(odb._feature_types))]
    assert len(args.feat_mask) == len(odb._feature_types)
    logging.info(
        "Running on {}/{} features".format(sum(args.feat_mask), len(odb._feature_types))
    )

    # Set weights
    if len(odb._feature_types) == 2 and "global_POINTNET" in odb.features:
        alpha = 0.63
        odb._feature_weights = [(alpha), (1 - alpha)]
    elif len(odb._feature_types) == 2 and "global_FCGF" in odb.features:
        alpha = 0.95
        odb._feature_weights = [(alpha), (1 - alpha)]
    elif len(odb._feature_types) == 1:
        odb._feature_weights = [1]
    else:
        raise RuntimeError()

    # This handles caching of results list
    fname = os.path.join(args.cache_dir, "__temp__resultslist.npz")
    try:
        results_list = np.load(fname, allow_pickle=True)["results_list"]
    except:
        # Batch and run the query
        results_list = []
        for idx, i in enumerate(range(0, len(odb.query_objects), args.batch_size)):
            results_list.extend(
                odb.hierarchical_query(
                    odb.query_objects[
                        i : min(i + args.batch_size, len(odb.query_objects))
                    ],
                    feat_mask=args.feat_mask,
                    topk=args.topk,
                    batch_num=idx,
                    query_cache_fname=os.path.join(
                        args.cache_dir, "__temp__realsies__{}_{}.npz"
                    ),
                    database_excl_list=database_excl_list,
                )
            )
        np.savez(fname, results_list=results_list)

    # Build render
    if args.render is not None:
        random.seed(4111)

        stop = 1000

        # Scramble the order, and only render k of the objects
        reorder = random.choices(
            range(len(results_list)),
            k=len(results_list),
            # k=min(stop, len(results_list)),
        )

        # Build and save the summary render
        logging.info("Building summary render")
        image_results(
            odb,
            results_list,
            reorder,
            args.annotations,
            class_filter=[
                "jar",
                "bottle",
                "tower",
                "basket",
                "can",
                "birdhouse",
                "mug",
                "bowl",
                "flowerpot",
            ],
            stop=stop,
            resolution=(480, 480),
        ).save(args.render)

    # Compute chamfer
    if args.chamfer is not None:
        logging.info("Calculating chamfer")

        # Cache chamfer
        fname = os.path.join(args.cache_dir, "__temp__chamfer.npz")
        try:
            values = np.load(fname)["values"]
        except:
            values = get_eval_results(
                odb, results_list, args.threads, eval.chamfer, min(1, args.topk)
            )
            np.savez(fname, values=values)

        # Get total chamfer
        with open(args.chamfer, "w") as csv_file:
            val_list = values[:, 0]
            val_list = val_list[~np.isnan(val_list)]
            csv_file.write("Mean Chamfer: {}\n".format(val_list.mean()))
            for obj, r in zip(odb.query_objects, values):
                csv_file.write("{}, {} ".format(obj.true_class, obj.object_id))
                for c in r:
                    csv_file.write("{}, ".format(c))
                csv_file.write("\n")

        # Get mean class scores
        mean_chamfer_by_class = {}
        for o, v in zip(odb.query_objects, values):
            # Just compute for top1
            v = v[0]

            if o.true_class not in mean_chamfer_by_class:
                mean_chamfer_by_class[o.true_class] = [v]
            else:
                mean_chamfer_by_class[o.true_class].append(v)

        with open(os.path.splitext(args.chamfer)[0] + "_mean.csv", "w") as csv_file:
            for k, v in mean_chamfer_by_class.items():
                v = np.array(v)
                csv_file.write(
                    "{}, {} \n".format(
                        shapenet.class2string(k), v[np.logical_not(np.isnan(v))].mean()
                    )
                )

    # Compute normal consistency
    if args.norm_const is not None:
        logging.info("Calculating normal consistency")
        fname = os.path.join(args.cache_dir, "__temp__normal_const.npz")
        try:
            values = np.load(fname)["values"]
        except:
            values = get_eval_results(
                odb,
                results_list,
                args.threads,
                eval.normal_consistency,
                min(1, args.topk),
            )
            np.savez(fname, values=values)

        with open(args.norm_const, "w") as csv_file:
            val_list = values[:, 0]
            val_list = val_list[~np.isnan(val_list)]
            csv_file.write("Mean Normal Consistency: {}\n".format(val_list.mean()))
            for obj, r in zip(odb.query_objects, values):
                csv_file.write("{}, {} ".format(obj.true_class, obj.object_id))
                for c in r:
                    csv_file.write("{}, ".format(c))
                csv_file.write("\n")

    # Extract actual values
    if args.values is not None:
        logging.info("Extracting values")
        with open(args.values, "w") as csv_file:
            for obj, r in zip(odb.query_objects, results_list):
                csv_file.write("{}, {}".format(obj.true_class, obj.object_id))
                for c in r:
                    csv_file.write("{}, ".format(c[1]))
                csv_file.write("\n")

    if args.map:
        logging.info("Computing mAP")

        # Get all class ids
        all_class_ids = set()
        for o in odb.query_objects:
            # Dont consider unknown classes
            if o.true_class != "00000000":
                all_class_ids.add(o.true_class)
        all_class_ids = list(all_class_ids)

        results_list = [
            r
            for r, o in zip(results_list, odb.query_objects)
            if o.true_class != "00000000"
        ]
        known_query_objects = [
            o for o in odb.query_objects if o.true_class != "00000000"
        ]

        args.topk = 3

        # Build the integer-based retrieval matrix
        retreival_matrix = np.ones((len(known_query_objects), args.topk + 1)) * np.nan
        for o in range(len(known_query_objects)):
            # Store the class number, not the class string
            retreival_matrix[o, 0] = all_class_ids.index(
                known_query_objects[o].true_class
            )

            # Get all retrieved object classes
            for k in range(args.topk):
                if k < results_list[o].shape[0]:
                    try:
                        retreival_matrix[o, k + 1] = all_class_ids.index(
                            odb.database_objects[int(results_list[o][k, 0])].true_class
                        )
                    except ValueError:
                        retreival_matrix[o, k + 1] = -1

        # precision and recall are of shape (k, n_classes)
        precision, recall = analysis.get_retreival_precision_recall(
            retreival_matrix, len(all_class_ids)
        )
        precision = precision.T
        recall = recall.T

        with open(args.map, "w") as csv_file:
            csv_file.write("Mean Precision: {}\n".format(precision[:, 0].mean()))
            csv_file.write("Class, Precision, Recall\n")
            for c, p, r in zip(all_class_ids, precision, recall):
                csv_file.write(
                    "{}, {}, {}\n".format(shapenet.class2string(c), p[0], r[0])
                )

        with open(args.map + "_means.csv", "w") as csv_file:
            # csv_file.write("Mean Precision: {}\n".format(precision[:,0].mean()))
            csv_file.write("Precision, ")
            for p in precision.mean(axis=0):
                csv_file.write("{}, ".format(p))
            csv_file.write("\n")

            csv_file.write("Recall, ")
            for r in recall.mean(axis=0):
                csv_file.write("{}, ".format(r))
            csv_file.write("\n")

    if args.pr_curve is not None:
        logging.info("Building pr curve")

        # Get all class ids
        all_class_ids = set()
        for o in odb.query_objects:
            if o.true_class != "00000000":
                all_class_ids.add(o.true_class)
        all_class_ids = list(all_class_ids)

        known_query_objects = [
            o for o in odb.query_objects if o.true_class != "00000000"
        ]

        # Build the integer-based retrieval matrix
        retreival_matrix = np.ones((len(known_query_objects), args.topk + 1)) * np.nan
        for o in range(len(known_query_objects)):
            # Store the class number, not the class string
            retreival_matrix[o, 0] = all_class_ids.index(
                known_query_objects[o].true_class
            )

            # Get all retrieved object classes
            for k in range(args.topk):
                if k < results_list[o].shape[0]:
                    try:
                        retreival_matrix[o, k + 1] = all_class_ids.index(
                            odb.database_objects[int(results_list[o][k, 0])].true_class
                        )
                    except ValueError:
                        retreival_matrix[o, k + 1] = -1

        precision, recall = analysis.get_retreival_precision_recall(
            retreival_matrix, len(all_class_ids)
        )

        precision = np.vstack(
            (
                np.expand_dims(precision[0, :], axis=0),
                precision,
                np.zeros((1, precision.shape[1])),
            )
        )
        recall = np.vstack(
            (
                np.zeros((1, recall.shape[1])),
                recall,
                np.expand_dims(recall[-1, :], axis=0),
            )
        )

        plt.figure(figsize=(8, 5), dpi=1200)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])

        pruned_precision = []
        pruned_recall = []
        pruned_class_list = []
        for p, r, c in zip(precision.T, recall.T, all_class_ids):
            if (p.sum() != 0) and (r.sum() != 0):
                pruned_precision.append(p)
                pruned_recall.append(r)
                pruned_class_list.append(c)

        # Take the mean over all classes
        plt.plot(np.array(pruned_recall).T, np.array(pruned_precision).T)
        plt.legend(
            [shapenet.class2string(c) for c in pruned_class_list],
            bbox_to_anchor=(1.2, 0.5),
            loc="center",
        )
        plt.tight_layout()
        plt.savefig(args.pr_curve)
        plt.savefig("test.png")
