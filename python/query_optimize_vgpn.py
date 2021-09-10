import logging
import os
import argparse
import numpy as np
import multiprocessing

import matplotlib.pyplot as plt

from proxies.database import ObjectDatabase
import proxies.errors as errors
import proxies.logger as logger
import proxies.eval as eval


def eval_fn(fn, *args):
    try:
        return fn(*args)
    except errors.MeshNotClosedError:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
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
        "--cache_dir",
        type=str,
        default="__temp",
        help="Results will be dumped to temporary files so that evaluation can "
        + "be performed more quickly. These temporary files will live in this "
        + "directory.",
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default=None,
        help="If passed, will dump output of log to this file.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of threads to use during evaluation.",
    )
    parser.add_argument(
        "--use_gpu",
        default=False,
        action="store_true",
        help="If passed, will perform query on the gpu.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Line search resolution.",
    )
    parser.add_argument(
        "--alpha_start",
        type=float,
        default=0.0,
        help="",
    )
    parser.add_argument(
        "--alpha_end",
        type=float,
        default=1.0,
        help="",
    )
    parser.add_argument(
        "--feat_mask",
        type=int,
        nargs="+",
        help="A binary mask indicating which features to use.",
    )
    parser.add_argument(
        "--chamfer_file",
        type=str,
        default="chamfer.csv",
        help="",
    )
    parser.add_argument(
        "--top_file",
        type=str,
        default="top1_top5.csv",
        help="",
    )
    logger.add_logger_args(parser)
    args = parser.parse_args()
    logger.configure_logging(args)

    assert os.path.isdir(args.cache_dir)

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

    # Double check the feature mask
    if args.feat_mask is None:
        args.feat_mask = [True for _ in range(len(odb._feature_types))]
    assert len(args.feat_mask) == len(odb._feature_types)
    logging.info(
        "Running on {}/{} features".format(sum(args.feat_mask), len(odb._feature_types))
    )

    chamfer_stacker = []
    class_top1_stacker = []
    class_top5_stacker = []

    main_pool = multiprocessing.Pool(args.threads)

    for alpha in np.linspace(args.alpha_start, args.alpha_end, args.steps):
        print("== Testing alpha: {} ==".format(round(alpha, 2)))
        # Set weights
        odb._feature_weights = [(alpha), (1 - alpha)]

        feat_mask = args.feat_mask.copy()
        if odb._feature_weights[0] == 0:
            feat_mask[0] = False
        elif odb._feature_weights[1] == 0:
            feat_mask[1] = False

        # Query
        results_list = odb.hierarchical_query(
            odb.query_objects,
            feat_mask=feat_mask,
            query_cache_fname=os.path.join(args.cache_dir, "__temp__{}_{}.npz"),
        )

        # Chamfer distance
        result_futures = [
            main_pool.apply_async(
                eval_fn,
                args=(
                    obj.eval,
                    eval.chamfer,
                    odb.database_objects[int(r[0, 0])],
                ),
            )
            for obj, r in zip(odb.query_objects, results_list)
        ]

        # Wait
        futures = [r.get() for r in result_futures]

        # Remove any bad samples, get final result
        chamfer_results = np.array([r for r in futures if r is not None])
        chamfer_stacker.append(np.expand_dims(chamfer_results, axis=1))

        # Class score
        top_5_matrix = np.zeros((len(odb.query_objects), 5))
        for o in range(len(odb.query_objects)):
            for k in range(5):
                # Match!
                if k < results_list[o].shape[0]:
                    if (
                        odb.query_objects[o].class_id
                        == odb.database_objects[int(results_list[o][k, 0])].class_id
                    ):
                        top_5_matrix[o, k] = 1

        # Compute cumsum
        top_5_matrix = np.clip(np.cumsum(top_5_matrix, axis=1), 0, 1)

        # Compute top 1 and top 5
        class_top1_stacker.append(top_5_matrix[:, 0].sum())
        class_top5_stacker.append(top_5_matrix[:, 4].sum())

    chamfer_stacker = np.hstack(chamfer_stacker)

    # Write all chamfer distances
    with open(args.chamfer_file, "w") as csv_file:
        csv_file.write(" , ,")
        for alpha in np.linspace(args.alpha_start, args.alpha_end, args.steps):
            csv_file.write("{}, ".format(alpha))
        csv_file.write("\n")

        csv_file.write(" Means, ,")
        for c in chamfer_stacker.mean(axis=0):
            csv_file.write("{}, ".format(c))
        csv_file.write("\n")

        for obj, r in zip(odb.query_objects, chamfer_stacker):
            csv_file.write("{}, {}, ".format(obj.class_id, obj.object_id))
            for c in r:
                csv_file.write("{}, ".format(c))
            csv_file.write("\n")

    # Write top1 and top5
    with open(args.top_file, "w") as csv_file:
        csv_file.write(" ,")
        for alpha in np.linspace(args.alpha_start, args.alpha_end, args.steps):
            csv_file.write("{}, ".format(alpha))
        csv_file.write("\n")

        csv_file.write("top1, ")
        for top1 in class_top1_stacker:
            csv_file.write("{}, ".format(top1))
        csv_file.write("\n")

        csv_file.write("top5, ")
        for top5 in class_top5_stacker:
            csv_file.write("{}, ".format(top5))
