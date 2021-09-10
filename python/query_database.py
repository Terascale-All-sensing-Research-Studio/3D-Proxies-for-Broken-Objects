import argparse, os

import logging

from proxies.database import ObjectDatabase
import proxies.logger as logger


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
        default=6000,
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

    # Double check the feature mask
    if args.feat_mask is None:
        args.feat_mask = [True for _ in range(len(odb._feature_types))]
    assert len(args.feat_mask) == len(odb._feature_types)
    logging.info(
        "Running on {}/{} features".format(sum(args.feat_mask), len(odb._feature_types))
    )

    # Set weights
    if len(odb._feature_types) == 2 and "global_POINTNET" in odb.features:
        alpha = 0.5
        odb._feature_weights = [(alpha), (1 - alpha)]
        raise RuntimeError()
    elif len(odb._feature_types) == 2 and "global_FCGF" in odb.features:
        alpha = 0.5
        odb._feature_weights = [(alpha), (1 - alpha)]
        raise RuntimeError()
    elif len(odb._feature_types) == 1:
        odb._feature_weights = [1]
    else:
        raise RuntimeError()

    # Batch and run the query
    results_list = []
    for idx, i in enumerate(range(0, len(odb.query_objects), args.batch_size)):
        results_list.extend(
            odb.hierarchical_query(
                odb.query_objects[
                    i : min(i + args.batch_size, len(odb.query_objects) - 1)
                ],
                feat_mask=args.feat_mask,
                topk=args.topk,
                batch_num=idx,
                query_cache_fname=os.path.join(
                    args.cache_dir, "__temp__realsies__{}_{}.npz"
                ),
            )
        )
