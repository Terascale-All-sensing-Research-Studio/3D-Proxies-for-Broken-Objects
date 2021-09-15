import argparse, os
from collections import defaultdict

import proxies.logger as logger
import proxies.features as features
from proxies.database import ObjectDatabase


if __name__ == "__main__":
    feat_list = [
        "global_VGG",
        "global_POINTNET",
    ]

    parser = argparse.ArgumentParser(description="Creates a multi-feature database.")
    parser.add_argument(
        dest="root_dir",
        type=str,
        default=None,
        help="Directory that stores the data on disk. In the case of ShapeNet "
        + "this is '/home/.../ShapeNetCore.v2'. This does not have to be passed "
        + "if using fully populated database.",
    )
    parser.add_argument(
        dest="splits",
        type=str,
        default=None,
        help="Splits file.",
    )
    parser.add_argument(
        dest="feats",
        type=str,
        nargs="+",
        default=feat_list,
        help="A list of features to use. The order that you input these "
        + "is important. Options are: {}.".format(feat_list),
    )
    parser.add_argument(
        "--save",
        type=str,
        default="index",
        help="Path to directory containing saved database. This directory will "
        + "contain several files.",
    )
    parser.add_argument(
        "--db_max",
        type=int,
        default=8,
        help="Max instances of each class to use.",
    )
    parser.add_argument(
        "--q_max",
        type=int,
        default=2,
        help="Min instances of each class to use.",
    )
    parser.add_argument(
        "--render_resolution",
        default=[640, 640],
        type=int,
        nargs="+",
        help="",
    )
    parser.add_argument(
        "--pointcloud_noise",
        default=0.0,
        type=float,
        help="",
    )
    parser.add_argument(
        "--no_normals",
        default=False,
        action="store_true",
        help="",
    )
    logger.add_logger_args(parser)
    args = parser.parse_args()
    logger.configure_logging(args)

    if not isinstance(args.feats, list):
        args.feats = [args.feats]

    # Verify that types are correct, also get sizes
    feat_types, feat_sizes = features.verify_types(args.feats)

    assert os.path.isdir(os.path.dirname(args.root_dir))
    assert os.path.isdir(os.path.dirname(args.save))

    # Create the database
    odb = ObjectDatabase(
        splits_file=args.splits,
        obj_kwargs={
            "resolution": args.render_resolution,
            "noise": args.pointcloud_noise,
            "normals": (not args.no_normals),
        },
        root_dir=args.root_dir,
        feature_sizes=feat_sizes,
        feature_types=feat_types,
    )

    # Select a subset of the database objects
    new_obj_id_list = []
    class_counter = defaultdict(lambda: 0)
    for obj in odb._database_object_id_list:
        if class_counter[obj.class_id] < args.db_max:
            class_counter[obj.class_id] += 1
            new_obj_id_list.append(obj)

    print("Total database objects: {}".format(len(new_obj_id_list)))
    odb._database_object_id_list = new_obj_id_list

    # Select a subset of the query objects
    new_obj_id_list = []
    class_counter = defaultdict(lambda: 0)
    for obj in odb._query_object_id_list:
        if class_counter[obj.class_id] < args.q_max:
            class_counter[obj.class_id] += 1
            new_obj_id_list.append(obj)

    print("Total query objects: {}".format(len(new_obj_id_list)))
    odb._query_object_id_list = new_obj_id_list

    odb.poluate_metadata()
    odb.populate()
    odb.populate_query()
    odb.train()

    # Save the database
    odb.save(args.save)
