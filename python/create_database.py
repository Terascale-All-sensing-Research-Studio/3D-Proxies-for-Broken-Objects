import argparse, os

import proxies.logger as logger
import proxies.features as features
from proxies.database import ObjectDatabase


if __name__ == "__main__":
    feat_list = [
        "global_FCGF",
        "global_eDSIFT",
        "global_POINTNET",
        "global_VGG",
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
        "--add_query",
        default=False,
        action="store_true",
        help="If passed will add query to database.",
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
        add_query=args.add_query,
        feature_sizes=feat_sizes,
        feature_types=feat_types,
    )
    odb.poluate_metadata()
    odb.populate()
    odb.populate_query()
    odb.train()

    # Save the database
    odb.save(args.save)
