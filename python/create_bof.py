import argparse, os
import random

import numpy as np
import tqdm

import proxies.logger as logger
import proxies.features as features
import proxies.shapenet as shapenet
from proxies.bof import BagOfFeatures


if __name__ == "__main__":
    feat_list = [
        "SIFT_PANO",
        "FCGF",
    ]

    parser = argparse.ArgumentParser(
        description="Creates the bag-of-features "
        + "features. This doesn't support images that are not 640x640."
    )
    parser.add_argument(
        type=str,
        dest="input",
        help="Location of the dataset. Should contain precomputed features, "
        + "renders, etc.",
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
        + "maters. Options are: {}.".format(feat_list),
    )
    parser.add_argument(
        "--add_query",
        default=False,
        action="store_true",
        help="If passed will add query to database.",
    )
    parser.add_argument(
        "--scan_split",
        default=None,
        type=str,
        help="Pass an additional scan split to the databaset.",
    )
    logger.add_logger_args(parser)
    args = parser.parse_args()
    logger.configure_logging(args)

    if not isinstance(args.feats, list):
        args.feats = [args.feats]

    # Verify that types are correct, also get sizes
    feat_types, feat_sizes = features.verify_types(args.feats)

    assert len(feat_types) == 1
    feat_type = feat_types[0]

    if feat_type == "SIFT_PANO":
        hist_size = 80
    elif feat_type == "FCGF":
        hist_size = 256
    else:
        raise RuntimeError()

    print("Loading Split")
    train_split, test_split = shapenet.load_split(args.input, args.splits)

    if args.add_query:
        bof_train_split = train_split.copy() + test_split.copy()
    else:
        bof_train_split = train_split.copy()

    if args.scan_split is not None:
        assert os.path.isfile(args.scan_split)

    percent = 0.1

    # Count up all the classes
    class_lister = {}
    for idx, o in enumerate(bof_train_split):
        if o.class_id in class_lister:
            class_lister[o.class_id].append(idx)
        else:
            class_lister[o.class_id] = [idx]

    # Sample from them
    bof_samples = []
    for c in class_lister.keys():
        selected = random.sample(class_lister[c], int(len(class_lister[c]) * percent))
        bof_samples.extend(selected)

    print("Training on {}/{} objects".format(len(bof_samples), len(bof_train_split)))

    print("Loading Features")
    feats = []
    all_feat_size = []
    for idx in tqdm.tqdm(bof_samples):
        o = bof_train_split[idx]
        f = o.load_feature("c", feat_type)
        if f is None:
            continue

        num_feats = f.shape[0]
        feats.append(f)
        all_feat_size.append(f.shape[0])
    print("Mean feat size: {}".format(np.array(all_feat_size).mean()))

    feats = np.vstack(feats)
    print(feats.shape)

    # These are hardcoded, don't change them
    print("Training the BoF")
    if feat_type == "SIFT_PANO":
        hist_size = 80
    elif feat_type == "FCGF":
        hist_size = 256
    else:
        raise RuntimeError()
    bof = BagOfFeatures(feats.shape[1], hist_size)
    bof.train(feats)

    # Only run scan split
    if args.scan_split is not None:
        _, test_split = shapenet.load_split(args.input, args.scan_split)

        print("Encoding scan objects")
        for o in tqdm.tqdm(test_split):
            f = o.load_feature("b", feat_type, return_list=True)
            if f is not None:
                if feat_type == "SIFT_PANO":
                    assert len(f) == 12
                    for idx, f_ in zip(range(12), f):
                        # Compute and save feature
                        np.savez(
                            o.path_feature_b_global_edsift(angle=idx),
                            features=np.expand_dims(bof.encode(f_), axis=0),
                        )
                elif feat_type == "FCGF":
                    np.savez(
                        o.path_feature_b_global_fcgf(),
                        features=np.expand_dims(bof.encode(f[0]), axis=0),
                    )
                else:
                    raise RuntimeError()

            f = o.load_feature("c", feat_type, return_list=True)
            if f is not None:
                if feat_type == "SIFT_PANO":
                    assert len(f) == 12
                    for idx, f_ in zip(range(12), f):
                        # Compute and save feature
                        np.savez(
                            o.path_feature_c_global_edsift(angle=idx),
                            features=np.expand_dims(bof.encode(f_), axis=0),
                        )
                elif feat_type == "FCGF":
                    np.savez(
                        o.path_feature_c_global_fcgf(),
                        features=np.expand_dims(bof.encode(f[0]), axis=0),
                    )
                else:
                    raise RuntimeError()
    else:

        print("Encoding database objects")
        for o in tqdm.tqdm(bof_train_split):
            f = o.load_feature("c", feat_type, return_list=True)
            if f is None:
                continue
            if feat_type == "SIFT_PANO":
                assert len(f) == 12
                for idx, f_ in zip(range(12), f):
                    # Compute and save feature
                    np.savez(
                        o.path_feature_c_global_edsift(angle=idx),
                        features=np.expand_dims(bof.encode(f_), axis=0),
                    )
            elif feat_type == "FCGF":
                np.savez(
                    o.path_feature_c_global_fcgf(),
                    features=np.expand_dims(bof.encode(f[0]), axis=0),
                )
            else:
                raise RuntimeError()

        print("Encoding query objects")
        for o in tqdm.tqdm(test_split):
            f = o.load_feature("b", feat_type, return_list=True)
            if f is not None:
                if feat_type == "SIFT_PANO":
                    assert len(f) == 12
                    for idx, f_ in zip(range(12), f):
                        # Compute and save feature
                        np.savez(
                            o.path_feature_b_global_edsift(angle=idx),
                            features=np.expand_dims(bof.encode(f_), axis=0),
                        )
                elif feat_type == "FCGF":
                    np.savez(
                        o.path_feature_b_global_fcgf(),
                        features=np.expand_dims(bof.encode(f[0]), axis=0),
                    )
                else:
                    raise RuntimeError()

            f = o.load_feature("c", feat_type, return_list=True)
            if f is not None:
                if feat_type == "SIFT_PANO":
                    assert len(f) == 12
                    for idx, f_ in zip(range(12), f):
                        # Compute and save feature
                        np.savez(
                            o.path_feature_c_global_edsift(angle=idx),
                            features=np.expand_dims(bof.encode(f_), axis=0),
                        )
                elif feat_type == "FCGF":
                    np.savez(
                        o.path_feature_c_global_fcgf(),
                        features=np.expand_dims(bof.encode(f[0]), axis=0),
                    )
                else:
                    raise RuntimeError()
