def get_feature_size(feature_type):
    if feature_type == "SIFT":
        return 128
    if feature_type == "ORB":
        return 32
    if feature_type == "global_VGG":
        return 4096
    if feature_type == "SHOT":
        return 352
    if feature_type == "PFH":
        return 125
    if feature_type == "global_SHOT":
        return 352
    if feature_type == "global_PFH":
        return 125
    if feature_type == "global_POINTNET":
        return 256
    if feature_type == "SIFT_PANO":
        return 256
    if feature_type == "global_eDSIFT":
        return 80
    if feature_type == "FCGF":
        return 32
    if feature_type == "global_FCGF":
        return 256
    raise RuntimeError("Unknown feature type: {}".format(feature_type))


def path2feature_type(path):
    if "global_fcgf." in path:
        return "global_FCGF"
    if "fcgf" in path:
        return "FCGF"
    if "edsift" in path:
        return "global_eDSIFT"
    if "sift" in path and "pano" in path:
        return "SIFT_PANO"
    if "sift" in path:
        return "SIFT"
    if "orb" in path:
        return "ORB"
    if "global" in path and "vgg" in path:
        return "global_VGG"
    if "global" in path and "shot" in path:
        return "global_SHOT"
    if "shot" in path:
        return "SHOT"
    if "global" in path and "pfh" in path:
        return "global_PFH"
    if "shot" in path:
        return "PFH"
    if "pointnet" in path:
        return "global_POINTNET"
    raise RuntimeError("Could not infer type from path: {}".format(path))


def path2feature_size(path):
    return get_feature_size(path2feature_type(path))


def verify_types(feat_types):
    return feat_types, [get_feature_size(f) for f in feat_types]


def is_2d_type(feat_type):
    if (feat_type == "SIFT") or (feat_type == "ORB") or (feat_type == "global_VGG"):
        return True
    return False


def is_3d_type(feat_type):
    return not is_2d_type(feat_type)


def is_global_type(feat_type):
    if "global" in feat_type:
        return True
    return False


def is_local_type(feat_type):
    return not is_global_type(feat_type)
