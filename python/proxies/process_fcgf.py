import argparse, os
import importlib

import sys

sys.path.append("../../FCGF")

import torch
import trimesh
import numpy as np

import proxies.errors as errors

from model.resunet import ResUNetBN2C
from util.misc import extract_features

CLASSIFIER = None


def get_network():
    global CLASSIFIER

    if CLASSIFIER is not None:
        return CLASSIFIER

    # Hardcoded path to where model is stored
    root_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../FCGF/ResUNetBN2C-16feat-3conv.pth",
    )

    checkpoint = torch.load(root_dir)
    config = checkpoint["config"]
    model = ResUNetBN2C(
        1,
        config.model_n_out,
        normalize_feature=config.normalize_feature,
        conv1_kernel_size=config.conv1_kernel_size,
        D=3,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    CLASSIFIER = model.cuda()
    return CLASSIFIER


def process(f_in, f_out):
    model = trimesh.load(f_in)
    if not model.is_watertight:
        raise errors.MeshNotClosedError

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    network = get_network()
    with torch.no_grad():
        _, feature = extract_features(
            network,
            xyz=np.array(model.vertices),
            voxel_size=0.025,
            device=device,
            skip_check=True,
        )

    new_root = "/home/lambne/dev/data2/tsp/cultural_heritage_overflow"
    try:
        feature = feature.cpu().numpy()
        np.savez(f_out, features=feature)
    except PermissionError:
        new_save = new_root + f_out.split("cultural_heritage")[-1]
        print(new_save)

        if not os.path.isdir(os.path.dirname(new_save)):
            os.makedirs(os.path.dirname(new_save))
        np.savez(new_save, features=feature)
