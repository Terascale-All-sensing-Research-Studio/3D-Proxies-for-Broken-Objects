import argparse, os
import sys

import torch
import trimesh
import numpy as np

import proxies.errors as errors

sys.path.append(os.environ["FCGFPATH"])

from model.resunet import ResUNetBN2C
from util.misc import extract_features


CLASSIFIER = None


def get_network():
    global CLASSIFIER

    if CLASSIFIER is not None:
        return CLASSIFIER

    root_dir = os.path.join(
        os.environ["FCGFPATH"],
        "ResUNetBN2C-16feat-3conv.pth",
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

    feature = feature.cpu().numpy()
    np.savez(f_out, features=feature)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="The obj file to process.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="The feature file to generate (txt).",
    )
    args = parser.parse_args()

    # Process
    process(
        f_in=args.input,
        f_out=args.output,
    )
