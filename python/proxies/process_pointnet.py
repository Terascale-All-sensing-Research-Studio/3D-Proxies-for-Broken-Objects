import argparse, os, sys
import importlib

import torch
import trimesh
import numpy as np

import proxies.errors as errors


CLASSIFIER = None
POINTNET_PATH = os.environ["POINTNETPATH"]

def get_network(normals):
    global CLASSIFIER

    if CLASSIFIER is not None:
        return CLASSIFIER

    # Hardcoded path to where pointnet is stored
    root_dir = POINTNET_PATH
    sys.path.append(os.path.join(root_dir, "models"))

    if normals:
        model_dir = os.path.join(root_dir, "log/classification/pointnet2_msg_normals")
    else:
        model_dir = os.path.join(root_dir, "log/classification/pointnet2_ssg_wo_normals")

    # Get the loader
    model = importlib.import_module(os.listdir(model_dir + "/logs")[0].split(".")[0])

    # Load the classifier
    CLASSIFIER = model.get_model(40, normal_channel=normals).cuda()

    # Load the weights
    checkpoint = torch.load(model_dir + "/checkpoints/best_model.pth")
    CLASSIFIER.load_state_dict(checkpoint["model_state_dict"])
    return CLASSIFIER.eval()


def process(f_in, f_out, sigma=0.0, use_normals=True):
    model = trimesh.load(f_in)
    if not model.is_watertight:
        raise errors.MeshNotClosedError

    if sigma != 0.0:
        model.vertices += sigma * np.random.randn(model.vertices.shape[0], 3)
    vertices, normals = model.vertices, model.vertex_normals

    pointnet = get_network(use_normals)
    with torch.no_grad():
        if use_normals:
            points = (
                torch.from_numpy(np.hstack((vertices, normals)))
                .unsqueeze(0)
                .type(torch.float)
                .transpose(2, 1)
                .cuda()
            )
        else:
            points = (
                torch.from_numpy(vertices)
                .unsqueeze(0)
                .type(torch.float)
                .transpose(2, 1)
                .cuda()
            )
        feature = pointnet(points).cpu().numpy()
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
    parser.add_argument(
        "--normals",
        "-n",
        action="store_true",
        help="Use normals.",
    )
    args = parser.parse_args()

    # Process
    process(
        f_in=args.input,
        f_out=args.output,
        sigma=0.0,
        use_normals=args.normals,
    )
