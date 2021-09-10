import argparse
import numpy as np

import trimesh

import proxies.errors as errors


def process(f_in, f_out, n_points=500000, compress=True):

    # Load meshes
    mesh = trimesh.load(f_in)
    try:
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(
                [
                    trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
                    for m in mesh.geometry.values()
                ]
            )
    except IndexError:
        raise errors.MeshEmptyError

    points = mesh.sample(n_points)
    dtype = np.float32
    if compress:
        dtype = np.float16
    points = points.astype(dtype)

    np.savez(f_out, xyz=points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Performs uniform and surface "
        + "sampling on the input object and exports a point cloud."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input file for which to compute sampling values.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output file that stores sample points (.npz)",
    )
    parser.add_argument(
        "--n_points",
        "-n",
        type=int,
        default=30000,
        help="Total number of sample points.",
    )
    args = parser.parse_args()

    sp = process(f_in=args.input, f_out=args.output, n_points=args.n_points)
