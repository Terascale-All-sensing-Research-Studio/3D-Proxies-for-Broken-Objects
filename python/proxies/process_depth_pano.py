"""
raytrace.py
----------------
A very simple example of using scene cameras to generate
rays for image reasons.
Install `pyembree` for a speedup (600k+ rays per second)
"""

# conda install -c conda-forge pyembree trimesh pyglet tqdm pillow

from __future__ import division

import argparse
import time
import math
import logging

from PIL import Image

import trimesh
import numpy as np

import proxies.errors as errors

import trimesh
import numpy as np


def pts_on_dodecahedron():
    vertices = np.array(
        [
            -0.57735,
            -0.57735,
            0.57735,
            0.934172,
            0.356822,
            0,
            0.934172,
            -0.356822,
            0,
            -0.934172,
            0.356822,
            0,
            -0.934172,
            -0.356822,
            0,
            0,
            0.934172,
            0.356822,
            0,
            0.934172,
            -0.356822,
            0.356822,
            0,
            -0.934172,
            -0.356822,
            0,
            -0.934172,
            0,
            -0.934172,
            -0.356822,
            0,
            -0.934172,
            0.356822,
            0.356822,
            0,
            0.934172,
            -0.356822,
            0,
            0.934172,
            0.57735,
            0.57735,
            -0.57735,
            0.57735,
            0.57735,
            0.57735,
            -0.57735,
            0.57735,
            -0.57735,
            -0.57735,
            0.57735,
            0.57735,
            0.57735,
            -0.57735,
            -0.57735,
            0.57735,
            -0.57735,
            0.57735,
            -0.57735,
            -0.57735,
            -0.57735,
        ]
    ).reshape((-1, 3), order="C")

    faces = (
        np.array(
            [
                19,
                3,
                2,
                12,
                19,
                2,
                15,
                12,
                2,
                8,
                14,
                2,
                18,
                8,
                2,
                3,
                18,
                2,
                20,
                5,
                4,
                9,
                20,
                4,
                16,
                9,
                4,
                13,
                17,
                4,
                1,
                13,
                4,
                5,
                1,
                4,
                7,
                16,
                4,
                6,
                7,
                4,
                17,
                6,
                4,
                6,
                15,
                2,
                7,
                6,
                2,
                14,
                7,
                2,
                10,
                18,
                3,
                11,
                10,
                3,
                19,
                11,
                3,
                11,
                1,
                5,
                10,
                11,
                5,
                20,
                10,
                5,
                20,
                9,
                8,
                10,
                20,
                8,
                18,
                10,
                8,
                9,
                16,
                7,
                8,
                9,
                7,
                14,
                8,
                7,
                12,
                15,
                6,
                13,
                12,
                6,
                17,
                13,
                6,
                13,
                1,
                11,
                12,
                13,
                11,
                19,
                12,
                11,
            ]
        ).reshape((-1, 3), order="C")
        - 1
    )
    m = trimesh.Trimesh(vertices, faces)

    return np.vstack(
        [
            m.vertices[m.faces[i : i + 3, :].flatten(), :].mean(axis=0)
            for i in range(0, m.faces.shape[0], 3)
        ]
    )


def rotation_matrix_from_vectors(vec1, vec2):
    """Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (
        vec2 / np.linalg.norm(vec2)
    ).reshape(3)
    v = np.cross(a, b)
    if any(v):  # if not all zeros then
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    else:
        return np.eye(3)  # cross of all zeros only occurs on identical directions


def redner_depth(mesh, resolution=(200, 628), mode="L", dist=5):

    # Build origin vector list
    origin_stacker = []
    vector_stacker = []
    for h in np.linspace(-0.5, 0.5, resolution[0], endpoint=False):
        for r in np.linspace(0.0, np.pi * 2, resolution[1], endpoint=False):
            origin_stacker.append(np.array([math.sin(r) * dist, h, math.cos(r) * dist]))
            vector_stacker.append(np.array([0, h, 0]) - origin_stacker[-1])
    origins = np.vstack(origin_stacker)
    vectors = np.vstack(vector_stacker)

    # do the actual ray-mesh queries
    pixels = trimesh.util.grid_linspace(
        bounds=[[0, resolution[1] - 1], [resolution[0] - 1, 0]], count=resolution
    ).astype(int)
    points, index_ray, _ = mesh.ray.intersects_location(
        origins, vectors, multiple_hits=False
    )

    # for each hit, find the distance along its vector
    depth = np.sqrt(np.sum((points - origins[index_ray]) ** 2, axis=1))

    # find pixel locations of actual hits
    pixel_ray = pixels[index_ray]

    # scale depth against range (0.0 - 1.0)
    depth_float = (depth - depth.min()) / depth.ptp()

    # convert depth into 0 - 255 uint8
    depth_int = (depth_float * 255).round().astype(np.uint8)

    # create a numpy array we can turn into an image
    # doing it with uint8 creates an `L` mode greyscale image
    depth_map = np.zeros(resolution, dtype=np.uint8)

    # assign depth to correct pixel locations
    depth_map[pixel_ray[:, 0], pixel_ray[:, 1]] = depth_int

    return np.array(Image.fromarray(depth_map).convert(mode))


def process(f_in, f_out, angle=None):
    mesh = trimesh.load(f_in)

    if angle is not None:
        # Apply ddh rotation
        rot = rotation_matrix_from_vectors([0, -1, 0], pts_on_dodecahedron()[angle, :])
        mesh.vertices = np.dot(mesh.vertices, rot)

    img = redner_depth(mesh, mode="RGB", resolution=(128, 256), dist=0.55)
    Image.fromarray(img).save(f_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Normalizes an object. This "
        + "entails scaling it so that it fits inside a unit cube, applying "
        + "smoothing, and checking if the object is watertight."
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Input file to normalize."
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Output normalized file."
    )
    args = parser.parse_args()

    # Process
    val = process(f_in=args.input, f_out=args.output, angle=0)
