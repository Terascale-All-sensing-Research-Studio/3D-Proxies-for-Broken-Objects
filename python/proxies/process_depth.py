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
import logging

from PIL import Image

import trimesh
import numpy as np

import proxies.errors as errors


def render_depth(mesh, resolution=(480, 640), mode="L", angle_y=0, angle_x=-35.0):

    # scene will have automatically generated camera and lights
    try:
        scene = mesh.scene()
    except AttributeError:
        raise errors.MeshNotClosedError

    # Get the initial camera transform (identity)
    camera_old = np.eye(4)

    # Move the camera back a little
    mat = trimesh.transformations.translation_matrix([0, 0, 1.5])
    camera_old = np.dot(mat, camera_old)

    # Orient the camera so its facing slightly down
    mat = trimesh.transformations.rotation_matrix(
        angle=np.radians(angle_x), direction=[1, 0, 0], point=scene.centroid
    )
    camera_old = np.dot(mat, camera_old)

    mat = trimesh.transformations.rotation_matrix(
        angle=np.radians(angle_y), direction=[0, 1, 0], point=scene.centroid
    )
    camera_old = np.dot(mat, camera_old)

    # Apply the transform
    scene.graph[scene.camera.name] = camera_old

    # set resolution, in pixels
    scene.camera.resolution = list(resolution)

    # make it relative to resolution so pixels per degree is same
    scene.camera.fov = 60 * (scene.camera.resolution / scene.camera.resolution.max())

    # convert the camera to rays with one ray per pixel
    origins, vectors, pixels = scene.camera_rays()

    # do the actual ray- mesh queries
    points, index_ray, _ = mesh.ray.intersects_location(
        origins, vectors, multiple_hits=False
    )

    # for each hit, find the distance along its vector
    depth = trimesh.util.diagonal_dot(points - origins[0], vectors[index_ray])
    # find pixel locations of actual hits
    pixel_ray = pixels[index_ray]

    # create a numpy array we can turn into an image
    # doing it with uint8 creates an `L` mode greyscale image
    a = np.zeros(scene.camera.resolution, dtype=np.uint8)

    # scale depth against range (0.0 - 1.0)
    depth_float = (depth - depth.min()) / depth.ptp()

    # convert depth into 0 - 255 uint8
    depth_int = (depth_float * 255).round().astype(np.uint8)
    # assign depth to correct pixel locations
    a[pixel_ray[:, 0], pixel_ray[:, 1]] = depth_int

    return np.array(Image.fromarray(a).convert(mode))


def process(f_in, f_out, angle):
    mesh = trimesh.load(f_in)
    img = render_depth(mesh, resolution=[640, 640], mode="RGB", angle_y=angle)
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
