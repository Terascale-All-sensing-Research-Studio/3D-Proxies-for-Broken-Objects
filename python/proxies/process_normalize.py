import argparse

import trimesh
import numpy as np

import proxies.errors as errors


def add_normals(f_in, f_out):
    mesh = trimesh.load(f_in)
    mesh.vertex_normals
    mesh.export(f_out)


def normalize(mesh):
    # Get the overall size of the object
    mesh_min, mesh_max = np.min(mesh.vertices, axis=0), np.max(mesh.vertices, axis=0)
    size = mesh_max - mesh_min

    # Center the object
    vertices = mesh.vertices - ((size / 2.0) + mesh_min)

    # Normalize scale of the object
    vertices = vertices * (1.0 / np.max(size))

    return trimesh.Trimesh(
        vertices=vertices,
        faces=mesh.faces,
        vertex_colors=mesh.visual.vertex_colors,
    )


def process(f_in, f_out):
    """
    Given a mesh, will normalize and smooth that mesh, then save. Will
    throw RuntimeError if the mesh cannot be smoothed or if the mesh is
    not watertight after smoothing.
    """
    # Load mesh
    mesh = trimesh.load(f_in)
    try:
        if isinstance(mesh, trimesh.Scene):
            mesh = []
            for m in mesh.geometry.values():
                if isinstance(m.visual, trimesh.visual.color.ColorVisuals):
                    mesh.append(
                        trimesh.Trimesh(
                            vertices=m.vertices,
                            faces=m.faces,
                            vertex_colors=m.visual.vertex_colors,
                        )
                    )
                else:
                    mesh.append(trimesh.Trimesh(vertices=m.vertices, faces=m.faces))
            mesh = trimesh.util.concatenate(mesh)
    except IndexError:
        raise errors.MeshEmptyError

    if len(mesh.vertices) > 1000000:
        raise errors.MeshSizeError

    if not mesh.is_watertight:
        raise errors.MeshNotClosedError

    # mesh = normalize(mesh) # Mesh should already be normalized
    mesh = trimesh.smoothing.filter_laplacian(mesh)

    # Do not remove this - forces vertex normals to be generated
    mesh.vertex_normals
    mesh.export(f_out)


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
    val = process(f_in=args.input, f_out=args.output)
