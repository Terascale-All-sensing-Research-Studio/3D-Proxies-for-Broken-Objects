import argparse

import trimesh
import numpy as np
from scipy.io import savemat

from libmesh import check_mesh_contains

import proxies.errors as errors


def write_ptcld_ply(f_out, vertices, color=None, cmap=None):
    """I couldnt find a library to write out ply files as ascii. What?"""
    vertices = np.round(vertices, decimals=6)

    if color is None:
        with open(f_out, "w") as f:
            f.write("ply \n")
            f.write("format ascii 1.0 \n")
            f.write("element vertex {} \n".format(vertices.shape[0]))
            f.write("property float x \n")
            f.write("property float y \n")
            f.write("property float z \n")
            f.write("element face 0 \n")
            f.write("property list uchar int vertex_indices \n")
            f.write("end_header \n")
            for vertex in vertices:
                f.write("{} {} {} \n".format(vertex[0], vertex[1], vertex[2]))
    else:
        if cmap is None:
            from matplotlib import cm

            cmap = cm.jet
        color = (((color - np.min(color)) / np.max(color)) * 255).astype(np.int)
        color = (cmap(color)[:, :3] * 255).astype(np.int)
        with open(f_out, "w") as f:
            f.write("ply \n")
            f.write("format ascii 1.0 \n")
            f.write("element vertex {} \n".format(vertices.shape[0]))
            f.write("property float x \n")
            f.write("property float y \n")
            f.write("property float z \n")
            f.write("property uchar red \n")
            f.write("property uchar green \n")
            f.write("property uchar blue \n")
            f.write("element face 0 \n")
            f.write("property list uchar int vertex_indices \n")
            f.write("end_header \n")
            for (r, g, b), vertex in zip(color, vertices):
                f.write(
                    "{} {} {} {} {} {} \n".format(
                        vertex[0], vertex[1], vertex[2], r, g, b
                    )
                )


def sample_points(
    mesh,
    mask=None,
    mask_uniform=False,
    n_points=100000,
    uniform_ratio=0.5,
    padding=0.1,
    sigma=0.01,
):
    def apply_mask(points, vertices, fracture_mask):
        # For each surface point, find the closest mesh point
        tree = KDTree(vertices)
        _, ind = tree.query(points, k=1)

        # If that point is on the fracture, throw it out
        del_list = np.in1d(ind.flatten(), fracture_mask)
        return points[np.logical_not(del_list), :]

    # Compute more sample points than we need, in case we throw some out
    overshoot = 1.0
    if mask is not None:
        overshoot = 1.5

    # Compute number of surface and uniform points
    n_points_uniform = int(n_points * float(uniform_ratio))
    n_points_surface = n_points - n_points_uniform

    points_surface, points_uniform = np.empty((0, 3)), np.empty((0, 3))
    while (points_surface.shape[0] < n_points_surface) or (
        points_uniform.shape[0] < n_points_uniform
    ):

        # Generate uniform points
        boxsize = 1 + padding
        pts_to_sample = int(n_points_uniform * overshoot)
        points_uniform = np.vstack(
            (points_uniform, boxsize * (np.random.rand(pts_to_sample, 3) - 0.5))
        )

        # Handle multiple meshes
        if isinstance(mesh, list):
            pts_to_sample = int((n_points_surface / len(mesh)) * overshoot)
            points_surface = np.vstack(
                (points_surface, np.vstack([m.sample(pts_to_sample) for m in mesh]))
            )
            vertices = np.concatenate([m.vertices for m in mesh], axis=0)
        else:
            pts_to_sample = int(n_points_surface * overshoot)
            points_surface = np.vstack((points_surface, mesh.sample(pts_to_sample)))
            vertices = mesh.vertices

        # Remove any unwanted faces
        if mask is not None:
            # Obtain a mask by directly comparing the vertex values
            fracture_mask = np.where(
                intersect_mesh(vertices, np.load(mask)["fracture_vertices"])
            )[0]

            # Apply the mask
            points_surface = apply_mask(points_surface, vertices, fracture_mask)
            if mask_uniform:
                points_uniform = apply_mask(points_uniform, vertices, fracture_mask)

    # Finally, apply sigma to the surface points
    points_surface += sigma * np.random.randn(points_surface.shape[0], 3)

    # Randomize the points so that excess points are removed fairly
    points_surface = points_surface[np.random.permutation(points_surface.shape[0]), :]
    points_uniform = points_uniform[np.random.permutation(points_uniform.shape[0]), :]

    # If we have too many points, throw them out
    points_surface = points_surface[:n_points_surface, :]
    points_uniform = points_uniform[:n_points_uniform, :]

    return np.vstack([points_surface, points_uniform])


def show_points(points_surface, points_uniform):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = Axes3D(fig)

    n_points = 500
    ax.scatter(
        points_surface[:n_points, 0],
        points_surface[:n_points, 1],
        points_surface[:n_points, 2],
        c="r",
    )
    ax.scatter(
        points_uniform[:n_points, 0],
        points_uniform[:n_points, 1],
        points_uniform[:n_points, 2],
        c="b",
    )
    plt.show()


def process(
    f_in,
    f_out,
    f_samp=None,
    mask=None,
    mask_uniform=False,
    n_points=100000,
    uniform_ratio=0.5,
    padding=0.2,
    sigma=0.01,
    compress=True,
    packbits=False,
    validate=True,
):
    """ """
    mesh = trimesh.load(f_in)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [
                trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
                for m in mesh.geometry.values()
            ]
        )

    if validate:
        if not mesh.is_watertight:
            raise errors.MeshNotClosedError

    if f_samp is None:
        # Get sample points
        points = sample_points(
            mesh=mesh,
            mask=mask,
            mask_uniform=mask_uniform,
            n_points=n_points,
            uniform_ratio=uniform_ratio,
            padding=padding,
            sigma=sigma,
        )
    else:
        points = np.array(trimesh.load(f_samp).vertices)
    assert points.shape[0] == n_points, "Loaded sample points were the wrong size"

    # Get occupancies
    occupancies = check_mesh_contains(mesh, points)
    if sum(occupancies.astype(int)) < (n_points * 0.02):
        # This is a bad sample
        raise errors.MeshEmptyError

    # Compress
    dtype = np.float32
    if compress:
        dtype = np.float16
    points = points.astype(dtype)

    if packbits:
        occupancies = np.packbits(occupancies)
        np.savez(f_out, xyz=points, occ=occupancies, packed_bits=np.array([1]))

    # Save
    np.savez(f_out, xyz=points, occ=occupancies)


def add_sign(f_in, f_out):
    mesh = trimesh.load(f_in)

    resolution = (32, 32, 32)
    padding = 0
    grid_pts = np.meshgrid(
        *[np.linspace(0, 1.0 + (padding * 2), d) - (0.5 + padding) for d in resolution]
    )
    query_pts = np.vstack([p.flatten() for p in grid_pts]).T

    occupancies = check_mesh_contains(mesh, query_pts)

    query_pts = np.vstack([p.flatten() for p in grid_pts]).T
    dist, _ = trimesh.proximity.ProximityQuery(mesh).vertex(query_pts)

    dist = np.clip(dist, 0, 0.1)

    new_dist = []
    for o, p in zip(occupancies, dist):
        if o == 1:
            new_dist.append(-p)
        else:
            new_dist.append(p)

    new_dist = np.array(new_dist).reshape(resolution)
    np.savez(f_out, sdf=new_dist)


def voxelize(f_in, f_out, angle=0, resolution=50, padding=0, save_as_mat=False):

    mesh = trimesh.load(f_in)

    if angle != 0:
        rotate = trimesh.transformations.rotation_matrix(
            angle=np.radians(45), direction=[0, 1, 0]
        )
        mesh.vertices = np.dot(mesh.vertices, rotate[:3, :3])

    if not mesh.is_watertight:
        raise errors.MeshNotClosedError

    dims = [resolution, resolution, resolution]
    grid_pts = np.meshgrid(
        *[np.linspace(0, 1.0 + (padding * 2), d) - (0.5 + padding) for d in dims]
    )
    query_pts = np.vstack([p.flatten() for p in grid_pts]).T

    occupancies = check_mesh_contains(mesh, query_pts)

    if save_as_mat:
        savemat(f_out, {"voxel": occupancies.reshape(dims)})
    else:
        query_pts = query_pts[np.where(occupancies)[0], :]
        write_ptcld_ply(f_out, query_pts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Computes the occupancy values "
        + "for samples points on and around an object. Accepts the arguments "
        + "common for sampling."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input file for which to compute occupancy values.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output file that stores occupancy values (.npz).",
    )
    parser.add_argument(
        "--samples",
        type=str,
        help="Input file that stores sample points to use (.ply).",
    )
    parser.add_argument(
        "--mask",
        "-m",
        type=str,
        help="Optionally specify a mask file containing vertices for which "
        + 'nearby occupancy values should not be computed. "Nearby" is '
        + "determined by a knn search with k=1.",
    )
    parser.add_argument(
        "--uniform",
        "-r",
        type=float,
        default=0.5,
        help="Uniform ratio. eg 1.0 = all uniform points, no surface points.",
    )
    parser.add_argument(
        "--mask_uniform",
        "-u",
        action="store_true",
        default=False,
        help="If passed, will compute mask for uniform points as well as "
        + "surface points. Else will just use mask for surface points.",
    )
    parser.add_argument(
        "--n_points",
        "-n",
        type=int,
        default=100000,
        help="Total number of sample points.",
    )
    parser.add_argument(
        "--padding",
        "-p",
        type=float,
        default=0.2,
        help="Extra padding to add when performing uniform sampling. eg 0 = "
        + "uniform sampling is done in unit cube.",
    )
    parser.add_argument(
        "--sigma",
        "-s",
        type=float,
        default=0.01,
        help="Sigma used to compute surface points perturbation.",
    )
    parser.add_argument(
        "--compress",
        "-c",
        action="store_true",
        default=False,
        help="If passed, will encode the points as float16.",
    )
    parser.add_argument(
        "--packbits",
        action="store_true",
        default=False,
        help="If passed will compress occupancies using packbits.",
    )
    parser.add_argument(
        "--skip_validate",
        action="store_false",
        default=True,
        help="If passed will skip checking if object is watertight.",
    )
    args = parser.parse_args()

    process(
        f_in=args.input,
        f_out=args.output,
        f_samp=args.samples,
        mask=args.mask,
        mask_uniform=args.mask_uniform,
        n_points=args.n_points,
        uniform_ratio=args.uniform,
        padding=args.padding,
        sigma=args.sigma,
        compress=args.compress,
        packbits=args.packbits,
        validate=args.skip_validate,
    )
