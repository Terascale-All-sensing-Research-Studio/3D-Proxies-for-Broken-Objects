import argparse, os
import logging

import pymesh
import trimesh
import numpy as np

import proxies.errors as errors


def paint_mesh(mesh_from, mesh_to, vertex_inds):

    # Vertices to transfer color to
    vertices = mesh_to.vertices[vertex_inds, :]

    # Find their nearest neighbor on source mesh
    _, v_idx = mesh_from.kdtree.query(vertices)

    # Standin color is white, opaque
    mesh_to.visual.vertex_colors = (
        np.ones((mesh_to.vertices.shape[0], 4)).astype(np.uint8) * 255
    )

    # Transfer the colors
    mesh_to.visual.vertex_colors[vertex_inds, :] = mesh_from.visual.vertex_colors[
        v_idx, :
    ]


def intersect_mesh(a, b, sig=5):
    """get mask of vertices in a occurring in both a and b, to apply to a"""
    av = [frozenset(np.round(v, sig)) for v in a]
    bv = set([frozenset(np.round(v, sig)) for v in b])
    return np.asarray(list(map(lambda v: v in bv, av)))


def break_mesh(
    mesh, tool=None, offset=0.0, rand_translation=0.1, noise=0.005, replicator=None, return_tool=False,
):
    """
    Create a break and return the break and the restoration part. Pass in
    a replicator dict to replicate a previous break.
    """

    if replicator is None:
        replicator = {}

    tool_type = replicator.setdefault("tool_type", np.random.randint(1, high=5))
    if tool_type == 1:
        tool = pymesh.generate_box_mesh(
            box_min=[-0.5, -0.5, -0.5], box_max=[0.5, 0.5, 0.5], subdiv_order=6
        )
    else:
        if tool_type == 2:
            tool = pymesh.generate_icosphere(0.5, [0.0, 0.0, 0.0], refinement_order=0)
        elif tool_type == 3:
            tool = pymesh.generate_icosphere(0.5, [0.0, 0.0, 0.0], refinement_order=1)
        elif tool_type == 4:
            tool = pymesh.generate_icosphere(0.5, [0.0, 0.0, 0.0], refinement_order=2)

        # Disjoint the vertices so that the icosphere isn't regular
        random_disjoint = replicator.setdefault(
            "random_disjoint", np.random.random(tool.vertices.shape)
        )
        tool = pymesh.form_mesh(
            tool.vertices + (random_disjoint * (0.1) - (0.1 / 2)), tool.faces
        )

        # Subdivide the mesh
        tool, __ = pymesh.split_long_edges(tool, noise * 5)
    vertices = tool.vertices

    # Offset the tool so that the break is roughly in the center
    set_offset = replicator.setdefault("set_offset", np.array([0.5 + offset, 0, 0]))
    vertices = vertices + set_offset

    # Add random noise to simulate fracture geometry
    noise = np.asarray([noise, noise, noise])
    random_noise = replicator.setdefault(
        "random_noise", np.random.random(vertices.shape)
    )
    vertices = vertices + (random_noise * (noise) - (noise / 2))

    # Add a random rotation
    # http://planning.cs.uiuc.edu/node198.html
    u, v, w = replicator.setdefault("random_rotation", np.random.random(3))
    q = [
        np.sqrt(1 - u) * np.sin(2 * np.pi * v),
        np.sqrt(1 - u) * np.cos(2 * np.pi * v),
        np.sqrt(u) * np.sin(2 * np.pi * w),
        np.sqrt(u) * np.cos(2 * np.pi * v),
    ]
    vertices = np.dot(pymesh.Quaternion(q).to_matrix(), vertices.T).T

    # Add a small random translation
    random_translation = replicator.setdefault(
        "random_translation", np.random.random(3)
    )
    vertices += random_translation * (rand_translation) - (rand_translation / 2.0)

    # Add a warp
    warp = lambda vs: np.asarray([(v ** 3) for v in vs])
    vertices += np.apply_along_axis(warp, 1, vertices)

    # Break
    broken = pymesh.boolean(mesh, pymesh.form_mesh(vertices, tool.faces), "difference")
    restoration = pymesh.boolean(
        mesh, pymesh.form_mesh(vertices, tool.faces), "intersection"
    )

    trimesh.Trimesh(vertices=tool.vertices, faces=tool.faces).export(
        "/opt/data/tsp/model1/tool1.obj"
    )
    if return_tool:
        return broken, restoration, replicator, tool
    return broken, restoration, replicator


def process(
    f_in,
    f_out,
    f_restoration=False,
    validate=True,
    save_meta=False,
    cache=True,
    max_break=0.5,
    min_break=0.3,
):

    samples = 1  # Code adapted to create multiple samples at once
    cur_sample = 0
    cur_retry = 0
    max_overall_retries = 3
    max_single_retries = 5
    offset = 0.5 - ((max_break + min_break) / 2.0) * 0.85
    saver = []
    # print('initial offset {}'.format(offset))

    # Aparently these two libraries do not play well together
    tri_mesh_in = trimesh.load(f_in)
    mesh_in = pymesh.form_mesh(tri_mesh_in.vertices, tri_mesh_in.faces)

    if (not mesh_in.is_manifold()) or (not mesh_in.is_closed()):
        raise errors.MeshNotClosedError

    while (cur_sample < samples) and (cur_retry < max_overall_retries):
        # Here's how you load the replicator
        # replicator = dict(np.load(os.path.join(os.path.dirname(f_out), 'model_broken_{}.npz'.format(cur_sample))))

        # Break
        mesh_out, rmesh_out, replicator, tool_mesh = break_mesh(
            mesh_in, replicator=None, offset=offset, return_tool=True
        )

        # Check to make sure enough of the object was removed
        for _ in range(max_single_retries):
            volume_removed = rmesh_out.volume / mesh_in.volume
            logging.debug("Removed \%{} of the mesh".format(volume_removed))
            if volume_removed < min_break:
                # print('Bad break ratio: {}'.format(volume_removed))
                replicator["set_offset"][0] -= 0.05
            elif volume_removed > max_break:
                # print('Bad break ratio: {}'.format(volume_removed))
                replicator["set_offset"][0] += 0.05
            else:
                break

            # Retry the break
            mesh_out, rmesh_out, replicator, tool_mesh = break_mesh(
                mesh_in, replicator=replicator, offset=offset, return_tool=True
            )
        else:
            cur_retry += 1
            continue

        # Perform output validation
        if validate:
            # We removed all of the vertices, or no vertices
            if (len(mesh_out.vertices) == 0) or (
                len(mesh_out.vertices) == len(mesh_in.vertices)
            ):
                cur_retry += 1
                logging.debug("Mesh validation failed, all or no vertices removed")
                continue

            # This shouldn't happen
            elif (
                (not mesh_out.is_manifold())
                or (not mesh_out.is_closed())
                or (not rmesh_out.is_manifold())
                or (not rmesh_out.is_closed())
            ):
                cur_retry += 1
                logging.debug("Mesh validation failed, result is not waterproof")
                continue

        # Save metadata
        if save_meta:
            fracture_inds = np.logical_not(
                intersect_mesh(mesh_out.vertices, mesh_in.vertices)
            )
            saver.append(
                (
                    np.savez_compressed,
                    {
                        "file": os.path.splitext(f_out)[0] + ".npz",
                        "fracture_vertices": mesh_out.vertices[fracture_inds, :],
                        **replicator,
                    },
                ),
            )

        # Here's how you get the fracture region only
        # mesh_out = mask_mesh(mesh_out, np.logical_not(intersect_mesh(mesh_out.vertices, mesh_in.vertices)))

        # Save the mesh
        tri_mesh_out_b = trimesh.Trimesh(
            vertices=mesh_out.vertices, faces=mesh_out.faces
        )
        paint_mesh(
            tri_mesh_in,
            tri_mesh_out_b,
            intersect_mesh(tri_mesh_out_b.vertices, tri_mesh_in.vertices),
        )

        # tri_mesh_out_b.export("fig2/broken.obj")
        # tool_mesh = trimesh.Trimesh(vertices=tool_mesh.vertices, faces=tool_mesh.faces)
        # tool_mesh.export("fig2/tool.obj")
        # exit()
        # Do not remove this - forces vertex normals to be generated
        tri_mesh_out_b.vertex_normals
        saver.append(
            (tri_mesh_out_b.export, {"file_obj": f_out}),
        )
        if f_restoration:

            tri_mesh_out_r = trimesh.Trimesh(
                vertices=rmesh_out.vertices, faces=rmesh_out.faces
            )
            paint_mesh(
                tri_mesh_in,
                tri_mesh_out_r,
                intersect_mesh(tri_mesh_out_r.vertices, tri_mesh_in.vertices),
            )
            # Do not remove this - forces vertex normals to be generated
            tri_mesh_out_r.vertex_normals
            saver.append(
                (tri_mesh_out_r.export, {"file_obj": f_restoration}),
            )
        cur_retry = 0
        cur_sample += 1

        # Dump the cache on every iteration
        if not cache:
            for fn, kw in saver:
                fn(**kw)
            saver = []

    if cur_sample != samples:
        raise errors.MeshBreakMaxRetriesError
    # Dump the cache
    else:
        for fn, kw in saver:
            fn(**kw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Breaks an object. To allow "
        + "for easy parallelization, the output file name will be appended "
        + "with the break number, assuming there are multiple breaks."
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Input file to break."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output broken file. Will be appended with break number.",
    )
    parser.add_argument(
        "--restoration",
        "-r",
        type=str,
        default=False,
        help="Optionally output restoration file. Will be appended with break "
        + "number.",
    )
    parser.add_argument(
        "--skip_validate",
        "-v",
        action="store_false",
        help="If passed will skip checking if object is watertight.",
    )
    parser.add_argument(
        "--max_break",
        type=float,
        default=1.0,
        help="Max amount of the object to remove (by volume).",
    )
    parser.add_argument(
        "--min_break",
        type=float,
        default=0.0,
        help="Min amount of the object to remove (by volume).",
    )
    parser.add_argument(
        "--meta",
        "-m",
        action="store_true",
        help="If passed will store the fracture vertices in a npz file.",
    )
    parser.add_argument(
        "--dump",
        "-d",
        action="store_true",
        help="If passed will disable object caching. Objects will be written "
        + "if they are watertight regardless of retries.",
    )
    args = parser.parse_args()

    val = process(
        f_in=args.input,
        f_out=args.output,
        f_restoration=args.restoration,
        validate=args.skip_validate,
        save_meta=args.meta,
        cache=(not args.dump),
        max_break=args.max_break,
        min_break=args.min_break,
    )
