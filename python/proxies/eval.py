import numpy as np

from sklearn.neighbors import KDTree

import proxies.errors as errors


# Evaluaion functions
def chamfer(obj1, obj2, num_pts=10000):
    """
    Computes the symmetric chamfer distance, i.e. the sum of both chamfers.
    """

    obj1_pts = obj1.load_model("c").sample(num_pts)
    obj2_pts = obj2.load_model("c").sample(num_pts)

    assert (
        obj1_pts.shape == obj2_pts.shape
    ), "Can only compare objects that have the same shape"

    one_distances, _ = KDTree(obj2_pts).query(obj1_pts)
    two_distances, _ = KDTree(obj1_pts).query(obj2_pts)

    return (two_distances.mean() + one_distances.mean()) / 2


def iou(obj1, obj2):
    """
    Computes the intersection over union (IOU) for two watertight 3d meshes.
    Note that this requires openscad (or other)
    """

    obj1 = obj1.load_model("c")
    obj2 = obj2.load_model("c")

    return obj1.intersection(obj2).volume() / obj1.union(obj2).volume()


def normal_consistency(obj1, obj2):
    """
    Computes the normal alignment for two 3d meshes.
    """

    obj1 = obj1.load_model("c")
    obj2 = obj2.load_model("c")

    def normal_diff(obj_from, obj_to):

        _, idx = obj_to.kdtree.query(obj_from.vertices)

        # Normalize the vertices, sometimes trimesh doesn't do this
        normals_from = obj_from.vertex_normals / np.linalg.norm(
            obj_from.vertex_normals, axis=-1, keepdims=True
        )
        normals_to = obj_to.vertex_normals / np.linalg.norm(
            obj_to.vertex_normals, axis=-1, keepdims=True
        )

        # Compute the dot product
        return (normals_to[idx] * normals_from).sum(axis=-1)

    return (normal_diff(obj1, obj2).mean() + normal_diff(obj2, obj1).mean()) / 2


def class_score(query_obj, objs_found, topk=1):
    """
    Computes the class score given a query object and a list of returned objects.
    Inputs should be a ShapenetObject and a list of ShapenetObjects.
    """
    return query_obj.true_class in set([o.true_class for o in objs_found][:topk])
