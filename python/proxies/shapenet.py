import functools
import logging
import os
import json

import numpy as np

try:
    import cv2
except ImportError:
    pass
try:
    import trimesh
except ImportError:
    pass

import proxies.utils as utils
import proxies.features as features
import proxies.errors as errors

SHAPENET_CLASSES = None
SCANNED_CLASSES = None


class ShapenetObject:
    def __init__(
        self,
        root,
        class_id,
        object_id,
        true_class=None,
        num_renders=1,
        resolution=(640, 640),
        noise=0.0,
        normals=True,
    ):
        self.root = root
        self._class_id = class_id
        self._object_id = object_id
        self._num_renders = num_renders
        self._resolution = tuple(resolution)
        self._noise = noise
        self._normals = normals

        self._feature_c = {}
        self._feature_b = {}
        self._feature_r = {}
        self._eval = {}

        self._true_class = true_class

    def __repr__(self):
        return (
            "ShapenetObject("
            + self.root
            + ", "
            + self._class_id
            + ", "
            + self._object_id
            + ")"
        )

    def __str__(self):
        return self._class_id + ":" + self._object_id

    def build_dirs(self):
        dir_path = os.path.join(self.root, self._class_id, self._object_id, "models")
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        dir_path = os.path.join(self.root, self._class_id, self._object_id, "renders")
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        dir_path = os.path.join(self.root, self._class_id, self._object_id, "features")
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        dir_path = os.path.join(self.root, self._class_id, self._object_id, "evals")
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

    @property
    def class_id(self):
        return self._class_id

    @property
    def object_id(self):
        return self._object_id

    @property
    def noise(self):
        return self._noise

    @property
    def resolution(self):
        return self._resolution

    @property
    def normals(self):
        return self._normals

    @property
    def true_class(self):
        if hasattr(self, "_true_class") and self._true_class is not None:
            if self._true_class == "UNKNOWN":
                return "00000000"
            return self._true_class
        return self.class_id

    @property
    def get_id(self):
        # Don't return root, because root can change between runs
        return [self._class_id, self._object_id]

    @property
    def get_id_str(self):
        # Don't return root, because root can change between runs
        return str(self._class_id) + str(self._object_id)

    def feature_type2path_handle_c(self, feature_type):
        if feature_type == "SIFT":
            return [
                functools.partial(self.path_feature_c_sift, angle=a)
                for a in range(0, 360, int(360 / self._num_renders))
            ]
        if feature_type == "SIFT_PANO":
            return [
                functools.partial(self.path_feature_c_sift_pano, angle=a)
                for a in range(12)
            ]
        if feature_type == "FCGF":
            return self.path_feature_c_fcgf
        if feature_type == "global_FCGF":
            return self.path_feature_c_global_fcgf
        if feature_type == "global_eDSIFT":
            return [
                functools.partial(self.path_feature_c_global_edsift, angle=a)
                for a in range(12)
            ]
        if feature_type == "ORB":
            return [
                functools.partial(self.path_feature_c_orb, angle=a)
                for a in range(0, 360, int(360 / self._num_renders))
            ]
        if feature_type == "global_VGG":
            return [
                functools.partial(self.path_feature_c_global_vgg, angle=a)
                for a in range(0, 360, int(360 / self._num_renders))
            ]
        if feature_type == "PFH":
            return self.path_feature_c_pfh
        if feature_type == "SHOT":
            return self.path_feature_c_shot
        if feature_type == "global_SHOT":
            return [
                self.path_feature_c_global_grid_offset_shot,
                self.path_feature_c_global_axis_cut_shot,
            ]
        if feature_type == "global_PFH":
            return [
                self.path_feature_c_global_grid_offset_pfh,
                self.path_feature_c_global_axis_cut_pfh,
            ]
        if feature_type == "global_POINTNET":
            return self.path_feature_c_global_pointnet
        raise RuntimeError("Unknown feature type: {}".format(feature_type))

    def feature_type2path_handle_b(self, feature_type):
        if feature_type == "SIFT":
            return [
                functools.partial(self.path_feature_b_sift, angle=a)
                for a in range(0, 360, int(360 / self._num_renders))
            ]
        if feature_type == "SIFT_PANO":
            return [
                functools.partial(self.path_feature_b_sift_pano, angle=a)
                for a in range(12)
            ]
        if feature_type == "FCGF":
            return self.path_feature_b_fcgf
        if feature_type == "global_FCGF":
            return self.path_feature_b_global_fcgf
        if feature_type == "global_eDSIFT":
            return [
                functools.partial(self.path_feature_b_global_edsift, angle=a)
                for a in range(12)
            ]
        if feature_type == "ORB":
            return [
                functools.partial(self.path_feature_b_orb, angle=a)
                for a in range(0, 360, int(360 / self._num_renders))
            ]
        if feature_type == "global_VGG":
            return [
                functools.partial(self.path_feature_b_global_vgg, angle=a)
                for a in range(0, 360, int(360 / self._num_renders))
            ]
        if feature_type == "PFH":
            return self.path_feature_b_pfh
        if feature_type == "SHOT":
            return self.path_feature_b_shot
        if feature_type == "global_PFH":
            # return self.path_feature_b_global_pfh
            return [
                self.path_feature_b_global_pfh,
                self.path_feature_b_nofrac_global_pfh,
            ]
        if feature_type == "global_SHOT":
            # return self.path_feature_b_global_shot
            return [
                self.path_feature_b_global_shot,
                self.path_feature_b_nofrac_global_shot,
            ]
        if feature_type == "global_POINTNET":
            return self.path_feature_b_global_pointnet
        raise RuntimeError("Unknown feature type: {}".format(feature_type))

    def feature_type2path_handle_r(self, feature_type):
        if feature_type == "SIFT":
            return [
                functools.partial(self.path_feature_r_sift, angle=a)
                for a in range(0, 360, int(360 / self._num_renders))
            ]
        if feature_type == "SIFT_PANO":
            return [
                functools.partial(self.path_feature_r_sift_pano, angle=a)
                for a in range(12)
            ]
        if feature_type == "FCGF":
            return self.path_feature_r_fcgf
        if feature_type == "global_FCGF":
            return self.path_feature_r_global_fcgf
        if feature_type == "global_eDSIFT":
            return [
                functools.partial(self.path_feature_r_global_edsift, angle=a)
                for a in range(12)
            ]
        if feature_type == "ORB":
            return [
                functools.partial(self.path_feature_r_orb, angle=a)
                for a in range(0, 360, int(360 / self._num_renders))
            ]
        if feature_type == "global_VGG":
            return [
                functools.partial(self.path_feature_r_global_vgg, angle=a)
                for a in range(0, 360, int(360 / self._num_renders))
            ]
        if feature_type == "PFH":
            return self.path_feature_r_pfh
        if feature_type == "SHOT":
            return self.path_feature_r_shot
        if feature_type == "global_PFH":
            # return self.path_feature_r_global_pfh
            return [
                self.path_feature_r_global_pfh,
                self.path_feature_r_nofrac_global_pfh,
            ]
        if feature_type == "global_SHOT":
            # return self.path_feature_r_global_shot
            return [
                self.path_feature_r_global_shot,
                self.path_feature_r_nofrac_global_shot,
            ]
        if feature_type == "global_POINTNET":
            return self.path_feature_r_global_pointnet
        raise RuntimeError("Unknown feature type: {}".format(feature_type))

    # High level getter and setter methods
    def get_feature(self, obj_type, feat_type):
        if obj_type == "c":
            return self._feature_c[feat_type]
        elif obj_type == "b":
            return self._feature_b[feat_type]
        elif obj_type == "r":
            return self._feature_r[feat_type]
        else:
            raise RuntimeError("Unkown obj_type: {}".format(obj_type))

    def set_feature(self, obj_type, feat_type, feature):
        if obj_type == "c":
            self._feature_c[feat_type] = feature
        elif obj_type == "b":
            self._feature_b[feat_type] = feature
        elif obj_type == "r":
            self._feature_r[feat_type] = feature
        else:
            raise RuntimeError("Unkown obj_type: {}".format(obj_type))

    def get_model(self, obj_type):
        if obj_type == "c":
            model = self._model_c
        elif obj_type == "b":
            model = self._model_b
        elif obj_type == "r":
            model = self._model_r
        else:
            raise RuntimeError("Unkown obj_type: {}".format(obj_type))
        if model is None:
            raise errors.MeshNotClosedError
        return model

    def set_model(self, obj_type, model):
        if obj_type == "c":
            self._model_c = model
        elif obj_type == "b":
            self._model_b = model
        elif obj_type == "r":
            self._model_r = model
        else:
            raise RuntimeError("Unkown obj_type: {}".format(obj_type))

    def get_points(self, obj_type):
        if obj_type == "c":
            return self._points_c
        elif obj_type == "b":
            return self._points_b
        elif obj_type == "r":
            return self._points_r
        else:
            raise RuntimeError("Unkown obj_type: {}".format(obj_type))

    def set_points(self, obj_type, ups):
        if obj_type == "c":
            self._points_c = ups
        elif obj_type == "b":
            self._points_b = ups
        elif obj_type == "r":
            self._points_r = ups
        else:
            raise RuntimeError("Unkown obj_type: {}".format(obj_type))

    # Eval methods
    def eval(self, eval_fn, obj):
        # fn hash is the name of the function
        fn_hash = str(eval_fn).split(" ")[1]

        if fn_hash not in self._eval:
            self._eval[fn_hash] = {}

        assert isinstance(obj, ShapenetObject)
        if obj.get_id_str in self._eval[fn_hash]:
            return self._eval[fn_hash][obj.get_id_str]
        return self._eval[fn_hash].setdefault(obj.get_id_str, eval_fn(self, obj))

    # Loader methods
    def load_feature(
        self,
        obj_type,
        feat_type,
        feat_size=None,
        cache=True,
        return_list=False,
        reload=False,
        **kwargs
    ):
        """
        Given an object, an object type (c, b, r), and a feature type, load all
        associated features.
        """

        if not reload:
            try:
                stacker = self.get_feature(obj_type, feat_type)
                if not return_list:
                    stacker = np.vstack(stacker)
                return stacker
            except KeyError:
                pass

        if feat_size is None:
            feat_size = features.get_feature_size(feat_type)

        # Obtain the loader handle
        handle = None
        if obj_type == "c":
            handle = self.feature_type2path_handle_c(feat_type)
        elif obj_type == "b":
            handle = self.feature_type2path_handle_b(feat_type)
        elif obj_type == "r":
            handle = self.feature_type2path_handle_r(feat_type)
        else:
            raise RuntimeError("Unkown obj_type: {}".format(obj_type))

        # Load the feature(s)
        if not isinstance(handle, list):
            handle = [handle]

        # Stack the features
        stacker = []
        for h in handle:
            logging.debug("Loading feature using function: {}".format(h))
            feat = load_feature(h(**kwargs), size=feat_size)
            if feat is None:
                return None
            stacker.append(feat)

        if cache:
            self.set_feature(obj_type, feat_type, stacker)

        if not return_list:
            stacker = np.vstack(stacker)
        return stacker

    def load_model(self, obj_type, cache=True, reload=False, **kwargs):
        """
        Given an object, an object type (c, b, r), load the associated model.
        """

        if not reload:
            try:
                return self.get_model(obj_type)
            except AttributeError:
                pass

        try:
            if obj_type == "c":
                model = load_model(self.path_model_c(**kwargs))
            elif obj_type == "b":
                model = load_model(self.path_model_b(**kwargs))
            elif obj_type == "r":
                model = load_model(self.path_model_r(**kwargs))
            else:
                raise RuntimeError("Unkown obj_type: {}".format(obj_type))
        except errors.MeshNotClosedError as e:
            # This is to prevent future cache misses
            self.set_model(obj_type, None)
            raise e

        if cache:
            self.set_model(obj_type, model)
        return model

    def load_points(self, obj_type, cache=True, reload=False, **kwargs):
        """
        Given an object, an object type (c, b, r), load the associated upsampled points.
        """

        if not reload:
            try:
                return self.get_points(obj_type)
            except AttributeError:
                pass

        if obj_type == "c":
            points = load_points(self.path_model_c_upsampled(**kwargs))
        elif obj_type == "b":
            points = load_points(self.path_model_b_upsampled(**kwargs))
        elif obj_type == "r":
            points = load_points(self.path_model_r_upsampled(**kwargs))
        else:
            raise RuntimeError("Unkown obj_type: {}".format(obj_type))

        if cache:
            self.set_points(obj_type, points)
        return points

    def load_render(self, obj_type, **kwargs):
        """
        Given an object, an object type (c, b, r), load the associated render.
        This function does not support caching.
        """

        if obj_type == "c":
            render = cv2.imread(self.path_render_c(**kwargs))
        elif obj_type == "b":
            render = cv2.imread(self.path_render_b(**kwargs))
        elif obj_type == "r":
            render = cv2.imread(self.path_render_r(**kwargs))
        else:
            raise RuntimeError("Unkown obj_type: {}".format(obj_type))

        return render

    # == model path shortcuts ==
    def path_model_normalized(self):
        return os.path.join(
            self.root, self._class_id, self._object_id, "models", "model_normalized.obj"
        )

    def path_model_waterproofed(self):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "models",
            "model_waterproofed.obj",
        )

    def path_model_c(self):
        return os.path.join(
            self.root, self._class_id, self._object_id, "models", "model_c.obj"
        )

    def path_model_c_voxelized(self):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "models",
            "model_c_voxelized.ply",
        )

    def path_model_c_tdf(self):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "models",
            "model_c_tdf.npz",
        )

    def path_model_c_signed_tdf(self):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "models",
            "model_c_signed_tdf.npz",
        )

    def path_model_c_voxelized_rot(self, angle):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "models",
            "model_c_voxelized_rot_{}.mat".format(angle),
        )

    def path_model_b(self, idx=0):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "models",
            "model_b_{}.obj".format(idx),
        )

    def path_model_b_voxelized(self, idx=0):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "models",
            "model_b_voxelized_{}.ply".format(idx),
        )

    def path_model_b_tdf(self, idx=0):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "models",
            "model_b_tdf_{}.npz".format(idx),
        )

    def path_model_b_signed_tdf(self, idx=0):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "models",
            "model_b_signed_tdf_{}.npz".format(idx),
        )

    def path_model_b_voxelized_rot(self, angle, idx=0):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "models",
            "model_b_voxelized_rot_{}_{}.mat".format(angle, idx),
        )

    def path_model_r(self, idx=0):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "models",
            "model_r_{}.obj".format(idx),
        )

    def path_model_r_voxelized(self, idx=0):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "models",
            "model_r_voxelized_{}.ply".format(idx),
        )

    def path_model_r_tdf(self, idx=0):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "models",
            "model_r_tdf_{}.npz".format(idx),
        )

    def path_model_r_signed_tdf(self, idx=0):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "models",
            "model_r_signed_tdf_{}.npz".format(idx),
        )

    def path_model_r_voxelized_rot(self, angle, idx=0):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "models",
            "model_r_voxelize_rot_{}_{}.mat".format(angle, idx),
        )

    def path_model_b_nofrac(self, idx=0):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "models",
            "model_b_{}_nofrac.obj".format(idx),
        )

    def path_model_r_nofrac(self, idx=0):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "models",
            "model_r_{}_nofrac.obj".format(idx),
        )

    # === eval path shortcuts ===
    def path_model_c_upsampled(self):
        return os.path.join(
            self.root, self._class_id, self._object_id, "evals", "model_c_ups.npz"
        )

    def path_model_b_upsampled(self, idx=0):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "evals",
            "model_b_{}_ups.npz".format(idx),
        )

    def path_model_r_upsampled(self, idx=0):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "evals",
            "model_r_{}_ups.npz".format(idx),
        )

    # == render path shortcuts ==
    def path_render_c(self, angle):
        try:
            # For backwards compatability
            if self._resolution == (640, 640):
                return os.path.join(
                    self.root,
                    self._class_id,
                    self._object_id,
                    "renders",
                    "model_c_{}.png".format(angle),
                )
            return os.path.join(
                self.root,
                self._class_id,
                self._object_id,
                "renders",
                "model_c_{}_{}_{}.png".format(
                    angle, self._resolution[0], self._resolution[1]
                ),
            )
        except AttributeError:
            return os.path.join(
                self.root,
                self._class_id,
                self._object_id,
                "renders",
                "model_c_{}.png".format(angle),
            )

    def path_render_depth_c(self, angle):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "renders",
            "model_depth_c_{}.png".format(angle),
        )

    def path_render_depth_pano_c(self, angle):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "renders",
            "model_depth_pano_c_{}.png".format(angle),
        )

    def path_render_b(self, angle, idx=0):
        try:
            # For backwards compatability
            if self._resolution == (640, 640):
                return os.path.join(
                    self.root,
                    self._class_id,
                    self._object_id,
                    "renders",
                    "model_b_{}_{}.png".format(idx, angle),
                )
            return os.path.join(
                self.root,
                self._class_id,
                self._object_id,
                "renders",
                "model_b_{}_{}_{}_{}.png".format(
                    idx, angle, self._resolution[0], self._resolution[1]
                ),
            )
        except AttributeError:
            return os.path.join(
                self.root,
                self._class_id,
                self._object_id,
                "renders",
                "model_b_{}_{}.png".format(idx, angle),
            )

    def path_render_depth_b(self, angle, idx=0):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "renders",
            "model_depth_b_{}_{}.png".format(idx, angle),
        )

    def path_render_depth_pano_b(self, angle, idx=0):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "renders",
            "model_depth_pano_b_{}_{}.png".format(angle, idx),
        )

    def path_render_r(self, angle, idx=0):
        try:
            # For backwards compatability
            if self._resolution == (640, 640):
                return os.path.join(
                    self.root,
                    self._class_id,
                    self._object_id,
                    "renders",
                    "model_r_{}_{}.png".format(idx, angle),
                )
            return os.path.join(
                self.root,
                self._class_id,
                self._object_id,
                "renders",
                "model_r_{}_{}_{}_{}.png".format(
                    idx, angle, self._resolution[0], self._resolution[1]
                ),
            )
        except AttributeError:
            return os.path.join(
                self.root,
                self._class_id,
                self._object_id,
                "renders",
                "model_r_{}_{}.png".format(idx, angle),
            )

    def path_render_depth_r(self, angle, idx=0):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "renders",
            "model_depth_r_{}_{}.png".format(idx, angle),
        )

    def path_render_depth_pano_r(self, angle, idx=0):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "renders",
            "model_depth_pano_r_{}_{}.png".format(angle, idx),
        )

    # == feature path shortcuts ==
    def path_feature_c_sift(self, angle, flag=""):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "features",
            "model_c_{}_{}sift.npz".format(angle, flag),
        )

    def path_feature_c_sift_pano(self, angle, flag=""):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "features",
            "model_c_{}_{}sift_pano.npz".format(angle, flag),
        )

    def path_feature_c_global_edsift(self, angle, flag=""):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "features",
            "model_c_{}_{}_global_edsift.npz".format(angle, flag),
        )

    def path_feature_c_orb(self, angle, flag=""):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "features",
            "model_c_{}_{}orb.npz".format(angle, flag),
        )

    def path_feature_c_global_vgg(self, angle, flag=""):
        try:
            # For backwards compatability
            if self._resolution == (640, 640):
                return os.path.join(
                    self.root,
                    self._class_id,
                    self._object_id,
                    "features",
                    "model_c_{}_global_{}vgg.npz".format(angle, flag),
                )
            return os.path.join(
                self.root,
                self._class_id,
                self._object_id,
                "features",
                "model_c_{}_global_{}_{}_{}vgg.npz".format(
                    angle, flag, self._resolution[0], self._resolution[1]
                ),
            )
        except AttributeError:
            return os.path.join(
                self.root,
                self._class_id,
                self._object_id,
                "features",
                "model_c_{}_global_{}vgg.npz".format(angle, flag),
            )

    def path_feature_c_shot(self, flag=""):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "features",
            "model_c_{}shot.npz".format(flag),
        )

    def path_feature_c_pfh(self, flag=""):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "features",
            "model_c_{}pfh.npz".format(flag),
        )

    def path_feature_c_global_shot(self, flag=""):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "features",
            "model_c_global_{}shot.npz".format(flag),
        )

    def path_feature_c_global_pfh(self, flag=""):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "features",
            "model_c_global_{}pfh.npz".format(flag),
        )

    def path_feature_c_global_pointnet(self, flag=""):
        try:
            if self._noise == 0.0:
                if self._normals:
                    return os.path.join(
                        self.root,
                        self._class_id,
                        self._object_id,
                        "features",
                        "model_c_global_{}pointnet.npz".format(flag),
                    )
                return os.path.join(
                    self.root,
                    self._class_id,
                    self._object_id,
                    "features",
                    "model_c_global_{}_nonormals_pointnet.npz".format(flag),
                )
            if self._normals:
                return os.path.join(
                    self.root,
                    self._class_id,
                    self._object_id,
                    "features",
                    "model_c_global_{}_{}pointnet.npz".format(flag, self._noise),
                )
            return os.path.join(
                self.root,
                self._class_id,
                self._object_id,
                "features",
                "model_c_global_{}_{}_nonormals_pointnet.npz".format(flag, self._noise),
            )
        except AttributeError:
            return os.path.join(
                self.root,
                self._class_id,
                self._object_id,
                "features",
                "model_c_global_{}pointnet.npz".format(flag),
            )

    def path_feature_c_fcgf(self, flag=""):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "features",
            "model_c_global_{}_fcgf.npz".format(flag),
        )

    def path_feature_c_global_fcgf(self, flag=""):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "features",
            "model_c_global_{}_global_fcgf.npz".format(flag),
        )

    def path_feature_c_global_grid_offset_shot(self, flag=""):
        return self.path_feature_c_global_shot(flag=flag + "grid_offset_")

    def path_feature_c_global_grid_offset_pfh(self, flag=""):
        return self.path_feature_c_global_pfh(flag=flag + "grid_offset_")

    def path_feature_c_global_axis_cut_shot(self, flag=""):
        return self.path_feature_c_global_shot(flag=flag + "axis_cut_")

    def path_feature_c_global_axis_cut_pfh(self, flag=""):
        return self.path_feature_c_global_pfh(flag=flag + "axis_cut_")

    def path_feature_b_sift(self, angle, idx=0, flag=""):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "features",
            "model_b_{}_{}_{}sift.npz".format(idx, angle, flag),
        )

    def path_feature_b_sift_pano(self, angle, idx=0, flag=""):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "features",
            "model_b_{}_{}_{}sift_pano.npz".format(angle, idx, flag),
        )

    def path_feature_b_global_edsift(self, angle, idx=0, flag=""):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "features",
            "model_b_{}_{}_{}_global_edsift.npz".format(angle, idx, flag),
        )

    def path_feature_b_orb(self, angle, idx=0, flag=""):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "features",
            "model_b_{}_{}_{}orb.npz".format(idx, angle, flag),
        )

    def path_feature_b_global_vgg(self, angle, idx=0, flag=""):
        try:
            # For backwards compatability
            if self._resolution == (640, 640):
                return os.path.join(
                    self.root,
                    self._class_id,
                    self._object_id,
                    "features",
                    "model_b_{}_{}_global_{}vgg.npz".format(idx, angle, flag),
                )
            return os.path.join(
                self.root,
                self._class_id,
                self._object_id,
                "features",
                "model_b_{}_{}_global_{}_{}_{}vgg.npz".format(
                    idx, angle, flag, self._resolution[0], self._resolution[1]
                ),
            )
        except AttributeError:
            return os.path.join(
                self.root,
                self._class_id,
                self._object_id,
                "features",
                "model_b_{}_{}_global_{}vgg.npz".format(idx, angle, flag),
            )

    def path_feature_b_shot(self, idx=0, flag=""):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "features",
            "model_b_{}_{}shot.npz".format(idx, flag),
        )

    def path_feature_b_pfh(self, idx=0, flag=""):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "features",
            "model_b_{}_{}pfh.npz".format(idx, flag),
        )

    def path_feature_b_global_shot(self, idx=0, flag=""):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "features",
            "model_b_{}_global_{}shot.npz".format(idx, flag),
        )

    def path_feature_b_global_pfh(self, idx=0, flag=""):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "features",
            "model_b_{}_global_{}pfh.npz".format(idx, flag),
        )

    def path_feature_b_nofrac_shot(self, idx=0, flag=""):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "features",
            "model_b_nofrac_{}_{}shot.npz".format(idx, flag),
        )

    def path_feature_b_nofrac_pfh(self, idx=0, flag=""):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "features",
            "model_b_nofrac_{}_{}pfh.npz".format(idx, flag),
        )

    def path_feature_b_nofrac_global_shot(self, idx=0, flag=""):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "features",
            "model_b_nofrac_{}_global_{}shot.npz".format(idx, flag),
        )

    def path_feature_b_nofrac_global_pfh(self, idx=0, flag=""):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "features",
            "model_b_nofrac_{}_global_{}pfh.npz".format(idx, flag),
        )

    def path_feature_b_global_pointnet(self, idx=0, flag=""):
        try:
            if self._noise == 0.0:
                if self._normals:
                    return os.path.join(
                        self.root,
                        self._class_id,
                        self._object_id,
                        "features",
                        "model_b_{}_global_{}pointnet.npz".format(idx, flag),
                    )
                return os.path.join(
                    self.root,
                    self._class_id,
                    self._object_id,
                    "features",
                    "model_b_{}_global_{}_nonormals_pointnet.npz".format(idx, flag),
                )
            if self._normals:
                return os.path.join(
                    self.root,
                    self._class_id,
                    self._object_id,
                    "features",
                    "model_b_{}_global_{}_{}pointnet.npz".format(
                        idx, flag, self._noise
                    ),
                )
            return os.path.join(
                self.root,
                self._class_id,
                self._object_id,
                "features",
                "model_b_{}_global_{}_{}_nonormals_pointnet.npz".format(
                    idx, flag, self._noise
                ),
            )
        except AttributeError:
            return os.path.join(
                self.root,
                self._class_id,
                self._object_id,
                "features",
                "model_b_{}_global_{}pointnet.npz".format(idx, flag),
            )

    def path_feature_b_fcgf(self, idx=0, flag=""):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "features",
            "model_b_{}_global_{}_fcgf.npz".format(idx, flag),
        )

    def path_feature_b_global_fcgf(self, idx=0, flag=""):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "features",
            "model_b_{}_global_{}_global_fcgf.npz".format(idx, flag),
        )

    def path_feature_r_sift(self, angle, idx=0, flag=""):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "features",
            "model_r_{}_{}_{}sift.npz".format(idx, angle, flag),
        )

    def path_feature_r_sift_pano(self, angle, idx=0, flag=""):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "features",
            "model_r_{}_{}_{}sift_pano.npz".format(angle, idx, flag),
        )

    def path_feature_r_global_edsift(self, angle, idx=0, flag=""):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "features",
            "model_r_{}_{}_{}_global_edsift.npz".format(angle, idx, flag),
        )

    def path_feature_r_orb(self, angle, idx=0, flag=""):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "features",
            "model_r_{}_{}_{}orb.npz".format(idx, angle, flag),
        )

    def path_feature_r_global_vgg(self, angle, idx=0, flag=""):
        try:
            # For backwards compatability
            if self._resolution == (640, 640):
                return os.path.join(
                    self.root,
                    self._class_id,
                    self._object_id,
                    "features",
                    "model_r_{}_{}_global_{}vgg.npz".format(idx, angle, flag),
                )
            return os.path.join(
                self.root,
                self._class_id,
                self._object_id,
                "features",
                "model_r_{}_{}_global_{}_{}_{}vgg.npz".format(
                    idx, angle, flag, self._resolution[0], self._resolution[1]
                ),
            )
        except AttributeError:
            return os.path.join(
                self.root,
                self._class_id,
                self._object_id,
                "features",
                "model_r_{}_{}_global_{}vgg.npz".format(idx, angle, flag),
            )

    def path_feature_r_shot(self, idx=0, flag=""):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "features",
            "model_r_{}_{}shot.npz".format(idx, flag),
        )

    def path_feature_r_pfh(self, idx=0, flag=""):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "features",
            "model_r_{}_{}pfh.npz".format(idx, flag),
        )

    def path_feature_r_global_shot(self, idx=0, flag=""):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "features",
            "model_r_{}_global_{}shot.npz".format(idx, flag),
        )

    def path_feature_r_global_pfh(self, idx=0, flag=""):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "features",
            "model_r_{}_global_{}pfh.npz".format(idx, flag),
        )

    def path_feature_r_nofrac_shot(self, idx=0, flag=""):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "features",
            "model_r_nofrac_{}_{}shot.npz".format(idx, flag),
        )

    def path_feature_r_nofrac_pfh(self, idx=0, flag=""):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "features",
            "model_r_nofrac_{}_{}pfh.npz".format(idx, flag),
        )

    def path_feature_r_nofrac_global_shot(self, idx=0, flag=""):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "features",
            "model_r_nofrac_{}_global_{}shot.npz".format(idx, flag),
        )

    def path_feature_r_nofrac_global_pfh(self, idx=0, flag=""):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "features",
            "model_r_nofrac_{}_global_{}pfh.npz".format(idx, flag),
        )

    def path_feature_r_global_pointnet(self, idx=0, flag=""):
        try:
            if self._noise == 0.0:
                if self._normals:
                    return os.path.join(
                        self.root,
                        self._class_id,
                        self._object_id,
                        "features",
                        "model_r_{}_global_{}pointnet.npz".format(idx, flag),
                    )
                return os.path.join(
                    self.root,
                    self._class_id,
                    self._object_id,
                    "features",
                    "model_r_{}_global_{}_nonormals_pointnet.npz".format(idx, flag),
                )
            if self._normals:
                return os.path.join(
                    self.root,
                    self._class_id,
                    self._object_id,
                    "features",
                    "model_r_{}_global_{}_{}pointnet.npz".format(
                        idx, flag, self._noise
                    ),
                )
            return os.path.join(
                self.root,
                self._class_id,
                self._object_id,
                "features",
                "model_r_{}_global_{}_{}_nonormals_pointnet.npz".format(
                    idx, flag, self._noise
                ),
            )
        except AttributeError:
            return os.path.join(
                self.root,
                self._class_id,
                self._object_id,
                "features",
                "model_r_{}_global_{}pointnet.npz".format(idx, flag),
            )

    def path_feature_r_fcgf(self, idx=0, flag=""):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "features",
            "model_r_{}_global_{}_fcgf.npz".format(idx, flag),
        )

    def path_feature_r_global_fcgf(self, idx=0, flag=""):
        return os.path.join(
            self.root,
            self._class_id,
            self._object_id,
            "features",
            "model_r_{}_global_{}_global_fcgf.npz".format(idx, flag),
        )


# Loaders
def load_split(root_dir, splits_file):
    """
    Load a split file from disk, return the train and test partitions.
    """
    object_id_dict = json.load(open(splits_file, "r"))
    id_train_list = [
        ShapenetObject(root_dir, o[0], o[1]) for o in object_id_dict["id_train_list"]
    ]
    id_test_list = [
        ShapenetObject(root_dir, o[0], o[1]) for o in object_id_dict["id_test_list"]
    ]
    return id_train_list, id_test_list


def load_points(path):
    """
    Load array of points or a point cloud file from disk. If file is not found
    will throw an error.
    """
    ext = os.path.splitext(path)[-1]
    assert (
        ext == ".npz" or ext == ".obj"
    ), "Expected .npz or .obj file, got .{} file".format(ext)
    if not os.path.isfile(path):
        logging.debug("File dne: {}".format(path))
        raise FileNotFoundError
    logging.debug("Loading points from: {}".format(path))
    if ext == ".npz":
        return np.load(path)["xyz"]
    elif ext == ".obj":
        return load_model(path).vertices


def load_feature(path, size=None):
    """
    Load a feature from disk. If file is not found will return None.
    """
    assert (
        os.path.splitext(path)[-1] == ".npz"
    ), "Expected .npz file, got .{} file".format(os.path.splitext(path)[-1])

    logging.debug("Loading feature from: {}".format(path))
    if not os.path.isfile(path):
        logging.debug("File dne: {}".format(path))
        return None
    try:
        if size is not None:
            return np.load(path)["features"][:, :size].astype("float32")
        return np.load(path)["features"].astype("float32")
    except (IndexError, ValueError):
        # Corruption, not enough features, etc
        logging.debug("Problem loading feature: {}".format(path))
        return None


def load_model(path):
    """
    Load a trimesh mesh file from disk. If file is not found will throw an error.
    """
    # In the general case this might throw an error. All meshes should be a single geometry object at this point though.
    try:
        model = trimesh.load(path)
        if not model.is_watertight:
            raise errors.MeshNotClosedError
        return model
    except (AttributeError, ValueError):
        raise errors.MeshNotClosedError


# Get all objects in shapenet
def shapenet_find_all(root):
    """Return a list of all objects in the dataset. This takes a long time."""
    objects = []
    if os.path.exists(root):
        try:
            lowest_level = next(utils.get_file(root, [".obj"]))
        except StopIteration:
            return
        classes_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(lowest_level)))
        )
        for f in os.listdir(classes_dir):
            f = os.path.join(classes_dir, f)
            if os.path.isdir(f):
                class_id = f.split("/")[-1]
                for fprime in os.listdir(f):
                    fprime = os.path.join(f, fprime)
                    if os.path.isdir(fprime):
                        object_id = fprime.split("/")[-1]
                        objects.append(ShapenetObject(classes_dir, class_id, object_id))
    return objects


def class2string(class_id):
    global SHAPENET_CLASSES
    if SHAPENET_CLASSES is None:
        SHAPENET_CLASSES = json.load(open("proxies/shapenet_classes.json"))
    try:
        return SHAPENET_CLASSES[class_id]
    except KeyError:
        return "UNKNOWN"


def string2class(string):
    global SHAPENET_CLASSES
    if SHAPENET_CLASSES is None:
        SHAPENET_CLASSES = json.load(open("proxies/shapenet_classes.json"))
    for k in SHAPENET_CLASSES:
        if SHAPENET_CLASSES[k] == string:
            return k
    return "UNKNOWN"


def scan_dataset_instanceid2class(instance_id):
    global SCANNED_CLASSES
    if SCANNED_CLASSES is None:
        SCANNED_CLASSES = json.load(open("proxies/scanned_classes.json"))

    for cls, inst in SCANNED_CLASSES.items():
        if instance_id in inst:
            return cls
    return "00000000"


# Conversion to/from shapenet id
def shapenet_id2dir(root, object_id):
    """Return object directory from the shapenet id"""
    object_class, object_id = object_id.split(":")
    return os.path.join(root, object_class, object_id)


def shapenet_file2id(f):
    """Return the shapenet id from any file in the database"""
    object_dir = os.path.dirname(os.path.dirname(f))
    return object_dir.split("/")[-2] + ":" + object_dir.split("/")[-1]


# Methods to get specific files from object_id
def shapenet_id2model_normalized(root, class_id, object_id):
    """Return the source model (model_normalized.obj) from an object id"""
    # object_class, object_id = object_id.split(':')
    return os.path.join(root, class_id, object_id, "models", "model_normalized.obj")


def shapenet_id2image(root, object_id):
    """Return the path to a rendered image from the shapenet id"""
    render_dir = os.path.join(shapenet_id2dir(root, object_id), "renders")
    if os.path.exists(render_dir):
        return utils.get_file(render_dir, [".png", ".jpg"])
    return []


def shapenet_id2feats(root, object_id, feats, feat_exts):
    """Return the path to a feature from the shapenet id"""
    feature_dir = os.path.join(shapenet_id2dir(root, object_id), "features")
    if os.path.exists(feature_dir):
        for f in utils.get_file(feature_dir, feat_exts):
            if any([feat in f for feat in feats]):
                yield f
    return []
