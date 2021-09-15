import os, time, logging
import pickle, json
from collections import OrderedDict

import faiss
import tqdm
import numpy as np

import proxies.shapenet as shapenet
import proxies.features as feature_utils


class ObjectDatabase:
    def __init__(
        self,
        load_from=None,
        obj_kwargs=None,
        splits_file=None,
        root_dir=None,
        feature_sizes=None,
        feature_types=None,
        feature_weights=None,
        num_renders=8,
        add_query=False,
        cache=True,
        use_gpu=True,
        constrained_gpu=True,
    ):
        self._fais_index = OrderedDict()
        self._feature_index_to_object = OrderedDict()
        self._computed_metadata = False
        self._num_renders = num_renders
        self._cache = cache
        self._use_gpu = use_gpu
        self._constrained_gpu = constrained_gpu

        self._obj_kwargs = {}
        if obj_kwargs is not None:
            self._obj_kwargs = obj_kwargs

        if self._use_gpu:
            num_gpus = faiss.get_num_gpus()
            if num_gpus == 0:
                logging.info("No gpus found, reverting to cpu")
                self._use_gpu = False
            else:
                logging.info("Found {} gpus".format(num_gpus))

        if load_from is not None:
            start_time = time.time()
            self.load(load_from)
            logging.info("Loaded database took: {}".format(time.time() - start_time))

            if root_dir is not None:
                self.set_root(root_dir)
        else:
            assert os.path.isdir(root_dir)
            object_id_dict = json.load(open(splits_file, "r"))
            if add_query:
                self._database_object_id_list = [
                    shapenet.ShapenetObject(
                        root_dir,
                        o[0],
                        o[1],
                        num_renders=num_renders,
                        **self._obj_kwargs,
                    )
                    for o in object_id_dict["id_train_list"]
                    + object_id_dict["id_test_list"]
                ]
            else:
                self._database_object_id_list = [
                    shapenet.ShapenetObject(
                        root_dir,
                        o[0],
                        o[1],
                        num_renders=num_renders,
                        **self._obj_kwargs,
                    )
                    for o in object_id_dict["id_train_list"]
                ]

            # This is pretty jank
            processing_scanned = "scanned" in splits_file
            if processing_scanned:
                self._query_object_id_list = []
                for o in object_id_dict["id_test_list"]:
                    self._query_object_id_list.append(
                        shapenet.ShapenetObject(
                            root_dir,
                            o[0],
                            o[1],
                            true_class=shapenet.scan_dataset_instanceid2class(o[1]),
                            num_renders=num_renders,
                            **obj_kwargs,
                        )
                    )
            else:
                self._query_object_id_list = [
                    shapenet.ShapenetObject(
                        root_dir,
                        o[0],
                        o[1],
                        num_renders=num_renders,
                        **obj_kwargs,
                    )
                    for o in object_id_dict["id_test_list"]
                ]

            # Every parameter here needs to be in the save/load methods
            self._feature_sizes = feature_sizes
            self._feature_types = feature_types
            if feature_weights is None:
                self._feature_weights = [1.0] * len(feature_types)

            self._gpu_loaded = [False] * len(feature_types)

            assert (
                len(self._feature_weights)
                == len(self._feature_sizes)
                == len(self._feature_types)
            )
            self._feature_avgs = [np.zeros((s)) for s in feature_sizes]
            self._feature_stds = [np.ones((s)) for s in feature_sizes]
            self._feature_prune = [np.ones((s)) for s in feature_sizes]

            # Create the faiss indices
            for feat_size, feat_type in zip(feature_sizes, feature_types):
                index = faiss.IndexFlatL2(feat_size)
                self._fais_index[feat_type] = index
                self._feature_index_to_object[feat_type] = [[], []]

    def save(self, fname):
        """Database is saved as a set of files. Pass this function a directory."""

        if not os.path.isdir(fname):
            logging.debug("Creating directory: {}".format(fname))
            os.mkdir(fname)

        save_data = (
            self._feature_index_to_object,
            self._feature_sizes,
            self._feature_types,
            self._feature_weights,
            self._feature_stds,
            self._feature_avgs,
            self._feature_prune,
            self._database_object_id_list,
            self._query_object_id_list,
            self._computed_metadata,
            self._num_renders,
            self._obj_kwargs,
        )

        logging.info("Saving database to: {}".format(fname))
        with open(os.path.join(fname, "database.pkl"), "wb+") as f:
            pickle.dump(save_data, f, pickle.HIGHEST_PROTOCOL)

        for feat_idx, (feat_type, index) in enumerate(self._fais_index.items()):
            if self._gpu_loaded[feat_idx]:
                faiss.write_index(
                    faiss.index_gpu_to_cpu(index),
                    os.path.join(fname, str(feat_type) + ".index"),
                )
            else:
                faiss.write_index(
                    index,
                    os.path.join(fname, str(feat_type) + ".index"),
                )

    def load(self, fname):
        """Database is saved as a set of files."""

        logging.info("Loading database from: {}".format(fname))
        with open(os.path.join(fname, "database.pkl"), "rb") as f:
            save_data = pickle.load(f)

        # For backwards compatability
        if len(save_data) == 11:
            (
                self._feature_index_to_object,
                self._feature_sizes,
                self._feature_types,
                self._feature_weights,
                self._feature_stds,
                self._feature_avgs,
                self._feature_prune,
                self._database_object_id_list,
                self._query_object_id_list,
                self._computed_metadata,
                self._num_renders,
            ) = save_data
            self._obj_kwargs = {}
        else:
            (
                self._feature_index_to_object,
                self._feature_sizes,
                self._feature_types,
                self._feature_weights,
                self._feature_stds,
                self._feature_avgs,
                self._feature_prune,
                self._database_object_id_list,
                self._query_object_id_list,
                self._computed_metadata,
                self._num_renders,
                self._obj_kwargs,
            ) = save_data

        for idx, o in enumerate(self._query_object_id_list):
            if o.class_id == "scanned":
                self._query_object_id_list[idx] = shapenet.ShapenetObject(
                    o.root,
                    o.class_id,
                    o.object_id,
                    true_class=shapenet.scan_dataset_instanceid2class(o.object_id),
                    num_renders=o._num_renders,
                )

        self._gpu_loaded = [False] * len(self._feature_types)

        for feat_type in self._feature_types:
            self._fais_index[feat_type] = faiss.read_index(
                os.path.join(fname, str(feat_type) + ".index")
            )

    def train(self):
        for feat_type in self._feature_index_to_object.keys():
            object_ids = np.expand_dims(
                np.array(self._feature_index_to_object[feat_type][0]), axis=1
            )
            object_ids_sub_idx = np.expand_dims(
                np.array(self._feature_index_to_object[feat_type][1]), axis=1
            )
            self._feature_index_to_object[feat_type] = np.hstack(
                (object_ids, object_ids_sub_idx)
            )

        for feat_idx, feat_type in enumerate(self._feature_types):
            self._feature_weights[feat_idx] = self.get_default_feature_weights(
                feat_type
            )

    def set_root(self, new_root):
        for obj_idx in range(len(self._database_object_id_list)):
            self._database_object_id_list[obj_idx].root = new_root
        for obj_idx in range(len(self._query_object_id_list)):
            self._query_object_id_list[obj_idx].root = new_root

    def load_gpu(self):
        if not self._use_gpu:
            return
        for feat_idx, feat_type in enumerate(self._feature_types):
            if not self._gpu_loaded[feat_idx]:
                self.load_index_gpu(feat_type)
                self._gpu_loaded[feat_idx] = True

    def unload_gpu(self):
        for feat_idx, feat_type in enumerate(self._feature_types):
            if self._gpu_loaded[feat_idx]:
                self.unload_index_gpu(feat_type)
                self._gpu_loaded[feat_idx] = False

    def load_index_gpu(self, feat_type):
        if not self._use_gpu:
            return
        feat_idx = self._feature_types.index(feat_type)
        if not self._gpu_loaded[feat_idx]:
            self._fais_index[feat_type] = faiss.index_cpu_to_all_gpus(
                self._fais_index[feat_type]
            )
            self._gpu_loaded[feat_idx] = True

    def unload_index_gpu(self, feat_type):
        feat_idx = self._feature_types.index(feat_type)
        if not self._gpu_loaded[feat_idx]:
            self._fais_index[feat_type] = faiss.index_gpu_to_cpu(
                self._fais_index[feat_type]
            )
            self._gpu_loaded[feat_idx] = False

    @property
    def on_gpu(self):
        return self._gpu_loaded

    @property
    def using_gpu(self):
        return self._use_gpu

    @property
    def features(self):
        return [e for e in self._feature_types]

    @property
    def sizes(self):
        return [e for e in self._feature_sizes]

    @property
    def weights(self):
        return [e for e in self._feature_weights]

    @property
    def database_objects(self):
        return self._database_object_id_list

    @property
    def query_objects(self):
        return self._query_object_id_list

    def __len__(self):
        return len(self._database_object_id_list)

    def __getitem__(self, item):
        return self._database_object_id_list[item]

    def __contains__(self, item):
        return item in self._database_object_id_list

    def __str__(self):
        ret = "Database contains {} objects\n".format(len(self))
        ret += "{:<15} | {:<5} | {:<15}\n".format("feature", "size", "num features")
        for f in [
            "{:<15} | {:<5} | {:<15}\n".format(f, str(s), fi.ntotal)
            for f, s, fi in zip(self.features, self.sizes, self._fais_index.values())
        ]:
            ret += f
        return ret[:-1]

    def get_default_feature_weights(self, feat_type):
        return 1

    def get_feature(self, obj, obj_type, reload=False):
        """
        Return a list of features for a given object, loaded from disk.
        """
        feature_acc = []

        for feat_type, feat_size in zip(self._feature_types, self._feature_sizes):
            feature = obj.load_feature(
                obj_type, feat_type, feat_size, cache=self._cache, reload=reload
            )
            if feature is None:
                return None
            feature_acc.append(feature)
        return feature_acc

    def update_metadata(self, feature_type, field, value):
        """
        Update the metadata corresponding to one of the features.
        """
        feat_index = self._feature_types.index(feature_type)

        if field == "type":
            self._feature_types[feat_index] = value
        elif field == "size":
            self._feature_sizes[feat_index] = value
        elif field == "ext":
            self._feature_exts[feat_index] = value
        elif field == "avg":
            self._feature_avgs[feat_index] = value
        elif field == "std":
            self._feature_stds[feat_index] = value
        elif field == "prune":
            self._feature_prune[feat_index] = value
        else:
            raise RuntimeError

    def add_feature(
        self,
        feature_size=None,
        feature_type=None,
        feature_weight=None,
    ):
        """
        Add a new feature class to the database.
        """

        self._feature_sizes.append(feature_size)
        self._feature_types.append(feature_type)
        self._feature_weights.append(feature_weight)

        self._fais_index[feature_type] = faiss.IndexFlatL2(feature_size)
        self._feature_index_to_object[feature_type] = [[], []]

    def remove_feature(self, feature_type=None):
        """
        Use this to delete a feature. Note that you must manually save after
        doing this for the changes to persist.
        """
        feat_index = self._feature_types.index(feature_type)

        del self._feature_types[feat_index]
        del self._feature_sizes[feat_index]
        del self._feature_weights[feat_index]

        del self._fais_index[feature_type]
        del self._feature_index_to_object[feature_type]

    def add(self, obj_idx, feature_vec, feature_type, sub_idx=0):
        """
        Add one or more feature vectors to the database. Needs to be added with
        its corresponding index and feature type. Feature type must
        match with existing or key error will be thrown.
        """

        # Store the index of that object
        self._feature_index_to_object[feature_type][0].extend(
            [obj_idx] * feature_vec.shape[0]
        )
        self._feature_index_to_object[feature_type][1].extend(
            [sub_idx] * feature_vec.shape[0]
        )

        # Add that feature to faiss
        self._fais_index[feature_type].add(feature_vec)

    def poluate_metadata(self):
        """
        This handles computation of meand and std for normalization.
        """
        feature_acc = [list() for _ in range(len(self._feature_types))]

        logging.info("Polulating Metadata")
        pbar = tqdm.tqdm(range(len(self._database_object_id_list)))

        start_len = len(self._database_object_id_list)
        obj_idx = 0
        nans_removed = 0
        while obj_idx < len(self._database_object_id_list):
            # pbar.write("[{}]".format(os.path.dirname(obj.path_model_normalized())))

            obj_feature_acc = []
            for feat_idx, (feat_type, feat_size) in enumerate(
                zip(self._feature_types, self._feature_sizes)
            ):

                # Load the feature
                feature = self._database_object_id_list[obj_idx].load_feature(
                    "c", feat_type, feat_size, cache=self._cache
                )

                # If no feature was loaded, break and toss out this object
                if feature is None:
                    obj_feature_acc = None
                    break

                # Remove any nan features (this happens occasionally for 3d)
                nan_rows = np.isnan(feature).any(axis=1)
                nans_removed += np.count_nonzero(nan_rows)
                feature = feature[np.logical_not(nan_rows), :]
                obj_feature_acc.append(feature)

            # Toss the object
            if obj_feature_acc is None:
                del self._database_object_id_list[obj_idx]

            # Add it to the list
            else:
                for feat_idx, feature in enumerate(obj_feature_acc):
                    feature_acc[feat_idx].append(feature)
                obj_idx += 1

            pbar.update()
        pbar.close()

        if start_len - len(self._database_object_id_list) > 0:
            logging.info(
                "Discarded {} missing objects".format(
                    start_len - len(self._database_object_id_list)
                )
            )
        if nans_removed > 0:
            logging.info("Discarded {} nan features".format(nans_removed))

        # Compute and store mean and std
        for feat_idx, (feat_list, feat_type) in enumerate(
            zip(feature_acc, self._feature_types)
        ):
            feat_stack = np.vstack(feat_list)

            # Store the mean and std
            self.update_metadata(feat_type, "avg", np.mean(feat_stack, axis=0))
            self.update_metadata(feat_type, "std", np.std(feat_stack, axis=0))

            self._feature_avgs[feat_idx] = np.nan_to_num(
                self._feature_avgs[feat_idx], nan=0.0
            )
            self._feature_stds[feat_idx] = np.nan_to_num(
                self._feature_stds[feat_idx], nan=0.0
            )

            # Prune is used to prevent div by zero errors
            self._feature_prune[feat_idx][self._feature_stds[feat_idx] < 1e-6] = 0.0
            self._feature_stds[feat_idx][self._feature_stds[feat_idx] < 1e-6] = 1e-6

        self._computed_metadata = True

    def populate(self):
        """
        Iterate through databse objects and load features into the database.
        """

        assert self._computed_metadata, "Need to compute metadata before populating"

        logging.info("Polulating Database Features")
        pbar = tqdm.tqdm(range(len(self._database_object_id_list)))
        for obj_idx in pbar:
            # pbar.write("[{}]".format(os.path.dirname(obj.path_model_normalized())))

            for feat_type, feat_size in zip(self._feature_types, self._feature_sizes):

                if feature_utils.is_local_type(feat_type) and feature_utils.is_2d_type(
                    feat_type
                ):
                    # Indicates that the object has multiple sub-representations
                    feat_list = self._database_object_id_list[obj_idx].load_feature(
                        "c", feat_type, feat_size, cache=self._cache, return_list=True
                    )
                else:
                    feat_list = self._database_object_id_list[obj_idx].load_feature(
                        "c", feat_type, feat_size, cache=self._cache
                    )

                if feat_list is None:
                    continue

                if not isinstance(feat_list, list):
                    feat_list = [feat_list]

                for sub_idx, feature in enumerate(feat_list):
                    # Apply global normalization
                    feat_index = self._feature_types.index(feat_type)
                    feature = (
                        feature - self._feature_avgs[feat_index].astype("float32")
                    ) / self._feature_stds[feat_index].astype("float32")
                    feature = feature * self._feature_prune[feat_index].astype(
                        "float32"
                    )  # This will remove low var channels

                    # Add the feature
                    self.add(
                        feature_vec=feature,
                        feature_type=feat_type,
                        obj_idx=obj_idx,
                        sub_idx=sub_idx,
                    )

    def populate_query(self, reload=False):
        """
        Iterate through query objects and load features into the database.
        """
        if not self._cache:
            return

        logging.info("Polulating Query Features")
        pbar = tqdm.tqdm(range(len(self._query_object_id_list)))
        start_len = len(self._query_object_id_list)
        obj_idx = 0
        while obj_idx < len(self._query_object_id_list):
            feature = self.get_feature(
                self._query_object_id_list[obj_idx], obj_type="b", reload=reload
            )

            if feature is None:
                del self._query_object_id_list[obj_idx]
            else:
                obj_idx += 1
            pbar.update()
        pbar.close()

        if start_len - len(self._query_object_id_list) > 0:
            logging.info(
                "Discarded {} missing objects".format(
                    start_len - len(self._query_object_id_list)
                )
            )

    def query_fast(
        self,
        feature_vecs,
        indexer,
        num_objects,
        topk=5,
        depth=1000,
        keep=0.75,
        use_global=False,
        allowlist=None,
        excllist=None,
        batch_num=0,
        query_cache_fname=None,
    ):
        """
        Query the database with a list of vertically stacked features corresponding
        to several objects.
        """

        # Hashmaps are quick
        if use_global:
            object_stacker = [{} for _ in range(num_objects)]
            num_features = sum([1 for v in feature_vecs if v is not None])
        else:
            object_stacker = [{} for _ in range(num_objects)]

        # Iterate over the individual feature databases
        for feat_idx, feat_type in enumerate(self._fais_index.keys()):
            if feature_vecs[feat_idx] is None:
                continue
            start_time = time.time()
            logging.info(
                "Performing feature {} query on index with {} entries".format(
                    feat_type, self._fais_index[feat_type].ntotal
                )
            )

            # Remove any nan features (this happens occasionally for 3d)
            feature = feature_vecs[feat_idx]
            feature = feature[~np.isnan(feature).any(axis=1), :]

            # Apply global normalization
            feature = (
                feature - self._feature_avgs[feat_idx].astype("float32")
            ) / self._feature_stds[feat_idx].astype("float32")
            feature = feature * self._feature_prune[feat_idx].astype("float32")

            depth = min(depth, self._fais_index[feat_type].ntotal)

            # Enable results caching
            if query_cache_fname is not None:
                fname = query_cache_fname.format(batch_num, feat_type, use_global)
                try:
                    data = np.load(fname)
                    dists = data["dists"]
                    knns = data["knns"]
                    logging.info("Loading took: {}".format(time.time() - start_time))
                except FileNotFoundError:
                    # Load the index onto the gpu
                    self.load_index_gpu(feat_type)

                    dists, knns = self._fais_index[feat_type].search(feature, depth)

                    # Unload the index
                    self.unload_index_gpu(feat_type)

                    np.savez(fname, dists=dists, knns=knns)
                    logging.info("Qeury took: {}".format(time.time() - start_time))
            else:
                # Load the index onto the gpu
                self.load_index_gpu(feat_type)

                dists, knns = self._fais_index[feat_type].search(feature, depth)

                # Unload the index
                self.unload_index_gpu(feat_type)

            start_time = time.time()
            logging.info("Organizing results of query ...")
            for obj_idx, (start, end) in tqdm.tqdm(enumerate(indexer[feat_idx])):
                if use_global:
                    dists, knns = dists.flatten(), knns.flatten()

                    # Get the slice corresponding to this object
                    obj_dists = dists[start * depth : end * depth]
                    obj_knns = knns[start * depth : end * depth]

                    # Sort
                    sorted_idxs = np.argsort(obj_dists)
                    obj_dists = obj_dists[sorted_idxs]
                    obj_knns = obj_knns[sorted_idxs]

                    seen_set = set()

                    for dist, object_id in zip(
                        obj_dists,
                        self._feature_index_to_object[feat_type][obj_knns, 0],
                    ):
                        # Only let those on the allowlist in
                        if (allowlist is not None) and (
                            object_id not in allowlist[obj_idx]
                        ):
                            continue

                        # Only let those on the allowlist in
                        obj_str_id = self.database_objects[object_id].get_id_str
                        if (excllist is not None) and (obj_str_id in excllist):
                            continue

                        # Only take first occurance
                        if object_id in seen_set:
                            continue
                        seen_set.add(object_id)

                        # Apply weight
                        dist *= (
                            self._feature_weights[feat_idx]
                            / self._feature_sizes[feat_idx]
                        )

                        # Increment counter and distance
                        if object_id not in object_stacker[obj_idx]:
                            object_stacker[obj_idx][object_id] = (1, dist)
                        else:
                            object_stacker[obj_idx][object_id] = (
                                object_stacker[obj_idx][object_id][0] + 1,
                                object_stacker[obj_idx][object_id][1] + dist,
                            )
                    else:
                        raise NotImplementedError

            logging.info("Oranization took: {}".format(time.time() - start_time))

        # Convert the dictionaries to numpy arrays
        for obj_idx in range(len(object_stacker)):
            if use_global:
                # This will discard any that do not show up multiple times
                knns_values = np.array(
                    [
                        [k, d[1]]
                        for k, d in object_stacker[obj_idx].items()
                        if d[0] == num_features
                    ]
                )
            else:
                raise NotImplementedError

            if knns_values.shape[0] == 0:
                object_stacker[obj_idx] = np.empty()
            else:
                object_stacker[obj_idx] = knns_values[np.argsort(knns_values[:, 1]), :][
                    :topk, :
                ]

        return object_stacker

    def hierarchical_query(
        self,
        obj_list,
        topk=5,
        depth=2048,
        feat_mask=None,
        batch_num=0,
        query_cache_fname=None,
        database_excl_list=None,
    ):
        """
        Perform query using global features.
        """

        if query_cache_fname is not None:
            assert os.path.isdir(os.path.dirname(query_cache_fname))

        if not self._constrained_gpu:
            self.load_gpu()

        assert depth <= 2048, "Cannot query with depth greater than 2048"

        # Load the features from disk
        start_time = time.time()
        obj_feats = [self.get_feature(o, obj_type="b") for o in obj_list]

        # If any could not be loaded, remove them
        obj_feats = [f for f in obj_feats if f is not None]
        logging.debug("Loaded features in: {}s".format(time.time() - start_time))

        # Stack each feature
        feature_vecs = []
        indexer = []
        for feat_idx in range(len(obj_feats[0])):

            if isinstance(feat_mask, list) and not feat_mask[feat_idx]:
                feature_vecs.append(None)
                indexer.append([])
                continue

            # Concat the feature vectors by feature
            feature_vecs.append(np.vstack([b[feat_idx] for b in obj_feats]))

            # Record the start and end indices
            indexer.append([])
            idx_start = 0
            for feat in obj_feats:
                indexer[-1].append((idx_start, idx_start + feat[feat_idx].shape[0]))
                idx_start += feat[feat_idx].shape[0]

        feats_list = []
        for obj in obj_list:
            feature = self.get_feature(obj, obj_type="b")
            if feature is None:
                continue
            feats_list.append(feature)

        assert len(feature_vecs) == len(
            self._feature_types
        ), "Feature has incorrect size."

        global_features = [
            v if feature_utils.is_global_type(t) else None
            for v, t in zip(feature_vecs, self._feature_types)
        ]

        logging.debug("Performing global query for {} objects".format(len(indexer[0])))
        start_time = time.time()
        results_global = self.query_fast(
            feature_vecs=global_features,
            indexer=indexer,
            num_objects=len(obj_list),
            topk=topk,
            depth=depth,
            use_global=True,
            batch_num=batch_num,
            excllist=database_excl_list,
            query_cache_fname=query_cache_fname,
        )
        logging.debug("Global query took: {}".format(time.time() - start_time))
        return results_global
