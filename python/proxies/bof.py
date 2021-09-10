import logging

import faiss
import numpy as np


class BagOfFeatures:
    def __init__(self, feature_size, nfeatures, niter=20):
        self._feature_size = feature_size
        self._nfeatures = nfeatures
        self._niter = niter
        self._trained = False

    @property
    def features(self):
        assert self._trained
        return self._kmeans.centroids

    def train(self, feature_vec):
        self._kmeans = faiss.Kmeans(
            self._feature_size, self._nfeatures, niter=self._niter, gpu=True
        )

        self._kmeans.train(feature_vec)
        self._trained = True

    def encode(self, feature_vec):
        """
        Generates a histogram corresponding to the kmeans centroids.
        """
        assert self._trained
        _, knns = self._kmeans.index.search(feature_vec, 1)
        indices, counts = np.unique(knns.flatten(), return_counts=True)
        feature = np.zeros((self._nfeatures))
        feature[indices.astype(int)] = counts
        return feature
