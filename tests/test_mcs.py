# -*- coding: utf-8 -*-
"""Tests for the MolClusterkit package."""

import unittest
from pathlib import Path

import numpy as np

from MolClusterkit.mcs import MCSClustering


class TestMCS(unittest.TestCase):
    def setUp(self) -> None:
        self.testroot = Path(__file__).parent
        with (self.testroot / "resources/HIV-sample.smi").open("r") as f:
            self.smiles = f.read().splitlines()
        np.random.seed(42)
        self.scores = np.random.randint(0, 10, size=len(self.smiles))
        self.algorithms = ["DBSCAN", "Hierarchical", "Spectral", "GraphBased"]
        self.kwargs = {
            "DBSCAN": {"eps": 0.5, "min_samples": 2},
            "Hierarchical": {"t": 2},
            "Spectral": {"n_clusters": 2},
            "GraphBased": {"threshold": 0.5},
        }

    def test_MCSClustering(self):
        smiles_list = ["CCO", "CCN", "CCS"]  # Some test SMILES strings
        mcs_cluster = MCSClustering(smiles_list)

        # Test compute_similarity_matrix
        smarts_matrix, simi_matrix = mcs_cluster.compute_similarity_matrix()
        self.assertEqual(smarts_matrix.shape, (3, 3))
        self.assertEqual(simi_matrix.shape, (3, 3))

        # Test DBSCAN clustering
        labels_dbscan = mcs_cluster.dbscan_clustering(eps=0.5, min_samples=2)
        self.assertEqual(len(labels_dbscan), 3)

        # Test hierarchical clustering
        labels_hierarchical = mcs_cluster.hierarchical_clustering(t=2)
        self.assertEqual(len(labels_hierarchical), 3)

    def test_algorithms(self):
        smiles_list = ["CCO", "CCN", "CCS"]  # Some test SMILES strings
        for alg in self.algorithms:
            mcs_cluster = MCSClustering(smiles_list)
            mcs_cluster.compute_similarity_matrix()
            kwargs = self.kwargs[alg] if alg in self.kwargs else {}
            labels = mcs_cluster.cluster_molecules(algorithm=alg, **kwargs)
            self.assertEqual(len(labels), len(smiles_list))

    def test_big_data(self):
        smiles = self.smiles[:200]
        mcs_cluster = MCSClustering(smiles)
        mcs_cluster.compute_similarity_matrix()
        labels = mcs_cluster.cluster_molecules(algorithm="DBSCAN", eps=0.5)
        self.assertEqual(len(labels), len(smiles))

    def tearDown(self) -> None:
        return super().tearDown()


if __name__ == "__main__":
    unittest.main()
