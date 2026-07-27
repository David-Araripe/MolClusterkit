# -*- coding: utf-8 -*-
"""Tests for the BaseClusterer clustering algorithms via MCSClustering."""
import unittest

import numpy as np

from MolClusterkit.mcs import MCSClustering


class TestBaseClustererAlgorithms(unittest.TestCase):
    """Tests for BaseClusterer methods using MCSClustering as concrete class."""

    def setUp(self):
        self.smiles = ["CCO", "CCN", "CCS", "c1ccccc1", "CC(=O)O"]
        self.clusterer = MCSClustering(self.smiles)
        self.clusterer.compute_similarity_matrix(show_progress=False)

    def test_hierarchical_silhouette_clustering(self):
        labels = self.clusterer.hierarchical_silhouette_clustering(max_clusters=3)
        self.assertEqual(len(labels), len(self.smiles))

    def test_hierarchical_silhouette_clustering_best_k(self):
        labels = self.clusterer.hierarchical_silhouette_clustering(max_clusters=4)
        n_unique = len(np.unique(labels))
        self.assertGreaterEqual(n_unique, 2)
        self.assertLessEqual(n_unique, 4)

    def test_cluster_molecules_no_similarity_matrix(self):
        clusterer = MCSClustering(["CCO", "CCN"])
        with self.assertRaises(ValueError):
            clusterer.cluster_molecules(algorithm="DBSCAN")

    def test_cluster_molecules_unsupported_algorithm(self):
        with self.assertRaises(ValueError):
            self.clusterer.cluster_molecules(algorithm="KMeans")

    def test_graph_based_clustering(self):
        labels = self.clusterer.graph_based_clustering(threshold=0.3)
        self.assertEqual(len(labels), len(self.smiles))
        # All labels should be >= -1 (unassigned nodes get -1)
        self.assertTrue(all(l >= -1 for l in labels))

    def test_no_smiles_graph_based_raises(self):
        clusterer = MCSClustering()
        # Manually set a dummy similarity matrix to bypass that check
        clusterer.similarity_matrix = np.eye(3)
        with self.assertRaises(ValueError):
            clusterer.graph_based_clustering()


if __name__ == "__main__":
    unittest.main()
