# -*- coding: utf-8 -*-
"""Tests for the MolClusterkit.best_picker module."""
import unittest

import pandas as pd

from MolClusterkit.best_picker import butina_based_clustering, mcs_based_clustering


class TestBestPicker(unittest.TestCase):
    def setUp(self):
        self.smiles = ["CCO", "CCN", "CCS", "c1ccccc1", "CC(=O)O"]

    def test_butina_based_clustering_list_input(self):
        result = butina_based_clustering(self.smiles, smiles_col=None, njobs=1)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("cluster_id", result.columns)
        self.assertEqual(len(result), len(self.smiles))

    def test_mcs_based_clustering_list_input(self):
        result = mcs_based_clustering(
            self.smiles,
            smiles_col=None,
            algorithm="Hierarchical",
            n_jobs=1,
            timeout=15,
            t=2,
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("cluster_id", result.columns)
        self.assertEqual(len(result), len(self.smiles))

    def test_mcs_based_clustering_invalid_type(self):
        with self.assertRaises(ValueError):
            mcs_based_clustering(42, smiles_col=None)


if __name__ == "__main__":
    unittest.main()
