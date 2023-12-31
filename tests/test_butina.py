# -*- coding: utf-8 -*-
"""Tests for the MolClusterkit.butina module."""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from MolClusterkit.butina import ButinaClustering
from MolClusterkit.misc import reset_index_if_needed


class TestButina(unittest.TestCase):
    def setUp(self) -> None:
        self.testroot = Path(__file__).parent
        with (self.testroot / "resources/HIV-sample.smi").open("r") as f:
            self.smiles = f.read().splitlines()
        np.random.seed(42)
        self.scores = np.random.randint(0, 10, size=len(self.smiles))

    def test_reset_index_if_needed(self):
        # Create a test dataframe
        df = pd.DataFrame({"col1": [1, 2, 3, 4, 5]})
        df.drop(2, inplace=True)
        # Check if index is reset
        new_df = reset_index_if_needed(df)
        self.assertTrue(all(new_df.index == [0, 1, 2, 3]))

    def test_ButinaClustering(self):
        smiles_list = ["CCO", "CCN", "CCS"]  # Some test SMILES strings
        bclusterer = ButinaClustering(smiles_list)

        # Check initialization
        self.assertEqual(bclusterer.smiles_list, smiles_list)
        self.assertEqual(bclusterer.njobs, 8)  # default value
        self.assertIsNotNone(bclusterer.fingerprints)

        # Test clustering
        cluster_ids = bclusterer.cluster_molecules(dist_th=0.4)
        self.assertIsNotNone(bclusterer.mol_clusters)
        self.assertEqual(len(bclusterer.mol_clusters), len(np.unique(cluster_ids)))

    def tearDown(self) -> None:
        return super().tearDown()


if __name__ == "__main__":
    unittest.main()
