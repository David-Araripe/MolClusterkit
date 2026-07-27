# -*- coding: utf-8 -*-
"""Tests for the MolClusterkit.butina module."""
import unittest
from pathlib import Path

import numpy as np

from MolClusterkit.butina import ButinaClustering


class TestButina(unittest.TestCase):
    def setUp(self) -> None:
        self.testroot = Path(__file__).parent
        with (self.testroot / "resources/HIV-sample.smi").open("r") as f:
            self.smiles = f.read().splitlines()
        np.random.seed(42)
        self.scores = np.random.randint(0, 10, size=len(self.smiles))

    def test_ButinaClustering(self):
        smiles_list = ["CCO", "CCN", "CCS"]  # Some test SMILES strings
        bclusterer = ButinaClustering(smiles_list)

        # Check initialization
        self.assertEqual(bclusterer.smiles_list, smiles_list)
        self.assertEqual(bclusterer.njobs, 8)  # default value
        self.assertIsNotNone(bclusterer.fingerprints)

        # Test clustering
        cluster_ids = bclusterer.cluster_molecules(dist_th=0.4)
        # mol_clusters holds one label per molecule...
        self.assertEqual(len(bclusterer.mol_clusters), len(smiles_list))
        # ...while cluster_members groups the molecule indices by cluster.
        self.assertEqual(len(bclusterer.cluster_members), len(np.unique(cluster_ids)))

    def tearDown(self) -> None:
        return super().tearDown()


if __name__ == "__main__":
    unittest.main()
