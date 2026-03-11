# -*- coding: utf-8 -*-
"""Tests for the MolClusterkit.rascal module."""
import unittest

import numpy as np

try:
    from rdkit.Chem import rdRascalMCES  # noqa: F401

    HAS_RASCAL = True
except ImportError:
    HAS_RASCAL = False

from MolClusterkit.rascal import RascalMCES


@unittest.skipUnless(HAS_RASCAL, "rdRascalMCES not available")
class TestRascalMCES(unittest.TestCase):
    def setUp(self):
        self.smiles = ["CCO", "CCN", "CCS", "c1ccccc1", "CC(=O)O"]

    def test_init_default(self):
        rascal = RascalMCES()
        self.assertIsNone(rascal.smiles_list)
        self.assertIsNone(rascal.similarity_matrix)
        self.assertEqual(rascal.njobs, 8)
        self.assertIn("similarityThreshold", rascal.opts_dict)

    def test_init_with_smiles(self):
        rascal = RascalMCES(smiles_list=self.smiles)
        self.assertEqual(rascal.smiles_list, self.smiles)

    def test_mces_similarity_johnson(self):
        rascal = RascalMCES(similarityThreshold=0.0)
        smarts, score = rascal.mces_similarity(("CCO", "CCN"))
        self.assertIsInstance(smarts, str)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_mces_similarity_smaller_mces(self):
        rascal = RascalMCES(similarityThreshold=0.0)
        smarts, score = rascal.mces_similarity(
            ("CCO", "CCN"), similarity_metric="smaller/mces"
        )
        self.assertIsInstance(smarts, str)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_mces_similarity_invalid_smiles(self):
        rascal = RascalMCES()
        with self.assertRaises(ValueError):
            rascal.mces_similarity(("CCO", "INVALID_SMILES_XYZ"))

    def test_mces_similarity_identical(self):
        rascal = RascalMCES(similarityThreshold=0.0)
        _, score = rascal.mces_similarity(("CCO", "CCO"))
        self.assertAlmostEqual(score, 1.0)

    def test_compute_similarity_matrix(self):
        smiles = self.smiles[:3]
        rascal = RascalMCES(smiles_list=smiles, similarityThreshold=0.0)
        smarts_matrix, simi_matrix = rascal.compute_similarity_matrix(
            show_progress=False
        )
        n = len(smiles)
        self.assertEqual(simi_matrix.shape, (n, n))
        self.assertEqual(smarts_matrix.shape, (n, n))
        # Diagonal should be 1.0
        np.testing.assert_array_almost_equal(np.diag(simi_matrix), np.ones(n))
        # Should be symmetric
        np.testing.assert_array_almost_equal(simi_matrix, simi_matrix.T)

    def test_make_opts_overrides(self):
        rascal = RascalMCES(similarityThreshold=0.5)
        opts = rascal._make_opts(similarityThreshold=0.3, timeout=30)
        self.assertAlmostEqual(opts.similarityThreshold, 0.3)
        self.assertEqual(opts.timeout, 30)

    def test_no_smiles_compute_matrix_raises(self):
        rascal = RascalMCES()
        with self.assertRaises(ValueError):
            rascal.compute_similarity_matrix()


if __name__ == "__main__":
    unittest.main()
