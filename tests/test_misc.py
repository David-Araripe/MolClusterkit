# -*- coding: utf-8 -*-
"""Tests for the MolClusterkit.misc module."""
import unittest

import pandas as pd
from rdkit.Chem import rdFingerprintGenerator, MolFromSmiles

from MolClusterkit.misc import TanimotoDist, find_smiles_column, reset_index_if_needed


class TestFindSmilesColumn(unittest.TestCase):
    def test_single_match(self):
        df = pd.DataFrame({"SMILES": ["CCO", "CCN"], "score": [1.0, 2.0]})
        self.assertEqual(find_smiles_column(df), "SMILES")

    def test_case_insensitive(self):
        df = pd.DataFrame({"smiles": ["CCO"], "score": [1.0]})
        self.assertEqual(find_smiles_column(df), "smiles")

    def test_substring_match(self):
        df = pd.DataFrame({"canonical_smiles": ["CCO"], "score": [1.0]})
        self.assertEqual(find_smiles_column(df), "canonical_smiles")

    def test_no_match_raises(self):
        df = pd.DataFrame({"mol": ["CCO"], "score": [1.0]})
        with self.assertRaises(ValueError):
            find_smiles_column(df)

    def test_multiple_matches_raises(self):
        df = pd.DataFrame(
            {"SMILES": ["CCO"], "canonical_smiles": ["CCO"], "score": [1.0]}
        )
        with self.assertRaises(ValueError):
            find_smiles_column(df)


class TestResetIndexIfNeeded(unittest.TestCase):
    def test_needs_reset(self):
        df = pd.DataFrame({"col1": [1, 2, 3, 4, 5]})
        df.drop(2, inplace=True)
        result = reset_index_if_needed(df)
        self.assertTrue(all(result.index == [0, 1, 2, 3]))

    def test_already_reset(self):
        df = pd.DataFrame({"col1": [1, 2, 3]})
        result = reset_index_if_needed(df)
        self.assertTrue(all(result.index == [0, 1, 2]))


class TestTanimotoDist(unittest.TestCase):
    def _make_fp(self, smi):
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        return gen.GetFingerprint(MolFromSmiles(smi))

    def test_distance(self):
        fp1 = self._make_fp("CCO")
        fp2 = self._make_fp("CCN")
        dist = TanimotoDist(fp1, fp2)
        self.assertGreater(dist, 0.0)
        self.assertLess(dist, 1.0)

    def test_identical_is_zero(self):
        fp = self._make_fp("CCO")
        self.assertAlmostEqual(TanimotoDist(fp, fp), 0.0)


if __name__ == "__main__":
    unittest.main()
