# -*- coding: utf-8 -*-
"""Tests for the MolClusterkit.misc module."""
import unittest

from rdkit.Chem import MolFromSmiles, rdFingerprintGenerator

from MolClusterkit.misc import TanimotoDist


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
