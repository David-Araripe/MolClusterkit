# -*- coding: utf-8 -*-
"""Regression tests, one per defect fixed while bringing this branch up to date.

Each test reproduces a specific bug that shipped on the branch and asserts the
corrected behaviour, so a reintroduction fails loudly rather than silently
returning wrong clusters.
"""
import unittest

import numpy as np
import pandas as pd

from MolClusterkit import ButinaClustering, MCSClustering
from MolClusterkit.base_clusterer import BaseClusterer
from MolClusterkit.best_picker import butina_based_clustering, mcs_based_clustering
from MolClusterkit.parallel import ParallelApplier


def _square(x):
    """Module-level (picklable) function for the ParallelApplier tests."""
    return x * x


class TestButinaFingerprintKwargs(unittest.TestCase):
    """The passthrough this branch is named for: fingerprint kwargs must reach
    the generator instead of raising TypeError."""

    def test_fp_kwargs_are_honoured(self):
        bclusterer = ButinaClustering(
            ["CCO", "CCN", "c1ccccc1", "CCCC"], nBits=1024, radius=3, njobs=1
        )
        self.assertEqual(len(bclusterer.fingerprints), 4)
        self.assertEqual(bclusterer.fingerprints[0].GetNumBits(), 1024)


class TestInvalidSmilesRaise(unittest.TestCase):
    """Unparseable SMILES must raise (naming the offending rows) so cluster ids
    stay aligned with the input, rather than being silently dropped."""

    def test_invalid_smiles_raise_with_indices(self):
        with self.assertRaises(ValueError) as ctx:
            ButinaClustering(["CCO", "not_a_smiles", "CCN"], njobs=1)
        self.assertIn("1", str(ctx.exception))
        self.assertIn("not_a_smiles", str(ctx.exception))


class TestBestPickerPickBest(unittest.TestCase):
    """pick_best=True must return one row per cluster with cluster_id preserved
    as a column and original dtypes intact."""

    def setUp(self):
        self.df = pd.DataFrame(
            {
                "smiles": ["CCO", "CCN", "CCS", "c1ccccc1", "CC(=O)O"],
                "score": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )

    def test_butina_pick_best(self):
        result = butina_based_clustering(
            self.df, smiles_col="smiles", score_col="score", pick_best=True, njobs=1
        )
        self.assertIn("cluster_id", result.columns)
        self.assertEqual(result["score"].dtype, np.float64)
        # one row per cluster
        self.assertEqual(len(result), result["cluster_id"].nunique())

    def test_default_call_needs_no_score(self):
        # score_col is optional per the docstring; the default must not build a
        # 'None > 7.0' query.
        result = butina_based_clustering(self.df, smiles_col="smiles", njobs=1)
        self.assertIn("cluster_id", result.columns)
        self.assertEqual(len(result), len(self.df))


class TestMcsDefaults(unittest.TestCase):
    """mcs_based_clustering with only the required argument must run: a float
    timeout used to crash RDKit's FindMCS at the default of 1.5s."""

    def test_defaults_run(self):
        df = pd.DataFrame({"smiles": ["CCO", "CCN", "CCS", "c1ccccc1"]})
        result = mcs_based_clustering(df, smiles_col="smiles", n_jobs=1)
        self.assertIn("cluster_id", result.columns)
        self.assertEqual(len(result), len(df))


class TestMcsTimeoutCoercion(unittest.TestCase):
    """Any numeric timeout must be coerced to a positive int; sub-second values
    round up rather than becoming 0."""

    def test_float_timeout_coerced_up(self):
        self.assertEqual(MCSClustering(["CCO", "CCN"], timeout=1.5, njobs=1).timeout, 2)
        self.assertEqual(MCSClustering(["CCO", "CCN"], timeout=0.2, njobs=1).timeout, 1)

    def test_non_positive_timeout_raises(self):
        with self.assertRaises(ValueError):
            MCSClustering(["CCO", "CCN"], timeout=0, njobs=1)


class TestMcsSimilarityMetric(unittest.TestCase):
    """Both similarity metrics must be reachable through compute_similarity_matrix,
    and an unknown metric must raise rather than UnboundLocalError."""

    def test_smaller_mces_selectable(self):
        mcs = MCSClustering(["c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "CCN"], timeout=5, njobs=1)
        _, matrix = mcs.compute_similarity_matrix(similarity_metric="smaller/mces")
        # benzene is fully contained in aspirin, so containment saturates at 1.0
        self.assertAlmostEqual(matrix[0, 1], 1.0, places=6)

    def test_unknown_metric_raises(self):
        mcs = MCSClustering(timeout=5, njobs=1)
        with self.assertRaises(ValueError):
            mcs.mcs_similarity(("CCO", "CCN"), similarity_metric="bogus")


class TestParallelApplier(unittest.TestCase):
    """n_jobs=-1 must resolve to a real worker count instead of crashing, and an
    empty iterable must fail with an actionable message."""

    def test_n_jobs_minus_one(self):
        applier = ParallelApplier(func=_square, iterable=[1, 2, 3], n_jobs=-1)
        self.assertEqual(applier(), [1, 4, 9])
        self.assertGreaterEqual(applier.n_workers, 1)

    def test_empty_iterable_raises(self):
        with self.assertRaises(ValueError):
            ParallelApplier(func=_square, iterable=[], n_jobs=1)


class TestButinaSharedAlgorithms(unittest.TestCase):
    """ButinaClustering must reach the inherited clustering algorithms while its
    default 'Butina' path keeps working."""

    def setUp(self):
        self.smiles = ["CCO", "CCN", "CCS", "c1ccccc1", "CC(=O)O", "CCCC", "CCCCCC", "c1ccncc1"]

    def test_default_butina_still_works(self):
        labels = ButinaClustering(self.smiles, njobs=1).cluster_molecules(dist_th=0.4)
        self.assertEqual(len(labels), len(self.smiles))

    def test_shared_algorithm_reachable(self):
        labels = ButinaClustering(self.smiles, njobs=1).cluster_molecules(
            algorithm="Hierarchical", t=3
        )
        self.assertEqual(len(labels), len(self.smiles))

    def test_unknown_algorithm_raises(self):
        with self.assertRaises(ValueError):
            ButinaClustering(self.smiles, njobs=1).cluster_molecules(algorithm="Nope")


class TestHierarchicalDistanceMatrix(unittest.TestCase):
    """linkage must receive a condensed distance matrix (no ClusterWarning) and
    the distance threshold must actually be respected."""

    def _two_block_clusterer(self):
        # two tight blocks of 3, dissimilar across blocks
        sim = np.array(
            [
                [1.0, 0.9, 0.9, 0.1, 0.1, 0.1],
                [0.9, 1.0, 0.9, 0.1, 0.1, 0.1],
                [0.9, 0.9, 1.0, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 1.0, 0.9, 0.9],
                [0.1, 0.1, 0.1, 0.9, 1.0, 0.9],
                [0.1, 0.1, 0.1, 0.9, 0.9, 1.0],
            ]
        )
        clusterer = BaseClusterer(smiles_list=None, njobs=1)
        clusterer.similarity_matrix = sim
        return clusterer

    def test_threshold_respected_without_warning(self):
        import warnings

        clusterer = self._two_block_clusterer()
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any ClusterWarning fails the test
            labels = clusterer.hierarchical_clustering(
                t=0.5, method="average", criterion="distance"
            )
        # exactly two clusters, split along the block boundary
        self.assertEqual(len(np.unique(labels)), 2)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[1], labels[2])
        self.assertNotEqual(labels[2], labels[3])

    def test_silhouette_sweep_bounded_by_samples(self):
        # 6 samples with default max_clusters=20 used to crash at n_labels==n_samples
        clusterer = self._two_block_clusterer()
        labels = clusterer.hierarchical_silhouette_clustering()
        self.assertEqual(len(labels), 6)

    def test_too_few_samples_raises(self):
        sim = np.array([[1.0, 0.5], [0.5, 1.0]])
        clusterer = BaseClusterer(smiles_list=None, njobs=1)
        clusterer.similarity_matrix = sim
        with self.assertRaises(ValueError):
            clusterer.hierarchical_silhouette_clustering()


if __name__ == "__main__":
    unittest.main()
