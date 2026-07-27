# -*- coding: utf-8 -*-
"""Regression tests, one per defect fixed while bringing this branch up to date.

Each test reproduces a specific bug that shipped on the branch and asserts the
corrected behaviour, so a reintroduction fails loudly rather than silently
returning wrong clusters.
"""
import unittest

import numpy as np

from MolClusterkit import ButinaClustering, MCSClustering, RascalMCES
from MolClusterkit.base_clusterer import BaseClusterer
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


class TestClusterLabelsAreUniform(unittest.TestCase):
    """Every clustering method returns one integer label per molecule, in an
    array, and leaves that same array in `mol_clusters`. DBSCAN and GraphBased
    used to hand back plain lists, and `mol_clusters` used to mean a tuple of
    per-cluster index tuples after Butina but per-molecule labels otherwise."""

    smiles = ["CCO", "CCN", "CCS", "c1ccccc1", "c1ccccc1O", "CCCCCC"]

    def test_every_algorithm_returns_a_label_array(self):
        clusterer = ButinaClustering(self.smiles, njobs=1)
        algorithms = {
            "Butina": {"dist_th": 0.4},
            "DBSCAN": {"eps": 0.5, "min_samples": 2},
            "Hierarchical": {"t": 2},
            "HierarchicalSilhouette": {"max_clusters": 3},
            "Spectral": {"n_clusters": 2, "random_state": 0},
            "GraphBased": {"threshold": 0.3},
        }
        for algorithm, kwargs in algorithms.items():
            with self.subTest(algorithm=algorithm):
                labels = clusterer.cluster_molecules(algorithm=algorithm, **kwargs)
                self.assertIsInstance(labels, np.ndarray)
                self.assertTrue(np.issubdtype(labels.dtype, np.integer))
                self.assertEqual(len(labels), len(self.smiles))
                np.testing.assert_array_equal(clusterer.mol_clusters, labels)

    def test_butina_index_groups_live_in_cluster_members(self):
        clusterer = ButinaClustering(self.smiles, njobs=1)
        labels = clusterer.cluster_molecules(dist_th=0.4)
        # every molecule appears exactly once across the member tuples
        members = [idx for cluster in clusterer.cluster_members for idx in cluster]
        self.assertCountEqual(members, range(len(self.smiles)))
        # and the groups agree with the labels they were derived from
        self.assertEqual(len(clusterer.cluster_members), len(np.unique(labels)))
        for cluster_num, cluster in enumerate(clusterer.cluster_members):
            for idx in cluster:
                self.assertEqual(labels[idx], cluster_num)


class TestRascalOptions(unittest.TestCase):
    """RascalMCES used to swallow unknown constructor kwargs through **kwargs, so
    `timeout=` was silently ignored. It is now a real, validated parameter."""

    def test_timeout_reaches_the_options_object(self):
        rascal = RascalMCES(["CCO", "CCN"], timeout=5, njobs=1)
        self.assertEqual(rascal._make_opts().timeout, 5)

    def test_per_call_timeout_overrides_the_instance(self):
        rascal = RascalMCES(["CCO", "CCN"], timeout=5, njobs=1)
        self.assertEqual(rascal._make_opts(timeout=30).timeout, 30)

    def test_unknown_kwargs_are_rejected(self):
        with self.assertRaises(TypeError):
            RascalMCES(["CCO", "CCN"], njobs=1, notAnOption=True)

    def test_invalid_timeout_rejected(self):
        with self.assertRaises(TypeError):
            RascalMCES(["CCO", "CCN"], timeout=1.5, njobs=1)
        with self.assertRaises(ValueError):
            RascalMCES(["CCO", "CCN"], timeout=0, njobs=1)


class TestMCSAsStandaloneFrontEnd(unittest.TestCase):
    """MCSClustering is usable without a smiles_list, as a configured front-end to
    rdFMCS. Bad input on that path used to surface RDKit's bare "molecule is None"
    or a raw KeyError, neither of which says what was wrong."""

    series = ["c1ccccc1C(=O)O", "c1ccccc1CC(=O)O", "c1ccccc1CCC(=O)O"]

    def test_mcs_in_many_without_a_smiles_list(self):
        mcs = MCSClustering(ringMatchesRingOnly=True, completeRingsOnly=True, timeout=5)
        self.assertIsNone(mcs.smiles_list)
        result = mcs.mcs_in_many(self.series)
        self.assertGreater(result.numAtoms, 0)
        self.assertTrue(result.smartsString)

    def test_comparison_options_reach_rdkit(self):
        strict = MCSClustering(atomCompare="CompareElements", timeout=5)
        loose = MCSClustering(atomCompare="CompareAny", timeout=5)
        pair = ["CCCCO", "CCCCN"]
        # CompareAny lets the terminal heteroatoms match, CompareElements does not
        self.assertGreater(
            loose.mcs_in_many(pair).numAtoms, strict.mcs_in_many(pair).numAtoms
        )

    def test_invalid_smiles_named_in_mcs_in_many(self):
        mcs = MCSClustering(timeout=5)
        with self.assertRaises(ValueError) as ctx:
            mcs.mcs_in_many(["c1ccccc1", "not_a_smiles"])
        self.assertIn("1: not_a_smiles", str(ctx.exception))

    def test_unknown_option_name_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            MCSClustering(atomCompare="CompareNonsense")
        self.assertIn("CompareNonsense", str(ctx.exception))

    def test_unknown_setting_lists_every_supported_one(self):
        with self.assertRaises(ValueError) as ctx:
            MCSClustering(maximiseBonds=True)  # misspelling of maximizeBonds
        # the message used to omit the non-comparison settings entirely
        self.assertIn("maximizeBonds", str(ctx.exception))
        self.assertIn("atomCompare", str(ctx.exception))


class TestFuzzyClusteringSkipsSimilarityMatrix(unittest.TestCase):
    """fuzzy_mces_clustering used to compute the full pairwise MCES similarity
    matrix and then never read it, since RascalCluster takes only the molecules.
    On a real library that is a long computation done for nothing."""

    def test_no_similarity_matrix_is_computed(self):
        rascal = RascalMCES(["CCO", "CCN", "CCS", "c1ccccc1"], njobs=1)

        def fail(*args, **kwargs):
            raise AssertionError(
                "fuzzy_mces_clustering computed the similarity matrix it never uses"
            )

        rascal.compute_similarity_matrix = fail
        clusters = rascal.fuzzy_mces_clustering(cutoff=0.5)
        self.assertIsNone(rascal.similarity_matrix)
        self.assertGreater(len(clusters), 0)


if __name__ == "__main__":
    unittest.main()
