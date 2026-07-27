# -*- coding: utf-8 -*-
"""Module containing the ButinaClustering class and related functions."""
from functools import partial
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import DTypeLike
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina

from .base_clusterer import BaseClusterer
from .logger import logger
from .misc import TanimotoDist
from .parallel import ParallelApplier


class ButinaClustering(BaseClusterer):
    """
    A class for clustering molecules using the Butina algorithm.

    Attributes:
    - smiles_list: List of input SMILES strings.
    - njobs: Number of jobs for parallel processing.
    - fingerprints: Computed fingerprints for the input SMILES.
    - mol_clusters: cluster labels from the last clustering call, as an integer
        array with one label per molecule.
    - cluster_members: A tuple of tuples with the indexes within each cluster, set
        by `butina_clustering`. The first index of each tuple is the cluster centroid.
    - similarity_matrix: A similarity matrix for the input SMILES, computed
        after applying calling the `cluster_molecules` or the `compute_similarity_matrix` methods.

    Usage example:
    >>> smiles_list = [...]  # Your list of SMILES
    >>> bclusterer = ButinaClustering(smiles_list)
    >>> clusters = bclusterer.cluster_molecules(dist_th=0.4)
    >>> # if you want to assign the clusters to a dataframe:
    >>> df = df.assign(cluster_id = clusters)
    """

    def __init__(
        self,
        smiles_list: Optional[list[str]] = None,
        fp_func: Optional[Callable] = None,
        np_dtypes: DTypeLike = np.float32,
        njobs: int = 8,
        **fp_kwargs,
    ) -> None:
        """Initialize the Butina clustering class.

        Args:
            smiles_list: List of input SMILES strings.
            fp_func: Custom function to compute fingerprints. If a custom function is used,
                it should take a SMILES string as input and return a RDKit bit vector, which
                will be used by `DataStructs.BulkTanimotoSimilarity`. If None, the default
                Morgan fingerprint will be used.
            np_dtypes: numpy data type for the similarity matrix or arrays.
                Defaults to np.float32.
            njobs: Number of jobs for parallel processing. Defaults to 8.
            fp_kwargs: Additional keyword arguments to be passed to the fingerprint function.
                Examples for the default `GetMorganFingerprintAsBitVect` function are `radius`,
                `nBits`, and `useChirality`.
        """
        super().__init__(smiles_list=smiles_list, njobs=njobs, np_dtypes=np_dtypes)
        self.fp_kwargs = {**fp_kwargs}
        self.cluster_members = None
        self._similarity_matrix_fps = None
        self._set_fp_func(fp_func)
        self.fingerprints = self.calculate_fingerprints()

    def _set_fp_func(self, fp_func: Optional[Callable]):
        if fp_func is not None:
            self.fp_func = partial(fp_func, **self.fp_kwargs)
        else:
            self.fp_func = partial(self.smi2fp, **self.fp_kwargs)

    def calculate_fingerprints(
        self, smiles_list: Optional[list[str]] = None, show_progress=True
    ) -> list:
        """Compute fingerprints for the given SMILES list.

        Args:
            smiles: List of SMILES strings to compute fingerprints for. If None, the
                SMILES list provided at initialization will be used. Defaults to None.
            show_progress: Whether to show a progress bar. Defaults to True.

        Raises:
            ValueError: if any of the SMILES cannot be parsed into molecules.

        Returns:
            list: list of computed fingerprints, one per input SMILES."""
        logger.info("Computing fingerprints...")
        if smiles_list is None:
            smiles_list = self.smiles_list
        applier = ParallelApplier(
            func=self.fp_func,
            iterable=smiles_list,
            n_jobs=self.njobs,
            show_progress=show_progress,
        )
        fingerprints = applier()
        invalid = [
            (idx, smi)
            for idx, (smi, fp) in enumerate(zip(smiles_list, fingerprints))
            if fp is None
        ]
        if invalid:
            details = ", ".join(f"{idx}: {smi}" for idx, smi in invalid)
            logger.error(f"Could not parse: {details}!!\nRemove invalid SMILES...")
            raise ValueError(
                f"Could not parse SMILES into molecules at index: {details}"
            )
        return fingerprints

    @staticmethod
    def smi2fp(smi, radius: int = 2, nBits=2048, useChirality=True, **kwargs):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            logger.error(f"Invalid SMILES detected: {smi}")
            return None
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=nBits, includeChirality=useChirality, **kwargs
        )
        return morgan_gen.GetFingerprint(mol)

    def cluster_molecules(self, algorithm: str = "Butina", **kwargs) -> np.ndarray:
        """Cluster the molecules with the chosen algorithm.

        Unlike `MCSClustering` and `RascalMCES`, which raise when the similarity
        matrix is missing, this computes it on demand: it is Tanimoto over the
        fingerprints already held by the instance, so it is cheap and involves no
        choice of similarity metric that the caller should be making.

        Args:
            algorithm: algorithm to use for clustering. On top of the algorithms shared
                by all clustering classes ('DBSCAN', 'Hierarchical', 'Spectral',
                'GraphBased'), 'Butina' runs the Taylor-Butina algorithm on the
                fingerprints. Defaults to "Butina".
            kwargs: keyword arguments for the chosen algorithm, e.g. `dist_th` for
                'Butina' or `n_clusters` for 'Spectral'.

        Returns:
            labels: array of cluster labels, one per molecule.
        """
        if algorithm == "Butina":
            return self.butina_clustering(**kwargs)
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()
        return super().cluster_molecules(algorithm=algorithm, **kwargs)

    def butina_clustering(self, dist_th: float = 0.35) -> np.ndarray:
        """cluster the molecules based on the butina algorithm.

        The grouping of indices produced by the algorithm is kept in
        `self.cluster_members`, where the first index of each tuple is the
        cluster centroid.

        Args:
            dist_th: tanimoto distance threshold. for the butina clustering algorithm.
                The lower the value, the higher the amount of obtained clusters (and
                the more similar the compounds in each cluster). Defaults to 0.35.

        Returns:
            labels: array of cluster labels, one per molecule.
        """
        self.cluster_members = self._taylor_butina_clustering(
            self.fingerprints, dist_th=dist_th
        )
        cluster_id_list = np.zeros(len(self.fingerprints), dtype=int)
        for cluster_num, cluster in enumerate(self.cluster_members):
            cluster_id_list[list(cluster)] = cluster_num
        return self._store_labels(cluster_id_list)

    def compute_similarity_matrix(self, fps: Optional[list] = None) -> tuple[tuple]:
        """Applies the butina clustering algorithm to a list of fingerprints.

        Args:
            fps: fingerprints of compounds to be clustered with the Butina algorith. If
                None, the fingerprints calculated upon initialization will be used. Defaults to None.

        Returns:
            A tuple of tuples containing the indices of the compounds in each cluster.
        """

        if fps is None:
            fps = self.fingerprints
        similarities = []
        size = len(fps)
        simi_matrix = np.eye(size, dtype=self.np_dtypes)
        # calculate the builk tanimoto similarities
        for i in range(0, size):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :])
            similarities.extend(sims)
        similarities = np.array(similarities).flatten()
        # populate the similarity matrix
        r, c = np.triu_indices(size, 1)  # row, column indices, respectively
        simi_matrix[r, c] = similarities
        # add values for the lower triangle
        simi_matrix += simi_matrix.T - np.eye(size, dtype=self.np_dtypes)
        self.similarity_matrix = simi_matrix
        self._similarity_matrix_fps = fps
        return simi_matrix

    def _taylor_butina_clustering(
        self, fps: list, dist_th: float = 0.35
    ) -> tuple[tuple]:
        """Applies the butina clustering algorithm to a list of fingerprints.

        Args:
            fps: fingerprints of compounds to be clustered with the Butina algorith. If
                None, the fingerprints calculated upon initialization will be used. Defaults to None.
            dist_th: distance threshold. when close to 0, only very similar molecules are considered
                neighbors and clustered together. When closer to 1, even dissimilar molecules will
                be considered neighbors and grouped together. Defaults to 0.35.

        Returns:
            A tuple of tuples containing the indices of the compounds in each cluster.
        """
        if fps is None:
            fps = self.fingerprints
        # only reuse the cached matrix if it was computed from these very fingerprints,
        # otherwise it would mislabel the molecules being clustered here.
        if self.similarity_matrix is None or fps is not self._similarity_matrix_fps:
            self.compute_similarity_matrix(fps)
        size = len(fps)

        similarities = self.similarity_matrix[np.triu_indices(size, 1)].flatten()

        mol_clusters = Butina.ClusterData(  # now we cluster the data
            1 - similarities,  # convert to distance
            size,
            dist_th,
            isDistData=True,
            distFunc=TanimotoDist,
        )
        return mol_clusters


def plot_butina_scatter(
    best_clusters_df: pd.DataFrame,
    cutoff: float,
    cluster_col="cluster_id",
    score_col="pchembl_value_median",
    color_col: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the clustered molecules from the Butina clustering.

    Args:
        best_clusters_df: Dataframe output from Butina clustering.
        cutoff: Cutoff used in the Butina clustering.
        cluster_col: Column of the dataframe with cluster ids. Defaults to "cluster_id".
        score_col: Column with the score for the y-axis. Defaults to "pchembl_value_median".
        color_col: Column to color the plot with. Defaults to None.

    Returns:
        tuple[plt.Figure, plt.Axes]: Matplotlib figure and axes objects of the scatter plot.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    if color_col is not None:
        color_col = best_clusters_df[color_col]
    ax.scatter(
        best_clusters_df[cluster_col],
        best_clusters_df[score_col],
        alpha=0.3,
        c=color_col,
        cmap="plasma",
    )
    if color_col is not None:
        cbar = fig.colorbar(ax.collections[0], ax=ax)
        cbar.set_label(color_col.name)
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("pChEMBL value (median)")
    ax.set_title(f"Clustering with cutoff {cutoff}")
    # make a lineplot going through the median of each cluster
    ax.plot(
        best_clusters_df.groupby("cluster_id")[score_col].max(),
        c="black",
        lw=2,
        label="Max pChEMBL value in cluster",
        alpha=0.2,
    )
    ax.legend(
        bbox_to_anchor=(0.55, -0.1),
        loc="lower right",
        bbox_transform=fig.transFigure,
    )
    return fig, ax
