# -*- coding: utf-8 -*-
"""Module containing the ButinaClustering class and related functions."""
from functools import partial
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
from tqdm import tqdm

from .misc import TanimotoDist


class ButinaClustering:
    """
    A class for clustering molecules using the Butina algorithm.

    Attributes:
    - smiles_list (List[str]): List of input SMILES strings.
    - njobs (int): Number of jobs for parallel processing.
    - fingerprints (List): Computed fingerprints for the input SMILES.
    - mol_clusters (Tuple[Tuple]): A tuple of tuples with indexes within each cluster.
    - similarity_matrix (np.ndarray): A similarity matrix for the input SMILES, computed
        after applying calling the `cluster_molecules` or the `taylor_butina_clustering` methods.

    Usage example:
    >>> smiles_list = [...]  # Your list of SMILES
    >>> bclusterer = ButinaClustering(smiles_list)
    >>> clusters = bclusterer.cluster_molecules(dist_th=0.4)
    >>> # if you want to assign the clusters to a dataframe:
    >>> df = df.assign(cluster_id = clusters)
    """

    def __init__(self, smiles_list: List[str], njobs: int = 8):
        """Initialize the Butina clustering class.

        Args:
            smiles_list (List[str]): List of input SMILES strings.
            njobs (int, optional): Number of jobs for parallel processing. Defaults to 8.
        """
        self.smiles_list = smiles_list
        self.njobs = njobs
        self.fingerprints = self._compute_fingerprints()
        self.mol_clusters = None
        self.similarity_matrix = None

    def _compute_fingerprints(self, show_progress=True, radius: int = 2) -> List:
        """Compute fingerprints for the given SMILES list.

        Args:
            radius (int, optional): Radius for Morgan fingerprint. Defaults to 2.

        Returns:
            List: List of computed fingerprints."""
        logger.info("Computing fingerprints...")
        if show_progress:
            smiles_list = tqdm(self.smiles_list, total=len(self.smiles_list))
        else:
            smiles_list = self.smiles_list
        fingerprints = Parallel(n_jobs=self.njobs)(
            delayed(partial(self.smi2fp, radius=radius))(smi) for smi in smiles_list
        )
        return [fp for fp in fingerprints if fp is not None]

    @staticmethod
    def smi2fp(smi, radius: int = 2):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            logger.error(f"Invalid SMILES detected: {smi}")
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius)

    def cluster_molecules(self, dist_th: float = 0.35):
        """cluster the molecules based on the butina algorithm. Returns the clusters
        as a list of lists of indices.

        Args:
            dist_th: tanimoto distance threshold. for the butina clustering algorithm.
                The lower the value, the higher the amount of obtained clusters (and
                the more similar the compounds in each cluster). Defaults to 0.35.

        Returns:
            np.ndarray: array of cluster ids for each molecule.
        """
        self.mol_clusters = self.taylor_butina_clustering(
            self.fingerprints, dist_th=dist_th
        )
        cluster_id_list = np.zeros(len(self.fingerprints), dtype=int)
        for cluster_num, cluster in enumerate(self.mol_clusters):
            cluster_id_list[list(cluster)] = cluster_num
        return cluster_id_list

    def taylor_butina_clustering(
        self, fps: List, dist_th: float = 0.35
    ) -> Tuple[Tuple]:
        """Applies the butina clustering algorithm to a list of fingerprints.

        Args:
            fps: fingerprints of compounds to be clustered with the Butina algorith.
            dist_th: distance threshold. when close to 0, only very similar molecules are considered
                neighbors and clustered together. When closer to 1, even dissimilar molecules will
                be considered neighbors and grouped together. Defaults to 0.35.

        Returns:
            A tuple of tuples containing the indices of the compounds in each cluster.
        """

        similarities = []
        nfps = len(fps)
        simi_matrix = np.eye(nfps)
        # calculate the builk tanimoto similarities
        for i in range(0, nfps):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :])
            similarities.extend(sims)
        similarities = np.array(similarities).flatten()
        # populate the similarity matrix
        r, c = np.triu_indices(nfps, 1)  # row, column indices, respectively
        simi_matrix[r, c] = similarities
        # add values for the lower triangle
        simi_matrix += simi_matrix.T - np.eye(nfps)
        mol_clusters = Butina.ClusterData(  # now we cluster the data
            1 - similarities,  # convert to distance
            nfps,
            dist_th,
            isDistData=True,
            distFunc=TanimotoDist,
        )
        self.similarity_matrix = simi_matrix
        return mol_clusters


def plot_butina_scatter(
    best_clusters_df: pd.DataFrame,
    cutoff: float,
    cluster_col="cluster_id",
    score_col="pchembl_value_median",
    color_col: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the clustered molecules from the Butina clustering.

    Args:
        best_clusters_df (pd.DataFrame): Dataframe output from Butina clustering.
        cutoff (float): Cutoff used in the Butina clustering.
        cluster_col (str, optional): Column of the dataframe with cluster ids. Defaults to "cluster_id".
        score_col (str, optional): Column with the score for the y-axis. Defaults to "pchembl_value_median".
        color_col (str, optional): Column to color the plot with. Defaults to None.

    Returns:
        Tuple[plt.Figure, plt.Axes]: Matplotlib figure and axes objects of the scatter plot.
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
