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


def reset_index_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """Function that checks if the dataframe index is reset. If not, resets it."""
    if not np.array_equal(np.array(range(len(df))), df.index.values):
        logger.warning("DataFrame index is not reset... Will reset it now.")
        df.reset_index(drop=True, inplace=True)
    return df


class ButinaClustering:
    """
    A class for clustering molecules using the Butina algorithm.

    Attributes:
    - smiles_list (List[str]): List of input SMILES strings.
    - njobs (int): Number of jobs for parallel processing.
    - fingerprints (List): Computed fingerprints for the input SMILES.
    - mol_clusters (List): Clusters of molecules after performing clustering.

    Usage example:
    >>> smiles_list = [...]  # Your list of SMILES
    >>> bclusterer = ButinaClustering(smiles_list)
    >>> bclusterer.cluster(cutoff=0.4)
    >>> df_with_clusters = bclusterer.assign_clusters_to_dataframe(
    >>>     df, score_col="score_column_name"
    >>> )
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

    def _compute_fingerprints(self, show_progress=True, radius: int = 2) -> List:
        """Compute fingerprints for the given SMILES list.

        Args:
            radius (int, optional): Radius for Morgan fingerprint. Defaults to 2.

        Returns:
            List: List of computed fingerprints."""
        if show_progress:
            smiles_list = tqdm(self.smiles_list, total=len(self.smiles_list))
        else:
            smiles_list = self.smiles_list
        fingerprints = Parallel(n_jobs=self.njobs)(
            delayed(partial(self.smi2fp, radius=2))(smi) for smi in smiles_list
        )
        return [fp for fp in fingerprints if fp is not None]

    @staticmethod
    def smi2fp(smi, radius: int = 2):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            logger.warning(f"Invalid SMILES detected: {smi}")
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius)

    def cluster(self, cutoff: float = 0.35):
        self.mol_clusters = self.taylor_butina_clustering(
            self.fingerprints, cutoff=cutoff
        )
        return self.mol_clusters

    @staticmethod
    def taylor_butina_clustering(fingerprints: List, cutoff: float = 0.35):
        """Applies the butina clustering algorithm to a list of fingerprints.

        Args:
            fingerprints: fingerprints of compounds to be clustered with the Butina algorith.
            cutoff: when close to 0, only very similar molecules are considered neighbors and
                clustered together. When closer to 1, even dissimilar molecules will be considered
                neighbors and grouped together. Defaults to 0.30.
        """

        def TanimotoDist(fp1, fp2):
            sim = DataStructs.TanimotoSimilarity(fp1, fp2)
            return 1.0 - sim

        dists = []
        nfps = len(fingerprints)
        for i in range(1, nfps):
            sims = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[:i])
            dists.extend([1 - x for x in sims])
        mol_clusters = Butina.ClusterData(
            dists, nfps, cutoff, isDistData=True, distFunc=TanimotoDist
        )
        return mol_clusters

    def assign_clusters_to_dataframe(
        self, df: pd.DataFrame, score_col: str
    ) -> pd.DataFrame:
        """Assigns cluster ids to a dataframe based on the butina clustering.

        Args:
            df: dataframe to assign cluster ids to.
            score_col: column name containing the score of the compounds.

        Raises:
            ValueError: if self.cluster has not been run yet.

        Returns:
            dataframe with cluster ids assigned."""
        if self.mol_clusters is None:
            raise ValueError(
                "Please run clustering first by calling the 'cluster' method."
            )
        return self.assign_cluster_ids(self.mol_clusters, df, score_col)

    @staticmethod
    def assign_cluster_ids(
        mol_clusters, df: pd.DataFrame, score_col: str
    ) -> pd.DataFrame:
        """Assigns cluster ids to a dataframe based on the butina clustering.

        Args:
            mol_clusters: list of clusters from the butina clustering.
            df: dataframe to assign cluster ids to.
            score_col: column name containing the score of the compounds.

        Returns:
            dataframe with cluster ids assigned.
        """
        df = reset_index_if_needed(df)
        cluster_id_list = np.zeros(len(df), dtype=int)
        for cluster_num, cluster in enumerate(mol_clusters):
            cluster_id_list[list(cluster)] = cluster_num

        df = df.assign(cluster_id=cluster_id_list).sort_values(
            by=["cluster_id", score_col]
        )
        return df


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
