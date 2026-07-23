# -*- coding: utf-8 -*-
"""Module containing high-level functions that cluster a set of compounds and,
optionally, pick the best scoring compound of each cluster."""
from typing import Optional, Union

import numpy as np
import pandas as pd

from .butina import ButinaClustering
from .logger import logger
from .mcs import MCSClustering
from .misc import find_smiles_column


def _apply_score_cutoff(
    data: pd.DataFrame, score_col: Optional[str], score_cutoff
) -> pd.DataFrame:
    """Keep only the rows whose score exceeds the cutoff.

    The filter is applied only when both `score_col` and `score_cutoff` are given.

    Args:
        data: dataframe to filter.
        score_col: column holding the scores, or None to skip filtering.
        score_cutoff: keep rows where score_col > score_cutoff, or None to skip.

    Raises:
        ValueError: if the cutoff removes every row.

    Returns:
        pd.DataFrame: the filtered dataframe.
    """
    if score_col is not None and score_cutoff is not None:
        data = data.query(f"{score_col} > {score_cutoff}")
        if data.shape[0] == 0:
            raise ValueError(
                f"No compounds with a score above {score_cutoff} were found."
            )
    return data


def _pick_best_per_cluster(
    data: pd.DataFrame, score_col: Optional[str], pick_best: bool
) -> pd.DataFrame:
    """Reduce each cluster to its highest-scoring compound.

    Args:
        data: dataframe with a `cluster_id` column.
        score_col: column holding the scores, or None.
        pick_best: whether to reduce to the best per cluster.

    Returns:
        pd.DataFrame: one row per cluster when picking is requested, else `data`.
    """
    if pick_best and score_col is not None:
        data = data.loc[data.groupby("cluster_id")[score_col].idxmax()]
    return data


def butina_based_clustering(
    data: Union[pd.DataFrame, list[str], np.ndarray],
    smiles_col: Optional[str],
    score_col: Optional[str] = None,
    score_cutoff=None,
    dist_th=0.25,
    njobs=8,
    pick_best=False,
) -> pd.DataFrame:
    """Perform butina clustering on the given dataframe and return the best scoring

    Args:
        data: dataframe to be used for clustering.
        smiles_col: column name containing the smiles structures of the compounds.
            If is `None`, will try to find the smiles column based on a simple regex search.
        score_col: column name containing the scores. Optional. Defaults to None.
        score_cutoff: threshold to be used on score_col; keep only above cutoff. Only
            applied when score_col is also given. Defaults to None.
        dist_th: tanimoto distance threshold for the butina clustering. Defaults to 0.25.
        njobs: number of jobs for parallelization. Defaults to 8.
        pick_best: whether to pick the best scoring compound from each cluster. Defaults to False.

    Returns:
        pd.DataFrame: dataframe with the best scoring compounds from each cluster.
    """
    if any([isinstance(data, list), isinstance(data, np.ndarray)]):
        smiles_list = data
        bclusterer = ButinaClustering(smiles_list, njobs=njobs)
        cluster_ids = bclusterer.cluster_molecules(dist_th=dist_th)
        # make a dataframe with the smiles and the cluster ID
        data = pd.DataFrame(
            {"smiles": smiles_list, "cluster_id": cluster_ids}, index=None
        )
    elif isinstance(data, pd.DataFrame):
        if smiles_col is None:
            smiles_col = find_smiles_column(data)
        data = _apply_score_cutoff(data, score_col, score_cutoff)
        smiles_list = data[smiles_col].tolist()
        bclusterer = ButinaClustering(smiles_list, njobs=njobs)
        cluster_ids = bclusterer.cluster_molecules(dist_th=dist_th)
        data = data.assign(cluster_id=cluster_ids)
        data = _pick_best_per_cluster(data, score_col, pick_best)
        logger.info(f"Total amount of clusters: {len(data.cluster_id.unique())}")
    else:
        raise ValueError(
            "Data should be a pandas dataframe or a .smi file (list or array), but "
            f"{type(data)} was provided."
        )
    return data


def mcs_based_clustering(
    data,
    smiles_col: Optional[str],
    score_col: Optional[str] = None,
    score_cutoff=None,
    algorithm="DBSCAN",
    pick_best=False,
    n_jobs=8,
    timeout=15,
    mcs_kwargs=None,
    **kwargs,
) -> pd.DataFrame:
    """Perform MCS based clustering on the given dataframe.

    Args:
        data: dataframe to apply MCS clustering to.
        smiles_col: column name containing the smiles structures of the compounds.
            If is `None`, will try to find the smiles column based on a simple regex search.
        score_col: column name containing the scores. Optional. Defaults to None.
        score_cutoff: threshold to be used on score_col; keep only above cutoff. Only
            applied when score_col is also given. Defaults to None.
        algorithm: algorithm to use for clustering.
        pick_best: whether to pick the best scoring compound from each cluster. Defaults to False.
        n_jobs: number of jobs for parallelization. Defaults to 8.
        timeout: wall-time in seconds threshold for the algorithm to find the MCS. Must be
            an integer, as required by RDKit's `FindMCS`. Defaults to 15.
        mcs_kwargs: keyword arguments for the MCS algorithm.
        kwargs: keyword arguments for clustering algorithm.

    Returns:
        data: updated dataframe with cluster labels.
    """
    if mcs_kwargs is None:
        mcs_kwargs = {}
    if any([isinstance(data, list), isinstance(data, np.ndarray)]):
        smiles_list = data
        mcs_cluster = MCSClustering(
            smiles_list, timeout=timeout, njobs=n_jobs, **mcs_kwargs
        )
        mcs_cluster.compute_similarity_matrix()
        cluster_ids = mcs_cluster.cluster_molecules(algorithm=algorithm, **kwargs)
        data = pd.DataFrame(
            {"smiles": smiles_list, "cluster_id": cluster_ids}, index=None
        )
    elif isinstance(data, pd.DataFrame):
        if smiles_col is None:
            smiles_col = find_smiles_column(data)
        data = _apply_score_cutoff(data, score_col, score_cutoff)
        smiles_list = data[smiles_col].tolist()
        mcs_cluster = MCSClustering(
            smiles_list, timeout=timeout, njobs=n_jobs, **mcs_kwargs
        )
        mcs_cluster.compute_similarity_matrix()
        labels = mcs_cluster.cluster_molecules(algorithm=algorithm, **kwargs)
        data = data.assign(cluster_id=labels)
        data = _pick_best_per_cluster(data, score_col, pick_best)
        logger.info(f"Clustering done using {algorithm}.")
        logger.info(f"Total amount of clusters: {len(data.cluster_id.unique())}")
    else:
        raise ValueError(
            "Data should be a pandas dataframe or a .smi file (list or array), but "
            f"{type(data)} was provided."
        )
    return data
