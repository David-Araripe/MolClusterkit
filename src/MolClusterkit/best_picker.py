# -*- coding: utf-8 -*-
"""Module containing the functions that are used for by the command line interface."""
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from loguru import logger

from .butina import ButinaClustering
from .mcs import MCSClustering
from .misc import find_smiles_column


def butina_based_clustering(
    data: Union[pd.DataFrame, List[str], np.ndarray],
    smiles_col: Optional[str],
    score_col: Optional[str] = None,
    score_cutoff=7.0,
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
        score_cutoff: threshold to be used on score_col; keep only above cutoff. Defaults to 7.0.
        cutoff: cutoff for the butina clustering. Defaults to 0.25.
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
        if score_cutoff is not None:
            data = data.query(f"{score_col} > {score_cutoff}")
            if data.shape[0] == 0:
                raise ValueError(
                    f"No compounds with a score above {score_cutoff} were found."
                )
        smiles_list = data[smiles_col].tolist()
        bclusterer = ButinaClustering(smiles_list, njobs=njobs)
        cluster_ids = bclusterer.cluster_molecules(dist_th=dist_th)
        data = data.assign(cluster_id=cluster_ids)
        if all([pick_best, score_col is not None]):
            data = data.groupby("cluster_id").apply(
                lambda x: x.loc[x[score_col].idxmax()]
            )
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
    timeout=1.5,
    mcs_kwargs=None,
    **kwargs,
) -> pd.DataFrame:
    """Perform MCS based clustering on the given dataframe.

    Args:
        data: dataframe to apply MCS clustering to.
        smiles_col: column name containing the smiles structures of the compounds.
            If is `None`, will try to find the smiles column based on a simple regex search.
        score_col: column name containing the scores. Optional. Defaults to None.
        score_cutoff: threshold to be used on score_col; keep only above cutoff. Defaults to None.
        algorithm: algorithm to use for clustering.
        pick_best: whether to pick the best scoring compound from each cluster. Defaults to False.
        n_jobs: number of jobs for parallelization. Defaults to 8.
        timeout: wall-time in seconds threshold for the algorithm to find the MCS. Defaults to 1.5.
        mcs_kwargs: keyword arguments for the MCS algorithm.
        kwargs: keyword arguments for clustering algorithm.

    Returns:
        data: updated dataframe with cluster labels.
    """
    if mcs_kwargs is None:
        mcs_kwargs = {}
    if any([isinstance(data, list), isinstance(data, np.ndarray)]):
        smiles_list = data
        mcs_cluster = MCSClustering(smiles_list, timeout=timeout, **mcs_kwargs)
        mcs_cluster.compute_similarity_matrix(n_jobs=n_jobs)
        cluster_ids = mcs_cluster.cluster_molecules(algorithm=algorithm, **kwargs)
        data = pd.DataFrame(
            {"smiles": smiles_list, "cluster_id": cluster_ids}, index=None
        )
    elif isinstance(data, pd.DataFrame):
        if smiles_col is None:
            smiles_col = find_smiles_column(data)
        if score_cutoff is not None:
            data = data.query(f"{score_col} > {score_cutoff}")
            if data.shape[0] == 0:
                raise ValueError(
                    f"No compounds with a score above {score_cutoff} were found."
                )
        smiles_list = data[smiles_col].tolist()
        mcs_cluster = MCSClustering(smiles_list, timeout=timeout, **mcs_kwargs)
        mcs_cluster.compute_similarity_matrix(n_jobs=n_jobs)
        labels = mcs_cluster.cluster_molecules(algorithm=algorithm.lower(), **kwargs)
        data = data.assign(cluster_id=labels)
        if all([pick_best, score_col is not None]):
            data = data.groupby("cluster_id").apply(
                lambda x: x.loc[x[score_col].idxmax()]
            )
        logger.info(f"Clustering done using {algorithm}.")
        logger.info(f"Total amount of clusters: {len(data.cluster_id.unique())}")
    else:
        raise ValueError(
            "Data should be a pandas dataframe or a .smi file (list or array), but "
            f"{type(data)} was provided."
        )
    return data
