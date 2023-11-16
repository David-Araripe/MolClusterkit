from typing import Optional

from loguru import logger

from .butina import ButinaClustering
from .mcs import MCSClustering


def butina_based_clustering(
    df,
    smiles_col,
    score_col: Optional[str] = None,
    score_cutoff=7.0,
    cutoff=0.25,
    save_path=None,
    njobs=12,
    pick_best=False,
):
    """Perform butina clustering on the given dataframe and return the best scoring

    Args:
        df: dataframe to be used for clustering.
        smiles_col: column name containing the smiles structures of the compounds.
        score_col: column name containing the scores. Optional. Defaults to None.
        score_cutoff: threshold to be used on score_col. Defaults to 7.0.
        cutoff: cutoff for the butina clustering. Defaults to 0.25.
        save_path: path to save the resulting data frame to. Defaults to None.
        njobs: number of jobs for parallelization. Defaults to 12.

    Returns:
        pd.DataFrame: dataframe with the best scoring compounds from each cluster.
    """
    if score_cutoff is not None:
        df = df.query(f"{score_col} > {score_cutoff}")
    smiles_list = df[smiles_col].tolist()
    bclusterer = ButinaClustering(smiles_list, njobs=njobs)
    bclusterer.cluster(cutoff=cutoff)
    df = bclusterer.assign_clusters_to_dataframe(df, score_col=score_col)
    if all([pick_best, score_col is not None]):
        df = df.groupby("cluster_id").apply(lambda x: x.loc[x[score_col].idxmax()])
    logger.info(f"Total amount of clusters: {len(df.cluster_id.unique())}")
    if save_path is not None:
        df.to_csv(save_path, index=False)
    return df


def mcs_based_clustering(
    df,
    smiles_col,
    score_col: Optional[str] = None,
    score_cutoff=None,
    algorithm="DBSCAN",
    pick_best=False,
    n_jobs=12,
    timeout=1.5,
    **kwargs,
):
    """Perform MCS based clustering on the given dataframe.

    Args:
        df: dataframe to apply MCS clustering to.
        smiles_col: column name containing the smiles structures of the compounds.
        algorithm: algorithm to use for clustering.
        kwargs: keyword arguments for clustering algorithm.
        n_jobs: number of jobs for parallelization.

    Returns:
        df: updated dataframe with cluster labels.
    """
    if score_cutoff is not None:
        df = df.query(f"{score_col} > {score_cutoff}")
    smiles_list = df[smiles_col].tolist()
    mcs_cluster = MCSClustering(smiles_list, timeout=timeout)
    mcs_cluster.compute_similarity_matrix(n_jobs=n_jobs)
    labels = mcs_cluster.cluster_molecules(algorithm=algorithm.lower(), **kwargs)
    df = df.assign(cluster_id=labels)
    if all([pick_best, score_col is not None]):
        df = df.groupby("cluster_id").apply(lambda x: x.loc[x[score_col].idxmax()])
    logger.info(f"Clustering done using {algorithm}.")
    logger.info(f"Total amount of clusters: {len(df.cluster_id.unique())}")
    return df
