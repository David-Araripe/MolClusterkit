# -*- coding: utf-8 -*-
"""Module containing the MCS clustering class."""
from itertools import combinations
from typing import List, Tuple

import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from networkx.algorithms import community
from rdkit import Chem
from rdkit.Chem import rdFMCS
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN, SpectralClustering
from tqdm import tqdm

from .logger import logger

MCS_CONFIGS = {
    "AtomCompare": {
        "CompareAny": rdFMCS.AtomCompare.CompareAny,
        "CompareAnyHeavyAtom": rdFMCS.AtomCompare.CompareAnyHeavyAtom,
        "CompareElements": rdFMCS.AtomCompare.CompareElements,
        "CompareIsotopes": rdFMCS.AtomCompare.CompareIsotopes,
    },
    "BondCompare": {
        "CompareAny": rdFMCS.BondCompare.CompareAny,
        "CompareOrder": rdFMCS.BondCompare.CompareOrder,
        "CompareOrderExact": rdFMCS.BondCompare.CompareOrderExact,
    },
    "RingCompare": {
        "IgnoreRingFusion": rdFMCS.RingCompare.IgnoreRingFusion,
        "PermissiveRingFusion": rdFMCS.RingCompare.PermissiveRingFusion,
        "StrictRingFusion": rdFMCS.RingCompare.StrictRingFusion,
    },
}


class MCSClustering:
    """A class for clustering molecules based on Maximum Common Substructure (MCS) similarity.

    Attributes:
    - smiles_list (List[str]): List of input SMILES strings.
    - similarity_matrix (Optional[np.ndarray]): Computed similarity matrix for the input SMILES.

    Usage:
    >>> smiles_list = [...]  # Your list of SMILES
    >>> mcs_cluster = MCSClustering(smiles_list)
    >>> mcs_cluster.compute_similarity_matrix()
    >>> labels = mcs_cluster.cluster_molecules(algorithm='DBSCAN')
    """

    def __init__(self, smiles_list, timeout=15, **mcs_kwargs):
        """Initialize the Maximum Common Substructure (MCS) clustering class with a
        list of SMILES.

        Args:
            smiles_list: a list of smiles.
            timeout: a timeout for the MCS computation in seconds. Defaults to 1.5.
            mcs_kwargs: keyword arguments for the MCS algorithm. Will be parsed based
                on the values from MCS_CONFIGS.

        Usage:
            >>> smiles_list = [...]  # Your list of SMILES
            >>> mcs_cluster = MCSClustering(smiles_list)
            >>> mcs_cluster.compute_similarity_matrix()
            >>> labels = mcs_cluster.cluster_molecules(algorithm='DBSCAN')
        """
        self.smiles_list = smiles_list
        self.timeout = timeout
        self.similarity_matrix = None
        self.mcs_kwargs = {}
        self._setup_mcs_configs(**mcs_kwargs)
        self._check_low_timeout()

    def _check_low_timeout(self):
        """Check if the timeout is too low."""
        if self.timeout < 2:
            logger.warning(
                "Timeout is too low. The MCS algorithm might not find the MCS for some pairs, "
                "raising a not-so-clear error message. Consider increasing the timeout."
            )

    def _setup_mcs_configs(self, **mcs_kwargs):
        """Setup the MCS configurations."""
        for key, value in self.mcs_kwargs.items():
            if key in ["AtomCompare", "BondCompare", "RingCompare"]:
                self.mcs_kwargs[key] = MCS_CONFIGS[key][value]
            else:
                raise ValueError(
                    f"Unsupported MCS configuration: {key}. "
                    f"Supported configurations are: {list(MCS_CONFIGS.keys())}"
                )

    def _mcs_similarity(self, smipair: Tuple[str, str]):
        """Compute the MCS similarity between two molecules given their SMILES and
        return the fraction of matched atoms to the smaller molecule."""
        mols = [Chem.MolFromSmiles(smi) for smi in smipair]
        if any([mols[0] is None, mols[1] is None]):
            logger.error(
                f"Could not parse: {smipair[0]} or {smipair[1]}!!\nRemove invalid SMILES..."
            )
            raise ValueError("Could not parse SMILES into molecules.")
        mcs_result = rdFMCS.FindMCS(list(mols), timeout=self.timeout, **self.mcs_kwargs)
        min_atoms = min(mols[0].GetNumAtoms(), mols[1].GetNumAtoms())
        return mcs_result.smartsString, mcs_result.numAtoms / min_atoms

    def pairwise_mcs_similarity(self, smipair) -> Tuple[List[str], List[float]]:
        """Helper function to compute similarity of molecule pair i and j."""
        smarts_string, similarity = self._mcs_similarity(smipair=smipair)
        return smarts_string, similarity

    def compute_similarity_matrix(
        self, show_progress=True, n_jobs=8
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the similarity matrix based on MCS for all molecules.

        Args:
            n_jobs: number of jobs for parallel processing. Defaults to 8.
            show_progress: whether to show the progress bar. Defaults to True.

        Returns:
            smarts_matrix: np.ndarray with the smarts patterns of the MCS's.
            simi_matrix: np.ndarray with the similarity matrix.
        """
        # ---- First we compute the similarity matrix ----
        # create the similarity matrix with 1s in the diagonal
        n_mols = len(self.smiles_list)
        simi_matrix = np.eye(n_mols)
        # compute the similarity for all pairs of molecules and unpack results
        pairs = list(combinations(self.smiles_list, 2))
        if show_progress:
            pairs = tqdm(pairs, total=len(pairs))
        results = Parallel(n_jobs=n_jobs)(
            delayed(self.pairwise_mcs_similarity)(p) for p in pairs
        )
        smarts_strings, similarities = zip(*results)
        # take the indices of the upper triangle and populate the matrix
        r, c = np.triu_indices(n_mols, 1)  # row, column indices, respectively
        simi_matrix[r, c] = similarities
        # add values for the lower triangle
        simi_matrix += simi_matrix.T - np.eye(n_mols)
        # ---- Now we also create the matrix with the SMARTS ----
        smarts_matrix = np.full((n_mols, n_mols), "", dtype=object)
        smarts_matrix[r, c] = smarts_strings
        smarts_matrix += smarts_matrix.T  # add the upper and the lower triangles
        # Fill the diagonal with repeated strings
        for i, s in enumerate(self.smiles_list):
            smarts_matrix[i, i] = Chem.MolToSmarts(Chem.MolFromSmiles(s))
        # ---- Save the results as object's attributes ----
        self.similarity_matrix = simi_matrix
        self.smarts_matrix = smarts_matrix
        return smarts_matrix, simi_matrix

    def dbscan_clustering(
        self, eps: float = 0.5, min_samples: int = 5, **kwargs
    ) -> list:
        """DBSCAN clustering based on the similarity matrix.


        Args:
            eps: The maximum distance between two samples for one to be considered as in
                the neighborhood of the other. Defaults to 0.5.
            min_samples: The number of samples (or total weight) in a neighborhood for a
                point to be considered as a core point. Defaults to 5.

        Returns:
            labels: list of cluster labels."""
        distance_matrix = 1 - self.similarity_matrix
        clustering = DBSCAN(
            eps=eps, min_samples=min_samples, metric="precomputed", **kwargs
        ).fit(distance_matrix)
        return clustering.labels_.tolist()

    def hierarchical_clustering(self, t, method="ward", criterion="maxclust", **kwargs):
        """Hierarchical clustering based on the similarity matrix.

        Args:
            t: number of clusters or the threshold to cut the hierarchy.
            method: linkage algorithm to use. Options include 'single', 'complete',
                'average', 'ward'. Defaults to "ward".
            criterion: criterion to form flat clusters. Common choices are 'maxclust'
                and 'distance'. Defaults to "maxclust".

        Returns:
            labels: list of cluster labels.
        """
        distance_matrix = 1 - self.similarity_matrix
        Z = linkage(distance_matrix, method=method)
        labels = fcluster(Z, t, criterion=criterion, **kwargs)
        return labels - 1  # Adjusting the labels to be 0-based

    def graph_based_clustering(self, threshold: float = 0.7, **kwargs) -> list:
        """Graph-based clustering based on the similarity matrix using community detection.

        Args:
            threshold: similarity threshold. Edges with similarity below this are not
                added to the graph. Defaults to 0.7.

        Returns:
            labels: list of cluster labels.
        """
        G = nx.Graph()
        iter_arr = list(combinations(range(len(self.smiles_list)), 2))

        for i, j in iter_arr:
            # Adding an edge if similarity is above the threshold
            if self.similarity_matrix[i, j] > threshold:
                G.add_edge(i, j, weight=self.similarity_matrix[i, j])

        # Using community detection to cluster
        detected_communities = community.greedy_modularity_communities(G, **kwargs)
        # Converting communities to labels
        labels = [-1] * len(self.smiles_list)
        for cluster_id, comm in enumerate(detected_communities):
            for node in comm:
                labels[node] = cluster_id
        return labels

    def spectral_clustering(self, n_clusters: int, **kwargs) -> list:
        """Spectral clustering based on the similarity matrix.

        Args:
            n_clusters: number of clusters to form.

        Returns:
            labels: list of cluster labels.
        """
        if "random_state" not in kwargs:
            logger.warning(
                "No random_state provided for SpectralClustering! "
                "Should be passed in kwargs."
            )
        clustering = SpectralClustering(
            n_clusters=n_clusters, affinity="precomputed", **kwargs
        ).fit(self.similarity_matrix)
        return clustering.labels_

    def cluster_molecules(self, algorithm="DBSCAN", **kwargs):
        """Clusters molecules based on the computed similarity matrix.

        Args:
            algorithm: algorithm to use for clustering. Options include 'DBSCAN',
                'Hierarchical', 'Spectral', 'GraphBased'. Defaults to "DBSCAN".

        Raises:
            ValueError: if the similarity matrix has not been computed.
            ValueError: if the chosen algorithm is not supported.

        Returns:
            labels: list of cluster labels.
        """
        if self.similarity_matrix is None:
            raise ValueError(
                "Similarity matrix has not been computed. Run 'compute_similarity_matrix' first."
            )
        clustering_algorithms = {
            "DBSCAN": self.dbscan_clustering,
            "Hierarchical": self.hierarchical_clustering,
            "Spectral": self.spectral_clustering,
            "GraphBased": self.graph_based_clustering,
        }
        if algorithm not in clustering_algorithms:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        return clustering_algorithms[algorithm](**kwargs)

    def __call__(self, algorithm="DBSCAN", n_jobs=8, **kwargs):
        """Convenience function to compute similarity matrix and perform clustering."""
        self.compute_similarity_matrix(n_jobs=n_jobs)
        return self.cluster_molecules(algorithm=algorithm, **kwargs)
