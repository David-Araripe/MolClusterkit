from itertools import combinations
from typing import Optional

import networkx as nx
import numpy as np
from networkx.algorithms import community
from numpy.typing import DTypeLike
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN, SpectralClustering

from .logger import logger


class BaseClusterer:
    """
    A base class for the different clustering methods implemented in the package. It
    contains common methods and attributes that are shared among the clustering classes.

    Attributes:
        smiles_list: List of input SMILES strings.
        njobs: Number of jobs for parallel processing.
        np_dtypes: numpy data type for the similarity matrix or arrays.
    """

    def __init__(
        self,
        smiles_list: Optional[list[str]],
        njobs: int,
        np_dtypes: DTypeLike = np.float32,
    ) -> None:
        self.smiles_list = smiles_list
        self.njobs = njobs
        self.np_dtypes = np_dtypes
        self.mol_clusters = None
        self.similarity_matrix = None

    def compute_similarity_matrix(self):
        pass

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
        self.mol_clusters = clustering.labels_.tolist()
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
        self.mol_clusters = labels
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
        self.mol_clusters = labels
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
        self.mol_clusters = clustering.labels_
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
