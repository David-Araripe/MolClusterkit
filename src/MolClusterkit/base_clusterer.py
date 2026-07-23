# -*- coding: utf-8 -*-
"""Base clusterer class for the different clustering methods implemented in the package."""
from itertools import combinations
from typing import Optional

import networkx as nx
import numpy as np
from networkx.algorithms import community
from numpy.typing import DTypeLike
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score

from .logger import logger


# Inheritance structure in the package:
# BaseClusterer -> RascalMCES
# BaseClusterer -> MCSClustering
# BaseClusterer -> ButinaClustering
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

    def _resolve_smiles_list(self, smiles_list: Optional[list[str]] = None) -> list[str]:
        """Validate and store the SMILES list backing a similarity computation.

        Args:
            smiles_list: optional sequence of SMILES overriding the instance list.
                If given, it replaces `self.smiles_list`.

        Raises:
            TypeError: if `smiles_list` is a string/bytes or has no length.
            ValueError: if no SMILES list is available on the instance or the call.

        Returns:
            smiles_list: the resolved sequence of SMILES.
        """
        if smiles_list is not None:
            if isinstance(smiles_list, (str, bytes)) or not hasattr(
                smiles_list, "__len__"
            ):
                raise TypeError(
                    "smiles_list must be a sequence of SMILES strings, got "
                    f"{type(smiles_list).__name__}. Note that show_progress is no "
                    "longer the first positional argument of compute_similarity_matrix."
                )
            self.smiles_list = smiles_list
        if self.smiles_list is None:
            raise ValueError(
                "No SMILES list provided. Pass smiles_list to the constructor "
                "or to compute_similarity_matrix()."
            )
        return self.smiles_list

    def _distance_matrix(self) -> np.ndarray:
        """Square distance matrix derived from the similarity matrix.

        Both `scipy.spatial.distance.squareform` and `silhouette_score` with
        metric="precomputed" require an exactly symmetric matrix with a zero
        diagonal. Rounding noise in the similarity matrix can break either
        property, so they are enforced here rather than by disabling the checks.

        Returns:
            distance_matrix: symmetric, hollow square matrix of distances.
        """
        distance_matrix = 1 - np.asarray(self.similarity_matrix)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0.0)
        return distance_matrix

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
        distance_matrix = self._distance_matrix()
        clustering = DBSCAN(
            eps=eps, min_samples=min_samples, metric="precomputed", **kwargs
        ).fit(distance_matrix)
        self.mol_clusters = clustering.labels_.tolist()
        return clustering.labels_.tolist()

    def hierarchical_clustering(self, t, method="ward", criterion="maxclust", **kwargs):
        """Hierarchical clustering based on the similarity matrix.

        Args:
            t: number of clusters or the threshold to cut the hierarchy.
            method: linkage algorithm to use. Options include "single", "complete",
                "average", "ward". Defaults to "ward".
            criterion: criterion to form flat clusters. Common choices are "maxclust"
                and "distance". Defaults to "maxclust".

        Returns:
            labels: list of cluster labels.
        """
        distance_matrix = self._distance_matrix()
        # linkage reads a 2-D input as observations-by-features, so the pairwise
        # distances have to be condensed first.
        Z = linkage(squareform(distance_matrix), method=method)
        labels = fcluster(Z, t, criterion=criterion, **kwargs) - 1  # 0-based labels
        self.mol_clusters = labels
        return labels

    def hierarchical_silhouette_clustering(
        self, max_clusters=20, method="ward", criterion="maxclust", **kwargs
    ):
        """Hierarchical clustering based on the similarity matrix using silhouette score.
        This method will compute the silhouette score for 2 to `max_clusters` clusters and
        return the labels with the cluster number that maximizes the silhouette score.

        Args:
            max_clusters: maximum number of clusters to consider. Capped at the
                number of samples minus one, the largest value for which a
                silhouette score is defined.
            method: linkage algorithm to use. Options include "single", "complete",
                "average", "ward". Defaults to "ward".
            criterion: criterion to form flat clusters. Common choices are "maxclust"
                and "distance". Defaults to "maxclust".

        Raises:
            ValueError: if there are fewer than three samples, in which case no
                silhouette score can be computed.

        Returns:
            labels: list of cluster labels.
        """
        distance_matrix = self._distance_matrix()
        n_samples = distance_matrix.shape[0]
        if n_samples < 3:
            raise ValueError(
                "At least 3 molecules are needed to compare silhouette scores, "
                f"got {n_samples}."
            )
        # linkage reads a 2-D input as observations-by-features, so the pairwise
        # distances have to be condensed first.
        Z = linkage(squareform(distance_matrix), method=method)
        scores = []
        all_labels = []
        for t in range(2, min(max_clusters, n_samples - 1) + 1):
            labels = fcluster(Z, t, criterion=criterion, **kwargs) - 1  # 0-based
            scores.append(
                silhouette_score(distance_matrix, labels, metric="precomputed")
            )
            all_labels.append(labels)
        best_labels = all_labels[np.argmax(scores)]
        logger.info(f"Best number of clusters: {len(np.unique(best_labels))}")
        logger.info(f"Silhouette scores: {scores}")
        self.mol_clusters = best_labels
        return best_labels

    def graph_based_clustering(self, threshold: float = 0.7, **kwargs) -> list:
        """Graph-based clustering based on the similarity matrix using community detection.

        Args:
            threshold: similarity threshold. Edges with similarity below this are not
                added to the graph. Defaults to 0.7.

        Returns:
            labels: list of cluster labels.
        """
        self._resolve_smiles_list()
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
                'Hierarchical', 'HierarchicalSilhouette', 'Spectral', 'GraphBased'.
                Defaults to "DBSCAN".

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
            "HierarchicalSilhouette": self.hierarchical_silhouette_clustering,
            "Spectral": self.spectral_clustering,
            "GraphBased": self.graph_based_clustering,
        }
        if algorithm not in clustering_algorithms:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        return clustering_algorithms[algorithm](**kwargs)
