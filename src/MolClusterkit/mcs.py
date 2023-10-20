# -*- coding: utf-8 -*-
"""Module containing the MCS clustering class."""
from itertools import combinations

import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from networkx.algorithms import community
from rdkit import Chem
from rdkit.Chem import rdFMCS
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN, SpectralClustering


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

    def __init__(self, smiles_list):
        """Initialize the Maximum Common Substructure (MCS) clustering class with a
        list of SMILES.

        Args:
            smiles_list: a list of smiles.
        """
        self.smiles_list = smiles_list
        self.similarity_matrix = None

    def _mcs_similarity(self, smi1, smi2):
        """Compute the MCS similarity between two molecules given their SMILES and
        return the fraction of matched atoms to the smaller molecule."""
        mol1 = Chem.MolFromSmiles(smi1)
        mol2 = Chem.MolFromSmiles(smi2)

        mcs_result = rdFMCS.FindMCS([mol1, mol2])
        min_atoms = min(mol1.GetNumAtoms(), mol2.GetNumAtoms())
        return mcs_result.numAtoms / min_atoms

    def _pairwise_mcs_similarity(self, i, j):
        """Helper function to compute similarity of molecule pair i and j."""
        return i, j, self._mcs_similarity(self.smiles_list[i], self.smiles_list[j])

    def compute_similarity_matrix(self, n_jobs=8) -> np.ndarray:
        """Compute the similarity matrix based on MCS for all molecules.

        Args:
            n_jobs: number of jobs for parallel processing. Defaults to 8.
        """
        n_mols = len(self.smiles_list)
        matrix = np.zeros((n_mols, n_mols))
        pairs = [(i, j) for i in range(n_mols) for j in range(i + 1, n_mols)]
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._pairwise_mcs_similarity)(i, j) for i, j in pairs
        )
        # unpack the results and populate the matrix
        rows, cols, sims = zip(*results)
        matrix[rows, cols] = sims
        matrix[cols, rows] = sims
        self.similarity_matrix = matrix
        return matrix

    def dbscan_clustering(self, eps: float = 0.5, min_samples: int = 5) -> list:
        """DBSCAN clustering based on the similarity matrix.


        Args:
            eps: The maximum distance between two samples for one to be considered as in
                the neighborhood of the other. Defaults to 0.5.
            min_samples: The number of samples (or total weight) in a neighborhood for a
                point to be considered as a core point. Defaults to 5.

        Returns:
            labels: list of cluster labels."""
        distance_matrix = 1 - self.similarity_matrix
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit(
            distance_matrix
        )
        return clustering.labels_.tolist()

    def hierarchical_clustering(self, t, method="ward", criterion="maxclust"):
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
        labels = fcluster(Z, t, criterion=criterion)
        return labels - 1  # Adjusting the labels to be 0-based

    def graph_based_clustering(self, threshold: float = 0.7) -> list:
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
        detected_communities = community.greedy_modularity_communities(G)
        # Converting communities to labels
        labels = [-1] * len(self.smiles_list)
        for cluster_id, comm in enumerate(detected_communities):
            for node in comm:
                labels[node] = cluster_id
        return labels

    def spectral_clustering(self, n_clusters: int) -> list:
        """Spectral clustering based on the similarity matrix.

        Args:
            n_clusters: number of clusters to form.

        Returns:
            labels: list of cluster labels.
        """
        clustering = SpectralClustering(
            n_clusters=n_clusters, affinity="precomputed", random_state=42
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
