"""Module containing the Rascal MCES clustering class. The class supports the calculation
of a similarity matrix based on the MCES result, or clustering directly through fuzzy or
Butina-based clustering methods.
"""

from collections import defaultdict
from itertools import combinations
from typing import Literal, Optional, Tuple

import numpy as np
from numpy.typing import DTypeLike
from rdkit import Chem
from rdkit.Chem import rdRascalMCES

from MolClusterkit.base_clusterer import BaseClusterer

from .logger import logger
from .parallel import ParallelApplier


class RascalMCES(BaseClusterer):
    """A class for clustering molecules based on Rascal Maximum Common Edge Subgraph (MCES).
    For the technical details, see: https://eprints.whiterose.ac.uk/3568/1/willets3.pdf

    Attributes:
        smiles_list: list of input SMILES strings representing the molecules to cluster.
        similarity_matrix: computed similarity matrix for the input SMILES.
        smarts_matrix: matrix containing SMARTS patterns of the MCS.

    Usage:
        >>> smiles_list = [...]  # Your list of SMILES
        >>> rascal = RascalMCES(smiles_list)
        >>> rascal.compute_similarity_matrix()
        >>> clusters = rascal.cluster_butina()
    """

    def __init__(
        self,
        smiles_list: Optional[list[str]] = None,
        np_dtypes: DTypeLike = np.float32,
        similarityThreshold: float = 0.7,
        allBestMCESs: bool = True,
        singleLargestFrag: bool = True,
        njobs: int = 8,
    ):
        """Initialize the Rascal MCES clustering class with a list of SMILES and a defined
        numpy data type for the similarity matrix. Further, three options are available to
        control the MCES algorithm:
        - allBestMCESs: If True, reports all MCESs found of the same maximum size


        Args:
            smiles_list: List of SMILES strings to analyze
            np_dtypes: numpy data type for the similarity matrix
            similarityThreshold: Threshold below which MCES won't be run. Between 0.0
                and 1.0, default=0.7.
            allBestMCESs: If True, reports all MCESs found of the same maximum size
                Default False means just report the first found.
            singleLargestFrag: Return the just single largest fragment of the MCES. This
                is equivalent to running with allBestMCEs=True, finding the result with the
                largest largestFragmentSize, and calling its largestFragmentOnly method.
        """
        super().__init__(smiles_list=smiles_list, njobs=njobs, np_dtypes=np_dtypes)
        # Set up Rascal options
        self.opts_dict = {
            "similarityThreshold": similarityThreshold,
            "allBestMCESs": allBestMCESs,
            "singleLargestFrag": singleLargestFrag,
            "returnEmptyMCES": False,
        }

    def _mol_from_smiles(self, smiles: str) -> Chem.Mol:
        """Converts a SMILES string to an RDKit molecule object.

        Args:
            smiles: SMILES to be converted to RDKit Mol object.

        Returns:
            RDKit Mol object.
        """
        return Chem.MolFromSmiles(smiles)

    def _mols_from_smiles(self, smiles: list[str]) -> list[Chem.Mol]:
        """Converts a list of SMILES strings to RDKit molecule objects using joblib
        parallel processing.

        Args:
            smiles: list of SMILES to be converted to RDKit Mol objects.

        Returns:
            list of RDKit Mol objects.
        """
        applier = ParallelApplier(
            func=self._mol_from_smiles, iterable=smiles, n_jobs=self.njobs
        )
        return applier()

    def _make_opts(
        self,
        allBestMCESs: Optional[bool] = None,
        completeAromaticRings: Optional[bool] = None,
        equivalentAtoms: Optional[list[str]] = None,
        exactConnectionsMatch: Optional[bool] = None,
        ignoreBondOrders: Optional[bool] = None,
        maxBondMatchPairs: Optional[int] = None,
        maxFragSeparation: Optional[int] = None,
        minFragSize: Optional[int] = None,
        returnEmptyMCES: Optional[bool] = None,
        ringMatchesRingOnly: Optional[bool] = None,
        similarityThreshold: Optional[float] = None,
        singleLargestFrag: Optional[bool] = None,
        timeout: Optional[int] = None,
    ):
        """Create an RascalOptions object with the given settings. Passing parameters other
        than the default `None` will override the settings set on object initialization.

        Args:
            allBestMCESs: If True, reports all MCESs found of the same maximum size
                Default False means just report the first found.
            completeAromaticRings: If True (default), partial aromatic rings won't be returned.
            equivalentAtoms: SMARTS strings defining atoms that shouldbe considered equivalent.
                e.g.[F,Cl,Br,I] so all halogens will match each other. Space-separated list
                allowing more than 1class of equivalent atoms.
            exactConnectionsMatch: If True (default is False), atoms will only match atoms
                if they have the same number of explicit connections. E.g. the central atom of
                C(C)(C) won't match either atom in CC
            ignoreBondOrders: If True, will treat all bonds as the same, irrespective of order. Default=False.
            maxBondMatchPairs: Too many matching bond (vertex) pairs can cause the process
                to run out of memory. The default of 1000 is fairly safe. Increase with caution,
                as memory use increases with the square of this number.
            maxFragSeparation: Maximum number of bonds between fragments in the MCES for both
                to be reported. Default -1 means no maximum. If exceeded, the smaller fragment
                will be removed.
            minFragSize: Imposes a minimum on the number of atoms in a fragment that may be part
                of the MCES. Default -1 means no minimum.
            returnEmptyMCES: If the estimated similarity between the 2 molecules doesn't meet the
                similarityThreshold, no results are returned. If you want to know what the estimates
                were, set this to True, and examine the tier1Sim and tier2Sim properties of the result
                then returned.
            ringMatchesRingOnly: If True (default), ring bonds won't match non ring bonds.
            similarityThreshold: Threshold below which MCES won't be run. Between 0.0 and 1.0, default=0.7.
            singleLargestFrag: Return the just single largest fragment of the MCES. This is equivalent
                to running with allBestMCEs=True, finding the result with the largest largestFragmentSize,
                and calling its largestFragmentOnly method.
            timeout: Maximum time (in seconds) to spend on an individual MCESs determination.
                Default 60, -1 means no limit.

        Returns:
            _description_
        """
        opts = rdRascalMCES.RascalOptions()

        if allBestMCESs is not None:
            opts.allBestMCESs = allBestMCESs
        else:
            opts.allBestMCESs = self.opts_dict["allBestMCESs"]

        if completeAromaticRings is not None:
            opts.completeAromaticRings = completeAromaticRings
        if equivalentAtoms is not None:
            opts.equivalentAtoms = equivalentAtoms
        if exactConnectionsMatch is not None:
            opts.exactConnectionsMatch = exactConnectionsMatch
        if ignoreBondOrders is not None:
            opts.ignoreBondOrders = ignoreBondOrders
        if maxBondMatchPairs is not None:
            opts.maxBondMatchPairs = maxBondMatchPairs
        if maxFragSeparation is not None:
            opts.maxFragSeparation = maxFragSeparation
        if minFragSize is not None:
            opts.minFragSize = minFragSize

        if returnEmptyMCES is not None:
            opts.returnEmptyMCES = returnEmptyMCES
        else:
            opts.returnEmptyMCES = self.opts_dict["returnEmptyMCES"]

        if ringMatchesRingOnly is not None:
            opts.ringMatchesRingOnly = ringMatchesRingOnly
        if similarityThreshold is not None:
            opts.similarityThreshold = similarityThreshold
        else:
            opts.similarityThreshold = self.opts_dict["similarityThreshold"]

        if singleLargestFrag is not None:
            opts.singleLargestFrag = singleLargestFrag
        else:
            opts.singleLargestFrag = self.opts_dict["singleLargestFrag"]

        if timeout is not None:
            opts.timeout = timeout
        return opts

    def mces_similarity(
        self,
        smipair: Tuple[str, str],
        similarity_metric: Literal["johnson", "smaller/mces"] = "johnson",
        **kwargs,
    ) -> Tuple[str, float]:
        """Compute MCES between two molecules given their SMILES, returning the SMARTS
        pattern and a score. Two metrics are available to calculate the similarity between
        the two molecules, with values ranging from 0 to 1:

        1. Johnson metric:
        The similarity is calculated as the sum of the number of atoms
        and bonds in the MCES divided by the sum of the number of atoms and bonds in
        the two molecules.

        .. math::
        sim = \\frac{(E(MCES) + V(MCES))^2}{(E(Mol1) + V(Mol1)) * (E(Mol2) + V(Mol2))}

        2. Smaller/MCES metric:
        The similarity is calculated as number of atoms in the
        largest fragment of the MCES divided by the number of atoms in the smaller
        molecule.

        .. math::
        sim = \\frac{E(LargestFragment(MCES))}{min(E(Mol1), E(Mol2))}

        Args:
            smipair: Tuple of two SMILES strings
            similarity_metric: The similarity metric to use. Options are 'johnson' or
                'smaller/mces'. Default is 'johnson'.
            kwargs: keyword arguments for the MCES algorithm, used to create the settings
                object through `self._make_opts`.

        Returns:
            Tuple containing the SMARTS pattern and a similarity score between 0 and 1.
        """
        mols = [Chem.MolFromSmiles(smi) for smi in smipair]
        min_atoms = min(mols[0].GetNumAtoms(), mols[1].GetNumAtoms())
        if any(mol is None for mol in mols):
            logger.error(
                f"Could not parse: {smipair[0]} or {smipair[1]}!!\n"
                "Remove invalid SMILES..."
            )
            raise ValueError("Could not parse SMILES into molecules.")

        params = {**self.opts_dict, **kwargs}  # kwargs will override defaults
        opts = self._make_opts(**params)
        results = rdRascalMCES.FindMCES(mols[0], mols[1], opts)
        if not results:
            return "", 0.0

        if similarity_metric == "johnson":  # Johnson metric returns -1 if no MCES found
            simi_metric = max([r.tier2Sim if r.tier2Sim != -1 else 0 for r in results])
            # RASCAL is fast because it calculates two similarity metrics, the first
            # being cheaper to calculate. The second is only calculated if the first
            # is above the set threshold. Here we only return the second, under tier2Sim
        elif similarity_metric == "smaller/mces":
            simi_metric = results[0].largestFragmentSize / min_atoms
        return results[0].smartsString, simi_metric

    def compute_similarity_matrix(
        self,
        smiles_list: Optional[list[str]] = None,
        show_progress: bool = True,
        n_jobs: int = 8,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the similarity matrix based on MCES for all molecules.

        Args:
            smiles_list: Optional list of SMILES to override instance list
            show_progress: Whether to show progress bar
            n_jobs: Number of parallel jobs to run

        Returns:
            Tuple of the SMARTS matrix and the similarity matrix
        """
        if smiles_list is not None:
            self.smiles_list = smiles_list

        n_mols = len(self.smiles_list)
        simi_matrix = np.eye(n_mols, dtype=self.np_dtypes)

        # Compute similarities for all molecule pairs
        pairs = list(combinations(self.smiles_list, 2))

        applier = ParallelApplier(
            func=self.mces_similarity,
            iterable=pairs,
            n_jobs=n_jobs,
            show_progress=show_progress,
        )
        results = applier()

        # Unpack results
        smarts_strings, similarities = zip(*results)

        # Populate the matrices
        r, c = np.triu_indices(n_mols, 1)
        simi_matrix[r, c] = similarities
        simi_matrix += simi_matrix.T - np.eye(n_mols, dtype=self.np_dtypes)

        # Create SMARTS matrix
        smarts_matrix = np.full((n_mols, n_mols), "", dtype=object)
        smarts_matrix[r, c] = smarts_strings
        smarts_matrix += smarts_matrix.T

        # Fill diagonal with molecule SMARTS
        for i, s in enumerate(self.smiles_list):
            mol = Chem.MolFromSmiles(s)
            smarts_matrix[i, i] = Chem.MolToSmarts(mol)

        self.similarity_matrix = simi_matrix
        self.smarts_matrix = smarts_matrix
        return smarts_matrix, simi_matrix

    def fuzzy_mces_clustering(
        self,
        smiles_list: Optional[list[str]] = None,
        cutoff: float = 0.7,
        clusterMergeSim: float = 0.6,
        maxNumFrags: int = 2,
        minFragSize: int = 3,
        minIntraClusterSim: int = 0.9,
    ) -> list[list[int]]:
        """Perform fuzzy clustering based on RASCAL MCES similarity, where a single molecule
        can belong to multiple clusters. To quickly check for cluster membership, use the
        method `get_cluster_membership` to return a membershipo dictionary.

        Args:
            cutoff: Similarity cutoff for clustering
            smiles_list: Optional list of SMILES to override instance list
            clusterMergeSim: Two clusters are merged if the fraction of molecules they
                have in common is greater than this. Default=0.6.
            maxNumFrags: The maximum number of fragments allowed in the MCES for each
                pair of molecules. Default=2. So that the MCES isn't a lot of small
                fragments scattered around the molecules giving an inflated estimate
                of similarity.
            minFragSize: The minimum number of atoms in a fragment for it to be included
                in the MCES. Default=3.
            minIntraClusterSim: Two pairs of molecules are included in the same cluster
                if the similarity between their MCESs is greater than this. Default=0.9.

        Returns:
            List of clusters, where each cluster is a list of molecule indices
        """
        if smiles_list is not None:
            self.smiles_list = smiles_list

        if self.similarity_matrix is None:
            logger.warning("Similarity matrix not computed. Computing now...")
            self.compute_similarity_matrix()

        mols = self._mols_from_smiles(self.smiles_list)
        cluster_opts = rdRascalMCES.RascalClusterOptions()
        cluster_opts.similarityCutoff = cutoff
        cluster_opts.numThreads = self.njobs
        cluster_opts.clusterMergeSim = clusterMergeSim
        cluster_opts.maxNumFrags = maxNumFrags
        cluster_opts.minFragSize = minFragSize
        cluster_opts.minIntraClusterSim = minIntraClusterSim
        return rdRascalMCES.RascalCluster(mols, cluster_opts)

    def butina_mces_clustering(
        self,
        smiles_list: Optional[list[str]] = None,
        cutoff: float = 0.7,
        clusterMergeSim: float = 0.6,
        maxNumFrags: int = 2,
        minFragSize: int = 3,
        minIntraClusterSim: int = 0.9,
    ) -> list[list[int]]:
        """Performs clustering based on the Johnson similarity between molecules and
        their MCES, but using the Butina method (non-fuzzy) to assign molecules to clusters.

        Args:
            smiles_list: Optional list of SMILES to override instance list
            clusterMergeSim: Two clusters are merged if the fraction of molecules they
                have in common is greater than this. Default=0.6.
            maxNumFrags: The maximum number of fragments allowed in the MCES for each
                pair of molecules. Default=2. So that the MCES isn't a lot of small
                fragments scattered around the molecules giving an inflated estimate
                of similarity.
            minFragSize: The minimum number of atoms in a fragment for it to be included
                in the MCES. Default=3.
            minIntraClusterSim: Two pairs of molecules are included in the same cluster
                if the similarity between their MCESs is greater than this. Default=0.9.

        Returns:
            List of clusters, where each cluster is a list of molecule indices
        """
        if smiles_list is not None:
            self.smiles_list = smiles_list

        cluster_opts = rdRascalMCES.RascalClusterOptions()
        cluster_opts.similarityCutoff = cutoff
        cluster_opts.numThreads = self.njobs
        cluster_opts.clusterMergeSim = clusterMergeSim
        cluster_opts.maxNumFrags = maxNumFrags
        cluster_opts.minFragSize = minFragSize
        cluster_opts.minIntraClusterSim = minIntraClusterSim

        mols = self._mols_from_smiles(self.smiles_list)
        return rdRascalMCES.RascalCluster(mols, cluster_opts)

    def get_cluster_membership(self, clusters: list[list[int]]) -> defaultdict:
        """Get membership dictionary showing which clusters each molecule belongs to.

        Args:
            clusters: List of clusters from clustering method

        Returns:
            Dictionary mapping molecule indices to list of cluster indices
        """
        membership = defaultdict(list)
        for ci, cluster in enumerate(clusters):
            for mol_idx in cluster:
                membership[mol_idx].append(ci)
        return membership
