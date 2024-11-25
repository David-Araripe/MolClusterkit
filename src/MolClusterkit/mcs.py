# -*- coding: utf-8 -*-
"""Module containing the MCS clustering class."""
from itertools import combinations
from typing import Optional, Tuple

import numpy as np
from numpy.typing import DTypeLike
from rdkit import Chem
from rdkit.Chem import rdFMCS

from .base_clusterer import BaseClusterer
from .logger import logger
from .parallel import ParallelApplier

MCS_COMPARE_CONFIGS = {
    "atomCompare": {
        "CompareAny": rdFMCS.AtomCompare.CompareAny,
        "CompareAnyHeavyAtom": rdFMCS.AtomCompare.CompareAnyHeavyAtom,
        "CompareElements": rdFMCS.AtomCompare.CompareElements,
        "CompareIsotopes": rdFMCS.AtomCompare.CompareIsotopes,
    },
    "bondCompare": {
        "CompareAny": rdFMCS.BondCompare.CompareAny,
        "CompareOrder": rdFMCS.BondCompare.CompareOrder,
        "CompareOrderExact": rdFMCS.BondCompare.CompareOrderExact,
    },
    "ringCompare": {
        "IgnoreRingFusion": rdFMCS.RingCompare.IgnoreRingFusion,
        "PermissiveRingFusion": rdFMCS.RingCompare.PermissiveRingFusion,
        "StrictRingFusion": rdFMCS.RingCompare.StrictRingFusion,
    },
}

MCS_CONFIGS = {  # all args in https://rdkit.org/docs/source/rdkit.Chem.rdFMCS.html
    "maximizeBonds": bool,
    "threshold": float,
    "verbose": bool,
    "matchValences": bool,
    "ringMatchesRingOnly": bool,
    "completeRingsOnly": bool,
    "matchChiralTag": bool,
}


class MCSClustering(BaseClusterer):
    """A class for clustering molecules based on Maximum Common Substructure (MCS) similarity.

    Attributes:
        smiles_list: list of input SMILES strings representing the molecules to cluster.
        similarity_matrix: computed similarity matrix for the input SMILES.
        smarts_matrix: matrix containing SMARTS patterns of the MCS.
        timeout: a timeout for the MCS computation in seconds.
        mcs_kwargs: keyword arguments for the MCS algorithm.

    Usage:
    >>> smiles_list = [...]  # Your list of SMILES
    >>> mcs_cluster = MCSClustering(smiles_list)
    >>> mcs_cluster.compute_similarity_matrix()
    >>> labels = mcs_cluster.cluster_molecules(algorithm='DBSCAN')
    """

    def __init__(
        self,
        smiles_list=None,
        timeout=15,
        njobs: int = 8,
        np_dtypes: DTypeLike = np.float32,
        **mcs_kwargs,
    ):
        """Initialize the Maximum Common Substructure (MCS) clustering class with a
        list of SMILES.

        Args:
            smiles_list: a list of smiles.
            timeout: a timeout for the MCS computation in seconds. Defaults to 1.5.
            np_dtypes: numpy data type for the similarity matrix or arrays.
                Defaults to np.float32.
            mcs_kwargs: keyword arguments for the MCS algorithm. Will be parsed based
                on the values from MCS_CONFIGS.

        Usage:
            >>> smiles_list = [...]  # Your list of SMILES
            >>> mcs_cluster = MCSClustering(smiles_list)
            >>> mcs_cluster.compute_similarity_matrix()
            >>> labels = mcs_cluster.cluster_molecules(algorithm='DBSCAN')
        """
        super().__init__(smiles_list=smiles_list, njobs=njobs, np_dtypes=np_dtypes)
        self.timeout = timeout
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
        for key, value in mcs_kwargs.items():
            if key in ["atomCompare", "bondCompare", "ringCompare"]:
                self.mcs_kwargs[key] = MCS_COMPARE_CONFIGS[key][value]
            elif key in MCS_CONFIGS:
                self.mcs_kwargs[key] = value
            else:
                raise ValueError(
                    f"Unsupported MCS configuration: {key}. "
                    f"Supported configurations are: {list(MCS_COMPARE_CONFIGS.keys())}"
                )

    def find_mcs_in_many(self, smiles: list[str]):
        """Find the MCS in many molecules."""
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        return rdFMCS.FindMCS(mols, timeout=self.timeout, **self.mcs_kwargs)

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

    def pairwise_mcs_similarity(self, smipair) -> Tuple[list[str], list[float]]:
        """Helper function to compute similarity of molecule pair i and j."""
        smarts_string, similarity = self._mcs_similarity(smipair=smipair)
        return smarts_string, similarity

    def compute_similarity_matrix(
        self,
        smiles_list: Optional[list[str]] = None,
        show_progress=True,
        backend="loky",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the similarity matrix based on MCS for all molecules.

        Args:
            n_jobs: number of jobs for parallel processing. Defaults to 8.
            show_progress: whether to show the progress bar. Defaults to True.

        Returns:
            smarts_matrix: np.ndarray with the smarts patterns of the MCS's.
            simi_matrix: np.ndarray with the similarity matrix.
        """
        # create the similarity matrix with 1s in the diagonal
        if smiles_list is not None:
            self.smiles_list = smiles_list
        n_mols = len(self.smiles_list)
        simi_matrix = np.eye(n_mols, dtype=self.np_dtypes)
        # compute the similarity for all pairs of molecules and unpack results
        pairs = list(combinations(self.smiles_list, 2))
        applier = ParallelApplier(
            func=self.pairwise_mcs_similarity,
            iterable=pairs,
            n_jobs=self.njobs,
            show_progress=show_progress,
            backend=backend,
        )
        results = applier()
        smarts_strings, similarities = zip(*results)
        # take the indices of the upper triangle and populate the matrix
        r, c = np.triu_indices(n_mols, 1)  # row, column indices, respectively
        simi_matrix[r, c] = similarities
        # add values for the lower triangle
        simi_matrix += simi_matrix.T - np.eye(n_mols, dtype=self.np_dtypes)
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
