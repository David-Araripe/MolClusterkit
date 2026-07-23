# -*- coding: utf-8 -*-
"""Module containing the MCS clustering class."""
from itertools import combinations
from math import ceil
from typing import Literal, Optional, Tuple

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
            timeout: a timeout for the MCS computation in seconds. Must be positive;
                fractional values are rounded up to whole seconds, since RDKit only
                accepts an integer timeout. Defaults to 15.
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
        self.timeout = self._set_timeout(timeout)
        self.mcs_kwargs = {}
        self._setup_mcs_configs(**mcs_kwargs)
        self._check_low_timeout()

    def _set_timeout(self, timeout) -> int:
        """Coerce the timeout to the whole number of seconds that RDKit expects.

        `rdFMCS.FindMCS` only accepts an unsigned int, so floats are rounded up.
        Rounding up (rather than to nearest) keeps a sub-second timeout from
        becoming 0, which RDKit interprets as "no timeout at all".

        Args:
            timeout: requested timeout in seconds.

        Raises:
            TypeError: if the timeout is not a number.
            ValueError: if the timeout is not strictly positive.

        Returns:
            int: the timeout in whole seconds.
        """
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise TypeError(
                f"timeout must be a number of seconds, got {type(timeout).__name__}."
            )
        if timeout <= 0:
            raise ValueError(
                f"timeout must be a positive number of seconds, got {timeout}."
            )
        return ceil(timeout)

    def _check_low_timeout(self):
        """Check if the timeout is too low."""
        if self.timeout < 2:
            logger.warning(
                f"A timeout of {self.timeout}s is very low. The MCS search may be "
                "cancelled before it reaches the true maximum common substructure, "
                "which underestimates the similarity. Consider increasing the timeout."
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

    def mcs_in_many(self, smiles: list[str]):
        """Find the MCS in two or more molecules, given their SMILES.

        Args:
            smiles: list of SMILES strings.

        Returns:
            mcs_result: the MCS result object."""

        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        return rdFMCS.FindMCS(mols, timeout=self.timeout, **self.mcs_kwargs)

    def mcs_similarity(
        self,
        smipair: Tuple[str, str],
        similarity_metric: Literal["johnson", "smaller/mces"] = "johnson",
    ):
        """Compute MCES between two molecules given their SMILES, returning the SMARTS
        pattern and a score. Two metrics are available to calculate the similarity between
        the two molecules, with values ranging from 0 to 1:

        1. Johnson metric (default):
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
            smipair: tuple of two SMILES strings.
            similarity_metric: the similarity metric to use. Options are 'johnson' or
                'smaller/mces'. Defaults to 'johnson'.

        Raises:
            ValueError: if the SMILES cannot be parsed into molecules, or if an
                unsupported similarity_metric is given.

        Returns:
            smarts_string: the SMARTS pattern of the MCS.
            similarity: the fraction of matched atoms to the smaller molecule.
        """

        mols = [Chem.MolFromSmiles(smi) for smi in smipair]
        if any([mols[0] is None, mols[1] is None]):
            logger.error(
                f"Could not parse: {smipair[0]} or {smipair[1]}!!\nRemove invalid SMILES..."
            )
            raise ValueError("Could not parse SMILES into molecules.")
        mcs_result = rdFMCS.FindMCS(list(mols), timeout=self.timeout, **self.mcs_kwargs)
        if similarity_metric == "johnson":
            mcs_atoms, mcs_bonds = mcs_result.numAtoms, mcs_result.numBonds
            mol1_atoms, mol1_bonds = mols[0].GetNumAtoms(), mols[0].GetNumBonds()
            mol2_atoms, mol2_bonds = mols[1].GetNumAtoms(), mols[1].GetNumBonds()
            simi_metric = (mcs_atoms + mcs_bonds) ** 2 / (
                (mol1_atoms + mol1_bonds) * (mol2_atoms + mol2_bonds)
            )
        elif similarity_metric == "smaller/mces":
            simi_metric = mcs_result.numAtoms / min(
                mols[0].GetNumAtoms(), mols[1].GetNumAtoms()
            )
        else:
            raise ValueError(
                f"Unsupported similarity_metric: {similarity_metric!r}. "
                "Valid options are 'johnson' and 'smaller/mces'."
            )
        return mcs_result.smartsString, simi_metric

    def compute_similarity_matrix(
        self,
        smiles_list: Optional[list[str]] = None,
        show_progress=True,
        similarity_metric: Literal["johnson", "smaller/mces"] = "johnson",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the similarity matrix based on MCS for all molecules. Parallel
        processing uses the `njobs` set on the instance.

        Args:
            smiles_list: optional list of SMILES overriding the one held by the
                instance. Defaults to None.
            show_progress: whether to show the progress bar. Defaults to True.
            similarity_metric: the similarity metric to use for every pair. Options are
                'johnson' or 'smaller/mces'. Defaults to 'johnson'.

        Returns:
            smarts_matrix: np.ndarray with the smarts patterns of the MCS's.
            simi_matrix: np.ndarray with the similarity matrix.
        """
        # create the similarity matrix with 1s in the diagonal
        self._resolve_smiles_list(smiles_list)
        n_mols = len(self.smiles_list)
        simi_matrix = np.eye(n_mols, dtype=self.np_dtypes)
        # compute the similarity for all pairs of molecules and unpack results
        pairs = list(combinations(self.smiles_list, 2))
        applier = ParallelApplier(
            func=self.mcs_similarity,
            iterable=pairs,
            n_jobs=self.njobs,
            show_progress=show_progress,
        )
        results = applier(similarity_metric=similarity_metric)
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
