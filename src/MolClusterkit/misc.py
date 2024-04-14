# -*- coding: utf-8 -*-
"""Module containing small utilities functions used in the package."""
import re

import numpy as np
import pandas as pd
from rdkit import DataStructs

from .logger import logger


def find_smiles_column(df: pd.DataFrame) -> str:
    col_regex = re.compile("smiles", re.IGNORECASE)
    matching_cols = [col for col in df.columns if col_regex.findall(col)]
    if len(matching_cols) == 1:
        smiles_col = matching_cols[0]
    else:
        raise ValueError(
            "No smiles column was provided & no unique SMILES column in the dataframe."
        )
    return smiles_col


def reset_index_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """Function that checks if the dataframe index is reset. If not, resets it."""
    if not np.array_equal(np.array(range(len(df))), df.index.values):
        logger.warning("DataFrame index is not reset... Will reset it now.")
        df.reset_index(drop=True, inplace=True)
    return df


def TanimotoDist(fp1, fp2):
    sim = DataStructs.TanimotoSimilarity(fp1, fp2)
    return 1.0 - sim
