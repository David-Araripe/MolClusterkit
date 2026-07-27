# -*- coding: utf-8 -*-
"""Module containing small utilities functions used in the package."""
from rdkit import DataStructs


def TanimotoDist(fp1, fp2):
    sim = DataStructs.TanimotoSimilarity(fp1, fp2)
    return 1.0 - sim
