# -*- coding: utf-8 -*-
"""Module for clustering compounds within a dataframe using MCS clustering.

Usage example:

mcscluster --data_path <path_to_data> \
    --smiles_col touse_smiles \
    --score_col pchembl_value_median \
    --algorithm "DBSCAN" \
    --kwargs '{"eps": 0.3}' \
    --score_cutoff 7.0 \
    --score_col <e.g. pchembl_value> \
    --n_jobs 12
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from ..best_picker import mcs_based_clustering


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Applies the maximum common substructure algorithm for clustering molecules "
            "based on smiles structures as input. If desired, a column with scores can "
            "be used to only cluster compounds with a score above a certain threshold."
        )
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the dataframe containing the molecules to cluster.",
    )
    parser.add_argument(
        "--smiles_col",
        type=str,
        required=True,
        help="Column name containing the SMILES representation of molecules.",
    )
    parser.add_argument(
        "--score_col",
        type=str,
        help="Column name containing the scores to pick the best from.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--score_cutoff",
        type=str,
        help="Only cluster compounds where score_col > score_cutoff.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="DBSCAN",
        help=(
            "Clustering algorithm to use. Options include 'DBSCAN', 'Hierarchical', "
            "'Spectral', 'GraphBased'. Defaults to 'DBSCAN'"
        ),
    )
    parser.add_argument(
        "--kwargs",
        type=str,
        default="{}",
        help="Additional keyword arguments for clustering algorithm in JSON format.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=12,
        help="Number of jobs to run in parallel.",
    )
    parser.add_argument(
        "--pick_best",
        help="Whether to pick the best scoring compound from each cluster.",
        default=False,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if Path(args.data_path).suffix in [".gz", ".csv"]:
        df = pd.read_csv(args.data_path)
    elif Path(args.data_path).suffix == ".smi":
        smiles = Path(args.data_path).read_text().splitlines()
        df = pd.DataFrame(smiles, columns=[args.smiles_col])
        assert all(
            [args.score_col is None, args.score_cutoff is None]
        ), "If a SMILES file is provided, no score column should be provided."
    else:
        raise ValueError(
            "Data path should be a .csv, .gz or .smi file, but "
            f"{args.data_path} was provided."
        )
    kwargs = json.loads(args.kwargs)
    clustered_df = mcs_based_clustering(
        df,
        smiles_col=args.smiles_col,
        score_col=args.score_col,
        score_cutoff=args.score_cutoff,
        algorithm=args.algorithm,
        pick_best=args.pick_best,
        n_jobs=args.n_jobs,
        **kwargs,
    )
    clustered_df.to_csv(
        f"{args.data_path}_mcs_{args.algorithm}_clustered.csv", index=False
    )
