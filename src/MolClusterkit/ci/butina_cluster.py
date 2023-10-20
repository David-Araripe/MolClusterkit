# -*- coding: utf-8 -*-
"""Module for picking the best compounds within a dataframe using butina clustering.

For the preparation of datasets before querying SmallWorld, it could be ran as:

butinacluster --data_path <path_to_data> \
    --smiles_col touse_smiles \
    --score_col pchembl_value_median \
    --cutoff 0.7 \
    --score_cutoff 7.0 \
    --score_col <e.g. pchembl_value> \
    --n_jobs 12
"""

import argparse
from pathlib import Path

import pandas as pd

from ..best_picker import butina_based_clustering


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Applies the butina algorithm for clustering molecules "
            "based on smiles structures as input. If desired, a column with scores can "
            "be used to only cluster compounds with a score above a certain threshold."
        )
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path created by ModelBuilderABC.generate_results.",
    )
    parser.add_argument(
        "--smiles_col",
        type=str,
        required=True,
        help="Path to the file containing the molecules to screen.",
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
        "--cutoff",
        type=float,
        default=0.25,
        help=(
            "Cutoff for the butina clustering. The lower the value, "
            "the higher the amount of obtained clusters."
        ),
    )
    parser.add_argument(
        "--pick_best",
        help="Whether to pick the best scoring compound from each cluster.",
        default=False,
    )
    parser.add_argument("--njobs", type=int, default=12, help="Number of jobs.")
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
    butina_based_clustering(
        df,
        smiles_col=args.smiles_col,
        score_col=args.score_col,
        score_cutoff=args.score_cutoff,
        cutoff=args.cutoff,
        save_path=f"{args.data_path}_std_out_butina{int(args.cutoff*100)}.csv",
        pick_best=args.pick_best,
        njobs=args.njobs,
    )
