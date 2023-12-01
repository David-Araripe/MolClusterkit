# -*- coding: utf-8 -*-
r"""Module for clustering compounds within a dataframe using MCS clustering.

Usage example:

mcscluster -i "path/to/data.smi" \              # --input_path
    -smic "touse_smiles" \                      # --smiles_col
    -scor "pchembl_value_median" \              # --score_col
    -cut 7.0 \                                  # --score_cutoff
    -a "DBSCAN" \                               # --algorithm
    -k '{"eps": 0.3}' \                         # --kwargs
    -j 12 \                                     # --n_jobs
    -p \                                        # --pick_best
    -to 1.5 \                                   # --timeout
    -mcs '{"AtomCompare": "CompareElements"}' \ # --mcs_kwargs
    -o "path/to/output.csv"                     # --output_path
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
        "--input_path",
        "-i",
        dest="input_path",
        type=str,
        required=True,
        help="Path to the dataframe or .smi file with the SMILES to be clustered.",
    )
    parser.add_argument(
        "--smiles_col",
        "-smic",
        dest="smiles_col",
        type=str,
        required=False,
        help="Column name containing the SMILES representation of molecules. (if input is a dataframe...)",
    )
    parser.add_argument(
        "--score_col",
        "-scor",
        dest="score_col",
        type=str,
        help="Column name containing the scores to pick the best from.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--score_cutoff",
        "-cut",
        dest="score_cutoff",
        type=str,
        help="Only cluster compounds where score_col > score_cutoff.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--algorithm",
        "-a",
        dest="algorithm",
        type=str,
        default="DBSCAN",
        help=(
            "Clustering algorithm to use. Options include 'DBSCAN', 'Hierarchical', "
            "'Spectral', 'GraphBased'. Defaults to 'DBSCAN'"
        ),
    )
    parser.add_argument(
        "--kwargs",
        "-k",
        dest="kwargs",
        type=str,
        default="{}",
        help="Additional keyword arguments for clustering algorithm in JSON format.",
    )
    parser.add_argument(
        "--n_jobs",
        "-j",
        dest="n_jobs",
        type=int,
        default=8,
        help="Number of jobs to run in parallel.",
    )
    parser.add_argument(
        "--pick_best",
        "-p",
        dest="pick_best",
        help="Whether to pick the best scoring compound from each cluster.",
        default=False,
    )
    parser.add_argument(
        "--timeout",
        "-to",
        dest="timeout",
        help=(
            "Timeout time for the maximum common substructure algorithm. Input is the "
            "wall-clock seconds that the algorithm will use to find the MCS. "
            "Defaults to 1.5 seconds."
        ),
        default=15,
        type=float,
        required=False,
    )
    parser.add_argument(
        "--mcs_kwargs",
        "-mcs",
        dest="mcs_kwargs",
        help="Additional keyword arguments for MCS algorithm in JSON format. (see mcs.MCS_CONFIGS))",
        default="{}",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        dest="output_path",
        help="Path to save the clustered dataframe.",
        default=None,
        required=False,
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    if Path(args.input_path).suffix in [".gz", ".csv", ".tsv"]:
        if Path(args.input_path).suffix == ".tsv":
            sep = "\t"
        else:
            sep = ","
        data = pd.read_csv(args.input_path, sep=sep)
    elif Path(args.input_path).suffix == ".smi":
        data = Path(args.input_path).read_text().splitlines()
        assert all(
            [args.score_col is None, args.score_cutoff is None]
        ), "If a SMILES file is provided, no score column should be provided."
    else:
        raise ValueError(
            "Data path should be a .csv, .gz or .smi file, but "
            f"{args.input_path} was provided."
        )
    mcs_kwargs = json.loads(args.mcs_kwargs)
    kwargs = json.loads(args.kwargs)
    clustered_df = mcs_based_clustering(
        data,
        smiles_col=args.smiles_col,
        score_col=args.score_col,
        score_cutoff=args.score_cutoff,
        algorithm=args.algorithm,
        pick_best=args.pick_best,
        n_jobs=args.n_jobs,
        timeout=args.timeout,
        mcs_kwargs=mcs_kwargs,
        **kwargs,
    )
    if args.output_path is None:
        fname = Path(args.input_path).name.split(".")[0]
        output_path = f"{fname}_mcs_{args.algorithm}_clustered.csv"
    else:
        output_path = args.output_path
    clustered_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
