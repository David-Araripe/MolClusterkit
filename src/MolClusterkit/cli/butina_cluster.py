# -*- coding: utf-8 -*-
r"""Module for picking the best compounds within a dataframe using butina clustering.

For the preparation of datasets before querying SmallWorld, it could be ran as:

butinacluster -i "path/to/data.smi" \                # --input_path
    -smic "touse_smiles" \                           # --smiles_col
    -scor "pchembl_value_median" \                   # --score_col
    -cut 7.0 \                                       # --score_cutoff
    -dist 0.35 \                                     # --dist_th
    -j 12 \                                          # --n_jobs
    -p \                                             # --pick_best
    -o "path/to/output.csv"                          # --output_path
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
        "--dist_th",
        "-dist",
        dest="dist_th",
        required=False,
        default=0.35,
        type=float,
        help=(
            "distance threshold for the butina clustering algorithm. The lower the value "
            "the higher the amount of obtained clusters (and the more similar the compounds "
            "in each cluster). Defaults to 0.35."
        ),
    )
    parser.add_argument(
        "--n_jobs",
        "-j",
        dest="n_jobs",
        type=int,
        default=12,
        required=False,
        help="Number of jobs to run in parallel.",
    )
    parser.add_argument(
        "--pick_best",
        "-p",
        dest="pick_best",
        help="Activate the behaviour that the best scoring compound will be pickedfrom each cluster.",
        action="store_true",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        dest="output_path",
        help=(
            "Path to save the clustered dataframe. If not provided, will save in the "
            "same folder as the input data with "
        ),
        required=False,
        default=None,
    )
    return parser.parse_args()


def main():
    """Main function for the butina clustering."""
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
    clustered_data: pd.DataFrame = butina_based_clustering(
        data,
        smiles_col=args.smiles_col,
        score_col=args.score_col,
        score_cutoff=args.score_cutoff,
        dist_th=args.dist_th,
        njobs=args.n_jobs,
        pick_best=args.pick_best,
    )
    if args.output_path is None:
        fname = Path(args.input_path).name.split(".")[0]
        output_path = f"{fname}_butina_{int(args.dist_th*100)}_clustered.csv"
    else:
        output_path = args.output_path
    clustered_data.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
