# -*- coding: utf-8 -*-
"""Utinity functions used by the command line interface."""
import argparse
from pathlib import Path


def process_output_dir(args: argparse.Namespace, cli: str) -> Path:
    """Process the output directory for the clustering results based on the CLI arguments."""
    input_path = args.input_path
    output_path = args.output_path
    if output_path is None:
        fname = Path(input_path).name.split(".")[0]
        if cli == "butina":
            fname = f"{fname}_butina_{int(args.dist_th*100)}_clustered.csv"
        elif cli == "mcs":
            fname = f"{fname}_mcs_{args.algorithm}_clustered.csv"
        output_dir = Path(input_path).parent
    else:
        fname = Path(output_path).name
        output_dir = Path(output_path).parent
    return output_dir / fname
