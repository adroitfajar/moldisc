#!/usr/bin/env python3
"""Run a MolDisc campaign from a versioned JSON configuration."""

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from moldisc import moldisc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pause-after-cycles", type=int)
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Train/load SMILES-X and exit before GPT generation.",
    )
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.train_only:
        config.update(
            {
                "max_generation": -1,
                "max_generated_molecules": 0,
                "cycles": -1,
                "option_save": False,
            }
        )
    config["resume"] = args.resume
    config["pause_after_cycles"] = args.pause_after_cycles
    result = moldisc(**config)
    print(f"Returned {len(result)} selected molecules.")


if __name__ == "__main__":
    main()
