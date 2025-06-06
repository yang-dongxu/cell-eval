import argparse as ap
import logging

from .cli import (
    parse_args_prep,
    parse_args_run,
    run_evaluation,
    run_prep,
)

logger = logging.getLogger(__name__)


def get_args():
    parser = ap.ArgumentParser()
    subparsers = parser.add_subparsers(required=True, dest="subcommand")
    parse_args_prep(subparsers.add_parser("prep"))
    parse_args_run(subparsers.add_parser("run"))
    return parser.parse_args()


def main():
    """
    Main function to run the evaluation.
    """
    args = get_args()
    if args.subcommand == "run":
        run_evaluation(args)
    elif args.subcommand == "prep":
        run_prep(args)
    else:
        raise ValueError(f"Unrecognized subcommand: {args.subcommand}")


if __name__ == "__main__":
    main()
