import argparse as ap
import logging

from .cli import (
    parse_args_prep,
    parse_args_run,
    run_evaluation,
    run_prep,
    parse_args_metrics,
    run_metrics,
)

logger = logging.getLogger(__name__)


def get_args():
    parser = ap.ArgumentParser()
    subparsers = parser.add_subparsers(required=True, dest="subcommand")
    parse_args_prep(subparsers.add_parser("prep"))
    parse_args_run(subparsers.add_parser("run"))
    parse_args_metrics(subparsers.add_parser("metrics"))
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
    elif args.subcommand == "metrics":
        run_metrics(args)
    else:
        raise ValueError(f"Unrecognized subcommand: {args.subcommand}")


if __name__ == "__main__":
    main()
