import argparse as ap

from .cli import parse_args_run, run_evaluation


def get_args():
    parser = ap.ArgumentParser()
    subparsers = parser.add_subparsers(required=True, dest="subcommand")

    parser_prep = subparsers.add_parser("prep")
    parse_args_run(subparsers.add_parser("run"))

    return parser.parse_args()


def main():
    """
    Main function to run the evaluation.
    """
    args = get_args()
    if args.subcommand == "run":
        run_evaluation(args)
    elif args.subcommand == "pred":
        print("unimplemented")


if __name__ == "__main__":
    main()
