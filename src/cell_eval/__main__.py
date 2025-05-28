import argparse

import yaml


def parse_args():
    """
    CLI for evaluation
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained cell state model.")
    parser.add_argument(
        "-p",
        "--adata-pred",
        type=str,
        help="Path to the predicted adata object to evaluate",
    )
    parser.add_argument(
        "-r",
        "--adata-real",
        type=str,
        help="Path to the real adata object to evaluate against",
    )
    parser.add_argument(
        "--control-pert",
        type=str,
        default="non-targeting",
        help="Name of the control perturbation",
    )
    parser.add_argument(
        "--pert-col",
        type=str,
        default="target_name",
        help="Name of the column designated perturbations",
    )
    parser.add_argument(
        "--celltype-col",
        type=str,
        help="Name of the column designated celltype (optional)",
    )
    parser.add_argument(
        "--output-space",
        type=str,
        default="gene",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default="./cell-eval-outdir",
        help="Output directory to write to",
    )
    parser.add_argument(
        "--skip-dist-metrics",
        action="store_true",
    )
    parser.add_argument(
        "--skip-de-metrics",
        action="store_false",
    )
    parser.add_argument(
        "--skip-class-score",
        action="store_false",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
    )
    parser.add_argument(
        "--skip-normlog-check",
        action="store_true",
    )
    parser.add_argument(
        "--minimal-eval",
        action="store_true",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="wilcoxon",
    )
    parser.add_argument(
        "--fdr-threshold",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--config",
        type=str,
        help="If set, will load the config.yaml file from the output_dir and use it to set up the model. Note that CLI arguments are ignored if this is used",
    )
    return parser.parse_args()


def run_evaluation(args: argparse.ArgumentParser):
    from .metric_evaluator import MetricsEvaluator

    print("Reading adata objects")
    if args.config:
        # Read in config file
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        # Create the evaluator
        evaluator = MetricsEvaluator(
            path_pred=args.adata_pred,
            path_real=args.adata_real,
            include_dist_metrics=config["include_dist_metrics"],
            control_pert=config["control_pert"],
            pert_col=config["pert_col"],
            celltype_col=config["celltype_col"],
            output_space=config["output_space"],
            shared_perts=config["shared_perts"],
            outdir=config["outdir"],
            de_metric=config["de_metric"],
            class_score=config["class_score"],
            n_threads=config["n_threads"] if "n_threads" in config else None,
            batch_size=config["batch_size"] if "batch_size" in config else None,
            metric=config["metric"] if "metric" in config else "wilcoxon",
        )
    else:
        print(args)

        evaluator = MetricsEvaluator(
            path_pred=args.adata_pred,
            path_real=args.adata_real,
            include_dist_metrics=args.skip_dist_metrics,
            control_pert=args.control_pert,
            pert_col=args.pert_col,
            celltype_col=args.celltype_col,
            output_space=args.output_space,
            outdir=args.outdir,
            de_metric=args.skip_de_metrics,
            class_score=args.skip_class_score,
            n_threads=args.num_threads,
            batch_size=args.batch_size,
            skip_normlog_check=args.skip_normlog_check,
            minimal_eval=args.minimal_eval,
            metric=args.metric,
            fdr_threshold=args.fdr_threshold,
        )

    print("Running evaluation")
    # Compute the metrics
    evaluator.compute()

    # Save the metrics
    evaluator.save_metrics_per_celltype()
    print("Done")


def main():
    """
    Main function to run the evaluation.
    """

    # Parse arguments
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
