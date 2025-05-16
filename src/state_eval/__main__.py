import argparse

import yaml

from .metric_evaluator import MetricsEvaluator


def parse_args():
    """
    CLI for evaluation
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained cell state model.")
    parser.add_argument(
        "--adata_pred",
        type=str,
        default="/home/yhr/state-eval/adata_pred_subset.h5ad",
        help="Path to the predicted adata object to evaluate",
    )
    parser.add_argument(
        "--adata_true",
        type=str,
        default="/home/yhr/state-eval/adata_real_subset.h5ad",
        help="Path to the true adata object to evaluate against",
    )
    parser.add_argument(
        "--eval_config",
        type=str,
        default="/home/yhr/state-eval/config/eval_config.yaml",
        help="If set, will load the config.yaml file from the output_dir and use it to set up the model.",
    )
    return parser.parse_args()


def main():
    """
    Main function to run the evaluation.
    """

    # Parse arguments
    args = parse_args()
    print("Reading adata objects")

    # Read in config file
    with open(args.eval_config, "r") as f:
        config = yaml.safe_load(f)

    print("Running evaluation")
    # Create the evaluator
    evaluator = MetricsEvaluator(
        path_pred=args.adata_pred,
        path_real=args.adata_true,
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

    # Compute the metrics
    evaluator.compute()

    # Save the metrics
    evaluator.save_metrics_per_celltype()
    print("Done")


if __name__ == "__main__":
    main()
