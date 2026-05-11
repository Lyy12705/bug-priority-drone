import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from train_boundary_priority_model import (
    BOUNDARY_LABELS,
    LABELS,
    boundary_selection_score,
    class_report_rows,
    fit_candidate,
    load_features,
    metric_row,
    predict_refined,
)
from train_classifier_engine import build_candidates, parse_float_list, selection_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIGS = [
    {
        "name": "default",
        "summary_weight": 2.0,
        "description_weight": 1.0,
        "bigram_summary_weight": 2.0,
        "bigram_description_weight": 1.0,
        "k1": 1.2,
        "summary_b": 0.75,
        "description_b": 0.75,
    },
    {
        "name": "summary_high",
        "summary_weight": 3.0,
        "description_weight": 0.8,
        "bigram_summary_weight": 3.0,
        "bigram_description_weight": 0.8,
        "k1": 1.2,
        "k3": 100.0,
        "summary_b": 0.65,
        "description_b": 0.85,
    },
    {
        "name": "p2_error_keywords",
        "summary_weight": 3.4,
        "description_weight": 0.7,
        "bigram_summary_weight": 3.8,
        "bigram_description_weight": 0.6,
        "k1_unigram": 1.0,
        "k1_bigram": 1.1,
        "k3_unigram": 50.0,
        "k3_bigram": 80.0,
        "summary_b_unigram": 0.55,
        "summary_b_bigram": 0.55,
        "description_b_unigram": 0.90,
        "description_b_bigram": 0.90,
    },
    {
        "name": "summary_extreme",
        "summary_weight": 4.0,
        "description_weight": 0.6,
        "bigram_summary_weight": 3.5,
        "bigram_description_weight": 0.6,
        "k1": 1.2,
        "k3": 60.0,
        "summary_b": 0.55,
        "description_b": 0.90,
    },
    {
        "name": "bigram_high",
        "summary_weight": 2.8,
        "description_weight": 0.8,
        "bigram_summary_weight": 4.0,
        "bigram_description_weight": 0.5,
        "k1": 1.1,
        "k3": 80.0,
        "summary_b": 0.60,
        "description_b": 0.85,
    },
    {
        "name": "soft_summary_high",
        "summary_weight": 3.0,
        "description_weight": 0.8,
        "bigram_summary_weight": 3.0,
        "bigram_description_weight": 0.8,
        "k1": 0.9,
        "summary_b": 0.45,
        "description_b": 0.65,
    },
    {
        "name": "balanced_low_k1",
        "summary_weight": 2.2,
        "description_weight": 1.1,
        "bigram_summary_weight": 2.2,
        "bigram_description_weight": 1.1,
        "k1": 0.8,
        "summary_b": 0.55,
        "description_b": 0.75,
    },
    {
        "name": "description_balanced",
        "summary_weight": 1.5,
        "description_weight": 1.5,
        "bigram_summary_weight": 1.5,
        "bigram_description_weight": 1.5,
        "k1": 1.2,
        "summary_b": 0.75,
        "description_b": 0.75,
    },
    {
        "name": "strong_norm",
        "summary_weight": 2.0,
        "description_weight": 1.0,
        "bigram_summary_weight": 2.0,
        "bigram_description_weight": 1.0,
        "k1": 1.6,
        "summary_b": 0.85,
        "description_b": 0.95,
    },
    {
        "name": "soft_norm",
        "summary_weight": 2.0,
        "description_weight": 1.0,
        "bigram_summary_weight": 2.0,
        "bigram_description_weight": 1.0,
        "k1": 0.9,
        "summary_b": 0.45,
        "description_b": 0.55,
    },
]


def project_path(relative_path: str) -> str:
    return os.path.join(PROJECT_ROOT, relative_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Small BM25/BM25F parameter search with boundary-refined evaluation."
    )
    parser.add_argument("--config-names", default="summary_high")
    parser.add_argument("--model-types", default="sgd_log")
    parser.add_argument("--alpha-values", default="0.001,0.01,0.1")
    parser.add_argument("--c-values", default="0.1,0.3,1.0")
    parser.add_argument("--xgb-n-estimators", default="80", help="Comma-separated XGBoost estimator counts.")
    parser.add_argument("--xgb-max-depths", default="3", help="Comma-separated XGBoost max depths.")
    parser.add_argument("--xgb-learning-rates", default="0.08", help="Comma-separated XGBoost learning rates.")
    parser.add_argument("--lgbm-n-estimators", default="80", help="Comma-separated LightGBM estimator counts.")
    parser.add_argument("--lgbm-num-leaves", default="31", help="Comma-separated LightGBM num_leaves values.")
    parser.add_argument("--lgbm-learning-rates", default="0.08", help="Comma-separated LightGBM learning rates.")
    parser.add_argument("--selection-metric", choices=["macro_f1", "accuracy", "guarded_macro"], default="guarded_macro")
    parser.add_argument("--boundary-apply", choices=["base_1_2", "base_2_3", "base_1_2_3"], default="base_1_2")
    parser.add_argument(
        "--boundary-apply-values",
        default=None,
        help="Optional comma-separated modes to search: base_1_2,base_2_3,base_1_2_3.",
    )
    parser.add_argument("--p2-weight", type=float, default=0.10)
    parser.add_argument("--p2-weight-values", default=None, help="Optional comma-separated P2 objective weights.")
    parser.add_argument("--p1-weight", type=float, default=0.00)
    parser.add_argument("--p3-weight", type=float, default=0.00)
    parser.add_argument("--off-by-one-weight", type=float, default=0.00)
    parser.add_argument("--collapse-floor", type=float, default=0.55)
    parser.add_argument("--collapse-floor-values", default=None, help="Optional comma-separated P1/P3 recall floors.")
    parser.add_argument("--accuracy-drop-weight", type=float, default=0.10)
    parser.add_argument("--collapse-penalty-weight", type=float, default=0.20)
    parser.add_argument("--max-accuracy-drop", type=float, default=1.0)
    parser.add_argument("--boundary-p2-sample-weights", default="1.0", help="Comma-separated P2 sample weights for boundary model.")
    parser.add_argument("--input-clean", default=project_path("data/processed/eclipse_bug_reports_clean_with_dupe.csv"))
    parser.add_argument("--train-split", default=project_path("data/processed/experiment_splits_with_dupe/train_balanced_clean.csv"))
    parser.add_argument("--validation-split", default=project_path("data/processed/experiment_splits_with_dupe/validation_balanced_clean.csv"))
    parser.add_argument("--natural-split", default=project_path("data/processed/experiment_splits_with_dupe/natural_test_clean.csv"))
    parser.add_argument("--output-csv", default=project_path("reports/bm25_boundary_grid_search.csv"))
    parser.add_argument("--summary-md", default=project_path("reports/bm25_boundary_grid_search.md"))
    parser.add_argument("--best-model-path", default=project_path("models/bm25_boundary_best_model.joblib"))
    parser.add_argument("--best-eval-csv", default=project_path("reports/bm25_boundary_best_eval.csv"))
    parser.add_argument("--best-class-report-csv", default=project_path("reports/bm25_boundary_best_class_report.csv"))
    parser.add_argument("--skip-feature-build", action="store_true", help="Reuse existing feature dirs and REP JSON files.")
    return parser


def selected_configs(names: str) -> list[dict]:
    requested = {name.strip() for name in names.split(",") if name.strip()}
    configs = [config for config in CONFIGS if config["name"] in requested]
    missing = requested - {config["name"] for config in configs}
    if missing:
        raise ValueError(f"Unknown config names: {sorted(missing)}")
    return configs


def parse_str_values(value: str | None, default: str) -> list[str]:
    if not value:
        value = default
    return [item.strip() for item in value.split(",") if item.strip()]


def run_command(command: list[str]) -> None:
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def rep_json_path(config_name: str) -> str:
    return project_path(f"models/rep_minus_weights_bm25_{config_name}.json")


def feature_dir(config_name: str, split: str) -> str:
    return project_path(f"data/processed/features_bm25_{config_name}_{split}")


def config_value(config: dict, specific_key: str, shared_key: str, default: float) -> float:
    return float(config.get(specific_key, config.get(shared_key, default)))


def build_rep_and_features(args: argparse.Namespace, config: dict) -> None:
    config_name = config["name"]
    rep_path = rep_json_path(config_name)
    if not args.skip_feature_build:
        run_command([
            sys.executable,
            "scripts/train_rep_minus_weights.py",
            "--input",
            args.input_clean,
            "--output",
            rep_path,
            "--summary-weight",
            str(config["summary_weight"]),
            "--description-weight",
            str(config["description_weight"]),
            "--bigram-summary-weight",
            str(config["bigram_summary_weight"]),
            "--bigram-description-weight",
            str(config["bigram_description_weight"]),
            "--k1-unigram",
            str(config_value(config, "k1_unigram", "k1", 1.2)),
            "--k1-bigram",
            str(config_value(config, "k1_bigram", "k1", 1.2)),
            "--k3-unigram",
            str(config_value(config, "k3_unigram", "k3", 100.0)),
            "--k3-bigram",
            str(config_value(config, "k3_bigram", "k3", 100.0)),
            "--summary-b-unigram",
            str(config_value(config, "summary_b_unigram", "summary_b", 0.75)),
            "--summary-b-bigram",
            str(config_value(config, "summary_b_bigram", "summary_b", 0.75)),
            "--description-b-unigram",
            str(config_value(config, "description_b_unigram", "description_b", 0.75)),
            "--description-b-bigram",
            str(config_value(config, "description_b_bigram", "description_b", 0.75)),
        ])
        run_command([
            sys.executable,
            "scripts/build_features.py",
            "--input",
            args.train_split,
            "--feature-dir",
            feature_dir(config_name, "train"),
            "--mode",
            "fit",
            "--text-feature-mode",
            "enhanced",
            "--related-mode",
            "rep_minus",
            "--rep-weights-json",
            rep_path,
        ])
        run_command([
            sys.executable,
            "scripts/build_features.py",
            "--input",
            args.validation_split,
            "--feature-dir",
            feature_dir(config_name, "validation"),
            "--mode",
            "transform",
            "--reference-feature-dir",
            feature_dir(config_name, "train"),
            "--history-input",
            args.train_split,
            "--text-feature-mode",
            "enhanced",
            "--related-mode",
            "rep_minus",
            "--rep-weights-json",
            rep_path,
        ])
        run_command([
            sys.executable,
            "scripts/build_features.py",
            "--input",
            args.natural_split,
            "--feature-dir",
            feature_dir(config_name, "natural_test"),
            "--mode",
            "transform",
            "--reference-feature-dir",
            feature_dir(config_name, "train"),
            "--history-input",
            args.train_split,
            "--text-feature-mode",
            "enhanced",
            "--related-mode",
            "rep_minus",
            "--rep-weights-json",
            rep_path,
        ])


def train_and_evaluate_config(args: argparse.Namespace, config: dict) -> dict:
    config_name = config["name"]
    X_train, y_train, _ = load_features(feature_dir(config_name, "train"))
    X_val, y_val, _ = load_features(feature_dir(config_name, "validation"))
    X_nat, y_nat, _ = load_features(feature_dir(config_name, "natural_test"))

    best_base_model = None
    best_base_row = None
    best_base_score = -np.inf
    for model_type, params, model in build_candidates(args):
        fit_candidate(model, X_train, y_train)
        y_val_pred = model.predict(X_val).astype(int)
        row = metric_row(y_val, y_val_pred, "validation")
        row.update({
            "candidate_model_type": model_type,
            "candidate_params": params,
            "selection_score": selection_score(row, args.selection_metric),
        })
        if row["selection_score"] > best_base_score:
            best_base_score = float(row["selection_score"])
            best_base_model = model
            best_base_row = row

    if best_base_model is None or best_base_row is None:
        raise RuntimeError(f"No base model selected for {config_name}.")

    train_boundary_mask = np.isin(y_train, BOUNDARY_LABELS)
    X_boundary = X_train[train_boundary_mask]
    y_boundary = y_train[train_boundary_mask]

    best_boundary_model = None
    best_boundary_row = None
    best_boundary_score = -np.inf
    boundary_apply_values = parse_str_values(args.boundary_apply_values, args.boundary_apply)
    p2_weight_values = parse_float_list(args.p2_weight_values or str(args.p2_weight))
    collapse_floor_values = parse_float_list(args.collapse_floor_values or str(args.collapse_floor))
    boundary_p2_sample_weights = parse_float_list(args.boundary_p2_sample_weights)
    for boundary_apply in boundary_apply_values:
        for p2_weight in p2_weight_values:
            for collapse_floor in collapse_floor_values:
                for boundary_p2_sample_weight in boundary_p2_sample_weights:
                    sample_weight = np.ones(len(y_boundary), dtype=float)
                    sample_weight[y_boundary == 2] = boundary_p2_sample_weight
                    for model_type, params, model in build_candidates(args):
                        fit_candidate(model, X_boundary, y_boundary, sample_weight=sample_weight)
                        y_val_refined = predict_refined(best_base_model, model, X_val, boundary_apply)
                        row = metric_row(y_val, y_val_refined, "validation")
                        score = boundary_selection_score(
                            row,
                            best_base_row,
                            p2_weight=p2_weight,
                            collapse_floor=collapse_floor,
                            p1_weight=args.p1_weight,
                            p3_weight=args.p3_weight,
                            off_by_one_weight=args.off_by_one_weight,
                            accuracy_drop_weight=args.accuracy_drop_weight,
                            collapse_penalty_weight=args.collapse_penalty_weight,
                            max_accuracy_drop=args.max_accuracy_drop,
                        )
                        row.update({
                            "boundary_model_type": model_type,
                            "boundary_params": params,
                            "boundary_apply": boundary_apply,
                            "boundary_p2_sample_weight": boundary_p2_sample_weight,
                            "objective_p2_weight": p2_weight,
                            "objective_collapse_floor": collapse_floor,
                            "selection_score": score,
                        })
                        if score > best_boundary_score:
                            best_boundary_score = float(score)
                            best_boundary_model = model
                            best_boundary_row = row

    if best_boundary_model is None or best_boundary_row is None:
        raise RuntimeError(f"No boundary model selected for {config_name}.")

    y_base_nat = best_base_model.predict(X_nat).astype(int)
    y_refined_nat = predict_refined(best_base_model, best_boundary_model, X_nat, best_boundary_row["boundary_apply"])
    base_nat = metric_row(y_nat, y_base_nat, "natural_holdout")
    refined_nat = metric_row(y_nat, y_refined_nat, "natural_holdout")

    with open(rep_json_path(config_name), "r", encoding="utf-8") as fh:
        rep_config = json.load(fh)

    result = {
        "config_name": config_name,
        **{f"param_{key}": value for key, value in config.items() if key != "name"},
        "rep_unigram_weight": rep_config.get("unigram_feature_weight"),
        "rep_bigram_weight": rep_config.get("bigram_feature_weight"),
        "rep_product_weight": rep_config.get("product_weight"),
        "rep_component_weight": rep_config.get("component_weight"),
        "rep_severity_weight": rep_config.get("severity_weight"),
        "base_model_type": best_base_row["candidate_model_type"],
        "base_params": best_base_row["candidate_params"],
        "boundary_model_type": best_boundary_row["boundary_model_type"],
        "boundary_params": best_boundary_row["boundary_params"],
        "boundary_apply": best_boundary_row["boundary_apply"],
        "boundary_p2_sample_weight": best_boundary_row["boundary_p2_sample_weight"],
        "objective_p1_weight": args.p1_weight,
        "objective_p2_weight": best_boundary_row["objective_p2_weight"],
        "objective_p3_weight": args.p3_weight,
        "objective_off_by_one_weight": args.off_by_one_weight,
        "objective_collapse_floor": best_boundary_row["objective_collapse_floor"],
        "objective_accuracy_drop_weight": args.accuracy_drop_weight,
        "objective_collapse_penalty_weight": args.collapse_penalty_weight,
        "objective_max_accuracy_drop": args.max_accuracy_drop,
        "validation_score": best_boundary_score,
    }
    for key, value in base_nat.items():
        if key not in {"eval_set", "rows"}:
            result[f"base_{key}"] = value
    for key, value in refined_nat.items():
        if key not in {"eval_set", "rows"}:
            result[f"refined_{key}"] = value

    result["_model_bundle"] = {
        "model_type": "boundary_refined_direct_classifier",
        "feature_profile": f"bm25_{config_name}",
        "base_model": best_base_model,
        "base_settings": deepcopy(best_base_row),
        "boundary_model": best_boundary_model,
        "boundary_settings": deepcopy(best_boundary_row),
        "boundary_apply": best_boundary_row["boundary_apply"],
        "boundary_objective": {
            "p1_weight": args.p1_weight,
            "p2_weight": best_boundary_row["objective_p2_weight"],
            "p3_weight": args.p3_weight,
            "off_by_one_weight": args.off_by_one_weight,
            "collapse_floor": best_boundary_row["objective_collapse_floor"],
            "accuracy_drop_weight": args.accuracy_drop_weight,
            "collapse_penalty_weight": args.collapse_penalty_weight,
            "max_accuracy_drop": args.max_accuracy_drop,
            "boundary_p2_sample_weight": best_boundary_row["boundary_p2_sample_weight"],
        },
        "labels": LABELS,
        "feature_dirs": {
            "train": feature_dir(config_name, "train"),
            "validation": feature_dir(config_name, "validation"),
            "natural_holdout": feature_dir(config_name, "natural_test"),
        },
        "rep_weights_json": rep_json_path(config_name),
        "bm25_config": config,
    }
    result["_y_nat"] = y_nat
    result["_y_refined_nat"] = y_refined_nat
    return result


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    view = df[columns].copy()
    for column in columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: f"{value:.4f}")
    try:
        return view.to_markdown(index=False)
    except ImportError:
        header = "| " + " | ".join(columns) + " |"
        divider = "| " + " | ".join(["---"] * len(columns)) + " |"
        rows = ["| " + " | ".join(str(row[column]) for column in columns) + " |" for _, row in view.iterrows()]
        return "\n".join([header, divider, *rows])


def write_summary(path: str, rows: pd.DataFrame, best: pd.Series) -> None:
    cols = [
        "config_name",
        "base_model_type",
        "base_params",
        "boundary_model_type",
        "boundary_params",
        "boundary_apply",
        "boundary_p2_sample_weight",
        "objective_p1_weight",
        "objective_p2_weight",
        "objective_p3_weight",
        "objective_off_by_one_weight",
        "objective_collapse_floor",
        "refined_accuracy",
        "refined_macro_f1",
        "refined_p1_recall",
        "refined_p2_recall",
        "refined_p3_recall",
        "refined_mae",
        "rep_product_weight",
        "rep_component_weight",
        "rep_severity_weight",
    ]
    lines = [
        "# BM25 / BM25F Boundary Grid Search",
        "",
        "This experiment tests a small set of BM25F field-weight and normalization settings.",
        "",
        "Each config retrains REP- weights from duplicate links, rebuilds improved DRONE/REP- features, then evaluates the boundary-refined model.",
        "",
        "## Best Config",
        "",
        f"- Config: `{best['config_name']}`",
        f"- Accuracy: `{best['refined_accuracy']:.4f}`",
        f"- Macro F1: `{best['refined_macro_f1']:.4f}`",
        f"- P2 recall: `{best['refined_p2_recall']:.4f}`",
        f"- MAE: `{best['refined_mae']:.4f}`",
        "",
        "## All Configs",
        "",
        markdown_table(rows.sort_values(["refined_macro_f1", "refined_accuracy"], ascending=False), cols),
        "",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main() -> None:
    args = build_parser().parse_args()
    configs = selected_configs(args.config_names)
    rows = []
    full_results = []
    for config in configs:
        print(f"\n=== BM25 config: {config['name']} ===")
        build_rep_and_features(args, config)
        result = train_and_evaluate_config(args, config)
        full_results.append(result)
        clean_row = {key: value for key, value in result.items() if not key.startswith("_")}
        rows.append(clean_row)
        print(
            f"{config['name']}: refined_acc={clean_row['refined_accuracy']:.4f} "
            f"macro={clean_row['refined_macro_f1']:.4f} p2={clean_row['refined_p2_recall']:.4f}"
        )

    rows_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    rows_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    best_idx = rows_df.sort_values(
        ["refined_macro_f1", "refined_accuracy", "refined_p2_recall"],
        ascending=False,
    ).index[0]
    best_row = rows_df.loc[best_idx]
    best_result = full_results[best_idx]

    os.makedirs(os.path.dirname(args.best_model_path), exist_ok=True)
    joblib.dump(best_result["_model_bundle"], args.best_model_path)

    best_eval = pd.DataFrame([
        {
            "model_stage": "bm25_boundary_best",
            "eval_set": "natural_holdout",
            "rows": int(len(best_result["_y_nat"])),
            "accuracy": best_row["refined_accuracy"],
            "macro_f1": best_row["refined_macro_f1"],
            "weighted_f1": best_row["refined_weighted_f1"],
            "off_by_one_accuracy": best_row["refined_off_by_one_accuracy"],
            "mae": best_row["refined_mae"],
            "high_priority_recall_p1_p2": best_row["refined_high_priority_recall_p1_p2"],
            "low_priority_recall_p4_p5": best_row["refined_low_priority_recall_p4_p5"],
            "p1_recall": best_row["refined_p1_recall"],
            "p2_recall": best_row["refined_p2_recall"],
            "p3_recall": best_row["refined_p3_recall"],
            "p4_recall": best_row["refined_p4_recall"],
            "p5_recall": best_row["refined_p5_recall"],
        }
    ])
    best_eval.to_csv(args.best_eval_csv, index=False, encoding="utf-8-sig")
    class_df = pd.DataFrame(
        class_report_rows(best_result["_y_nat"], best_result["_y_refined_nat"], "natural_holdout")
    )
    class_df.to_csv(args.best_class_report_csv, index=False, encoding="utf-8-sig")
    write_summary(args.summary_md, rows_df, best_row)

    print(f"\nsaved grid -> {args.output_csv}")
    print(f"saved report -> {args.summary_md}")
    print(f"saved best model -> {args.best_model_path}")
    print(f"saved best eval -> {args.best_eval_csv}")
    print("\n=== Best BM25/BM25F Config ===")
    print(best_eval.to_string(index=False))
    print("\nConfusion matrix:")
    print(confusion_matrix(best_result["_y_nat"], best_result["_y_refined_nat"], labels=LABELS))


if __name__ == "__main__":
    main()
