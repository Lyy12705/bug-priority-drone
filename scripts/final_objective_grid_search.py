import argparse
import os

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from train_boundary_priority_model import (
    BOUNDARY_LABELS,
    LABELS,
    class_report_rows,
    fit_candidate,
    metric_row,
    predict_refined,
)
from train_classifier_engine import build_candidates, parse_float_list

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def project_path(relative_path: str) -> str:
    return os.path.join(PROJECT_ROOT, relative_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Grid search the final objective: accuracy + macro F1 + P2 recall balance.",
    )
    parser.add_argument("--train-feature-dir", default=project_path("data/processed/features_bm25_p2_error_keywords_train"))
    parser.add_argument("--validation-feature-dir", default=project_path("data/processed/features_bm25_p2_error_keywords_validation"))
    parser.add_argument("--natural-feature-dir", default=project_path("data/processed/features_bm25_p2_error_keywords_natural_test"))
    parser.add_argument("--model-path", default=project_path("models/final_objective_best_model.joblib"))
    parser.add_argument("--candidate-csv", default=project_path("reports/final_objective_grid_search.csv"))
    parser.add_argument("--eval-csv", default=project_path("reports/final_objective_best_eval.csv"))
    parser.add_argument("--class-report-csv", default=project_path("reports/final_objective_best_class_report.csv"))
    parser.add_argument("--summary-md", default=project_path("reports/final_objective_grid_search.md"))
    parser.add_argument("--model-types", default="sgd_log,linear_svm,ridge_classifier")
    parser.add_argument("--alpha-values", default="0.001,0.01,0.03,0.1,0.3")
    parser.add_argument("--c-values", default="0.1,0.3,1.0")
    parser.add_argument("--boundary-apply-values", default="base_1_2,base_1_2_3,base_2_3")
    parser.add_argument("--boundary-p2-sample-weights", default="1.0,1.1,1.2,1.4")
    parser.add_argument("--accuracy-weight", type=float, default=1.0)
    parser.add_argument("--macro-f1-weight", type=float, default=1.0)
    parser.add_argument("--p2-recall-weight", type=float, default=0.50)
    parser.add_argument("--p1-recall-weight", type=float, default=0.10)
    parser.add_argument("--p3-recall-weight", type=float, default=0.10)
    parser.add_argument("--mae-weight", type=float, default=0.20)
    parser.add_argument("--off-by-one-weight", type=float, default=0.10)
    parser.add_argument("--p1-floor", type=float, default=0.60)
    parser.add_argument("--p3-floor", type=float, default=0.72)
    parser.add_argument("--floor-penalty-weight", type=float, default=0.70)
    return parser


def load_features(feature_dir: str):
    X = load_npz(os.path.join(feature_dir, "X_features.npz"))
    y = np.load(os.path.join(feature_dir, "y.npy"))
    meta = pd.read_csv(os.path.join(feature_dir, "feature_meta.csv"))
    return X, y, meta


def parse_str_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def final_objective(row: dict, args: argparse.Namespace) -> float:
    p1_gap = max(0.0, args.p1_floor - float(row["p1_recall"]))
    p3_gap = max(0.0, args.p3_floor - float(row["p3_recall"]))
    return (
        args.accuracy_weight * float(row["accuracy"])
        + args.macro_f1_weight * float(row["macro_f1"])
        + args.p2_recall_weight * float(row["p2_recall"])
        + args.p1_recall_weight * float(row["p1_recall"])
        + args.p3_recall_weight * float(row["p3_recall"])
        + args.off_by_one_weight * float(row["off_by_one_accuracy"])
        - args.mae_weight * float(row["mae"])
        - args.floor_penalty_weight * (p1_gap + p3_gap)
    )


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    view = df[columns].copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: f"{value:.4f}")
    try:
        return view.to_markdown(index=False)
    except ImportError:
        header = "| " + " | ".join(view.columns) + " |"
        divider = "| " + " | ".join(["---"] * len(view.columns)) + " |"
        rows = ["| " + " | ".join(str(row[column]) for column in view.columns) + " |" for _, row in view.iterrows()]
        return "\n".join([header, divider, *rows])


def write_summary(path: str, candidates: pd.DataFrame, best_eval: pd.DataFrame) -> None:
    cand_cols = [
        "stage",
        "base_model_type",
        "base_params",
        "boundary_model_type",
        "boundary_params",
        "boundary_apply",
        "boundary_p2_sample_weight",
        "validation_objective",
        "validation_accuracy",
        "validation_macro_f1",
        "validation_p1_recall",
        "validation_p2_recall",
        "validation_p3_recall",
        "natural_accuracy",
        "natural_macro_f1",
        "natural_p2_recall",
        "natural_mae",
    ]
    eval_cols = [
        "eval_set",
        "accuracy",
        "macro_f1",
        "off_by_one_accuracy",
        "mae",
        "p1_recall",
        "p2_recall",
        "p3_recall",
        "p4_recall",
        "p5_recall",
    ]
    lines = [
        "# Final Objective Grid Search",
        "",
        "This search selects the model by a balanced validation objective:",
        "",
        "`accuracy + macro F1 + 0.5 * P2 recall + P1/P3 guardrails + off-by-one bonus - MAE penalty`",
        "",
        "The objective is used only on the validation set. The reported result is evaluated on natural_holdout.",
        "",
        "## Best Natural-Holdout Result",
        "",
        markdown_table(best_eval, eval_cols),
        "",
        "## Top Validation Candidates",
        "",
        markdown_table(candidates.sort_values("validation_objective", ascending=False).head(12), cand_cols),
        "",
        "## Interpretation",
        "",
        "- If this best result is not higher than the current saved best model, keep the current best model and treat this as a negative tuning result.",
        "- A validation objective can improve robustness, but it cannot guarantee a better natural_holdout score because the holdout remains unseen during selection.",
        "",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main() -> None:
    args = build_parser().parse_args()
    X_train, y_train, _ = load_features(args.train_feature_dir)
    X_val, y_val, _ = load_features(args.validation_feature_dir)
    X_nat, y_nat, _ = load_features(args.natural_feature_dir)

    base_fits = []
    rows = []
    for model_type, params, model in build_candidates(args):
        fit_candidate(model, X_train, y_train)
        y_val_pred = model.predict(X_val).astype(int)
        val_row = metric_row(y_val, y_val_pred, "validation")
        val_score = final_objective(val_row, args)
        y_nat_pred = model.predict(X_nat).astype(int)
        nat_row = metric_row(y_nat, y_nat_pred, "natural_holdout")
        base_fits.append((model_type, params, model, val_row))
        row = {
            "stage": "base_direct",
            "base_model_type": model_type,
            "base_params": params,
            "boundary_model_type": "",
            "boundary_params": "",
            "boundary_apply": "",
            "boundary_p2_sample_weight": 1.0,
            "validation_objective": val_score,
        }
        row.update({f"validation_{key}": value for key, value in val_row.items() if key not in {"eval_set", "rows"}})
        row.update({f"natural_{key}": value for key, value in nat_row.items() if key not in {"eval_set", "rows"}})
        rows.append(row)

    boundary_mask = np.isin(y_train, BOUNDARY_LABELS)
    X_boundary = X_train[boundary_mask]
    y_boundary = y_train[boundary_mask]
    boundary_apply_values = parse_str_values(args.boundary_apply_values)
    boundary_p2_sample_weights = parse_float_list(args.boundary_p2_sample_weights)

    best = None
    best_score = -np.inf
    for base_model_type, base_params, base_model, _base_val_row in base_fits:
        for boundary_apply in boundary_apply_values:
            for p2_sample_weight in boundary_p2_sample_weights:
                sample_weight = np.ones(len(y_boundary), dtype=float)
                sample_weight[y_boundary == 2] = p2_sample_weight
                for boundary_model_type, boundary_params, boundary_model in build_candidates(args):
                    fit_candidate(boundary_model, X_boundary, y_boundary, sample_weight=sample_weight)
                    y_val_refined = predict_refined(base_model, boundary_model, X_val, boundary_apply)
                    val_row = metric_row(y_val, y_val_refined, "validation")
                    val_score = final_objective(val_row, args)
                    y_nat_refined = predict_refined(base_model, boundary_model, X_nat, boundary_apply)
                    nat_row = metric_row(y_nat, y_nat_refined, "natural_holdout")
                    row = {
                        "stage": "boundary_refined",
                        "base_model_type": base_model_type,
                        "base_params": base_params,
                        "boundary_model_type": boundary_model_type,
                        "boundary_params": boundary_params,
                        "boundary_apply": boundary_apply,
                        "boundary_p2_sample_weight": p2_sample_weight,
                        "validation_objective": val_score,
                    }
                    row.update({f"validation_{key}": value for key, value in val_row.items() if key not in {"eval_set", "rows"}})
                    row.update({f"natural_{key}": value for key, value in nat_row.items() if key not in {"eval_set", "rows"}})
                    rows.append(row)
                    if val_score > best_score:
                        best_score = val_score
                        best = {
                            "bundle": {
                                "model_type": "boundary_refined_direct_classifier",
                                "feature_profile": "bm25_p2_error_keywords",
                                "base_model": base_model,
                                "boundary_model": boundary_model,
                                "boundary_apply": boundary_apply,
                                "boundary_p2_sample_weight": p2_sample_weight,
                                "final_objective": {
                                    "accuracy_weight": args.accuracy_weight,
                                    "macro_f1_weight": args.macro_f1_weight,
                                    "p2_recall_weight": args.p2_recall_weight,
                                    "p1_recall_weight": args.p1_recall_weight,
                                    "p3_recall_weight": args.p3_recall_weight,
                                    "mae_weight": args.mae_weight,
                                    "off_by_one_weight": args.off_by_one_weight,
                                    "p1_floor": args.p1_floor,
                                    "p3_floor": args.p3_floor,
                                },
                                "labels": LABELS,
                                "feature_dirs": {
                                    "train": args.train_feature_dir,
                                    "validation": args.validation_feature_dir,
                                    "natural_holdout": args.natural_feature_dir,
                                },
                                "base_settings": {
                                    "candidate_model_type": base_model_type,
                                    "candidate_params": base_params,
                                },
                                "boundary_settings": {
                                    "boundary_model_type": boundary_model_type,
                                    "boundary_params": boundary_params,
                                    "selection_score": val_score,
                                },
                            },
                            "y_nat_pred": y_nat_refined,
                            "nat_row": nat_row,
                        }

    candidates = pd.DataFrame(rows).sort_values("validation_objective", ascending=False)
    os.makedirs(os.path.dirname(args.candidate_csv), exist_ok=True)
    candidates.to_csv(args.candidate_csv, index=False, encoding="utf-8-sig")
    if best is None:
        raise RuntimeError("No candidate was selected.")

    eval_row = dict(best["nat_row"])
    eval_row["model_stage"] = "final_objective_best"
    eval_df = pd.DataFrame([eval_row])
    eval_df.to_csv(args.eval_csv, index=False, encoding="utf-8-sig")
    class_df = pd.DataFrame(class_report_rows(y_nat, best["y_nat_pred"], "natural_holdout"))
    class_df.to_csv(args.class_report_csv, index=False, encoding="utf-8-sig")
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    joblib.dump(best["bundle"], args.model_path)
    write_summary(args.summary_md, candidates, eval_df)

    print("=== Final Objective Grid Search ===")
    print(candidates.head(8)[[
        "stage",
        "base_model_type",
        "base_params",
        "boundary_model_type",
        "boundary_params",
        "boundary_apply",
        "boundary_p2_sample_weight",
        "validation_objective",
        "validation_accuracy",
        "validation_macro_f1",
        "validation_p2_recall",
        "natural_accuracy",
        "natural_macro_f1",
        "natural_p2_recall",
    ]].to_string(index=False))
    print("\nBest natural_holdout:")
    print(eval_df[[
        "accuracy",
        "macro_f1",
        "off_by_one_accuracy",
        "mae",
        "p1_recall",
        "p2_recall",
        "p3_recall",
        "p4_recall",
        "p5_recall",
    ]].to_string(index=False))
    print("\nConfusion matrix labels P1..P5:")
    print(confusion_matrix(y_nat, best["y_nat_pred"], labels=LABELS))
    print(f"saved candidates -> {args.candidate_csv}")
    print(f"saved eval -> {args.eval_csv}")
    print(f"saved report -> {args.summary_md}")


if __name__ == "__main__":
    main()
