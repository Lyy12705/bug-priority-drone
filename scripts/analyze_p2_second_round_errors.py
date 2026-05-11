import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def project_path(relative_path: str) -> str:
    return os.path.join(PROJECT_ROOT, relative_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Second-round analysis for remaining true-P2 boundary errors.",
    )
    parser.add_argument(
        "--error-csv",
        default=project_path("reports/p2_error_analysis_improved_priority_p2_keywords_natural.csv"),
        help="P2 boundary-error CSV from analyze_priority_errors.py.",
    )
    parser.add_argument(
        "--feature-meta",
        default=project_path("data/processed/features_bm25_p2_error_keywords_natural_test/feature_meta.csv"),
        help="feature_meta.csv for the evaluated natural_holdout feature set.",
    )
    parser.add_argument(
        "--output-csv",
        default=project_path("reports/p2_second_round_error_analysis.csv"),
        help="Detailed row-level output with second-round tags.",
    )
    parser.add_argument(
        "--summary-md",
        default=project_path("reports/p2_second_round_error_analysis.md"),
        help="Markdown summary output.",
    )
    return parser


def bool_col(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return pd.to_numeric(df[column], errors="coerce").fillna(0) > 0


def num_col(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def norm_col(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series("", index=df.index)
    return df[column].fillna("").astype(str).str.strip().str.lower()


def join_text(df: pd.DataFrame) -> pd.Series:
    summary = df.get("summary", pd.Series("", index=df.index)).fillna("").astype(str)
    desc = df.get("description_snippet", pd.Series("", index=df.index)).fillna("").astype(str)
    return summary + " " + desc


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    predicted = norm_col(df, "predicted_priority_name")
    severity = norm_col(df, "severity")
    product = norm_col(df, "product")
    component = norm_col(df, "component")

    df["error_direction"] = np.select(
        [predicted.str.contains("p1"), predicted.str.contains("p3")],
        ["P2_to_P1_too_severe", "P2_to_P3_too_light"],
        default="other",
    )
    df["severity_normal_or_lower"] = severity.isin(["normal", "minor", "trivial", "enhancement"])
    df["severity_major_or_critical"] = severity.isin(["major", "critical", "blocker"])
    df["product_platform_or_jdt"] = product.isin(["platform", "jdt"])
    df["component_ui_debug_core"] = component.isin(["ui", "debug", "core"])

    df["stack_exception_signal"] = (
        bool_col(df, "has_stack_trace")
        | bool_col(df, "summary_has_stack_trace")
        | bool_col(df, "has_exception_terms")
        | bool_col(df, "summary_has_exception_terms")
        | bool_col(df, "has_file_line_stack_terms")
        | bool_col(df, "summary_has_file_line_stack_terms")
    )
    df["high_impact_signal"] = (
        bool_col(df, "has_crash_terms")
        | bool_col(df, "summary_has_crash_terms")
        | bool_col(df, "has_blocking_terms")
        | bool_col(df, "summary_has_blocking_terms")
        | bool_col(df, "has_data_loss_terms")
        | bool_col(df, "summary_has_data_loss_terms")
        | bool_col(df, "has_security_terms")
        | bool_col(df, "summary_has_security_terms")
        | bool_col(df, "has_build_break_terms")
        | bool_col(df, "summary_has_build_break_terms")
    )
    df["regression_signal"] = bool_col(df, "has_regression_terms") | bool_col(df, "summary_has_regression_terms")
    df["debug_signal"] = bool_col(df, "has_debug_terms") | bool_col(df, "summary_has_debug_terms")
    df["ui_signal"] = bool_col(df, "has_ui_block_terms") | bool_col(df, "summary_has_ui_block_terms")
    df["update_install_signal"] = (
        bool_col(df, "has_install_update_terms")
        | bool_col(df, "summary_has_install_update_terms")
        | bool_col(df, "has_update_manager_terms")
        | bool_col(df, "summary_has_update_manager_terms")
    )
    df["weak_text_signal"] = ~(
        df["stack_exception_signal"]
        | df["high_impact_signal"]
        | df["regression_signal"]
        | df["debug_signal"]
        | df["update_install_signal"]
    )

    top1 = num_col(df, "related_top1_priority")
    top3 = num_col(df, "related_top3_avg_priority")
    top3_p1 = num_col(df, "related_top3_p1_rate")
    top3_p2 = num_col(df, "related_top3_p2_rate")
    top3_p3 = num_col(df, "related_top3_p3_rate")
    df["related_pull_to_p1"] = (top1 <= 1.5) | (top3_p1 >= 0.34)
    df["related_pull_to_p2"] = (top3_p2 >= 0.34) | ((top3 >= 1.75) & (top3 <= 2.25))
    df["related_pull_to_p3"] = (top1 >= 3.0) | (top3 >= 2.75) | (top3_p3 >= 0.34)
    return df


def assign_tags(row: pd.Series) -> str:
    tags = []
    if row["error_direction"] == "P2_to_P1_too_severe":
        if row["stack_exception_signal"]:
            tags.append("stack/exception pushed to P1")
        if row["high_impact_signal"]:
            tags.append("high-impact keywords pushed to P1")
        if row["component_ui_debug_core"]:
            tags.append("ui/debug/core historical signal")
        if row["related_pull_to_p1"]:
            tags.append("related reports lean P1")
        if row["severity_normal_or_lower"] and row["stack_exception_signal"]:
            tags.append("normal severity but severe-looking text")
    elif row["error_direction"] == "P2_to_P3_too_light":
        if row["severity_normal_or_lower"]:
            tags.append("normal/lower severity pulled to P3")
        if row["weak_text_signal"]:
            tags.append("missing obvious high-impact text")
        if row["related_pull_to_p3"]:
            tags.append("related reports lean P3")
        if row["update_install_signal"]:
            tags.append("install/update wording is ambiguous")
        if row["product_platform_or_jdt"]:
            tags.append("platform/jdt boundary case")
    if not tags:
        tags.append("general boundary ambiguity")
    return "; ".join(tags)


def count_table(df: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    counts = df[column].fillna("unknown").astype(str).value_counts().rename_axis(label).reset_index(name="count")
    counts["percent"] = counts["count"] / len(df) if len(df) else 0.0
    return counts


def bool_rate_table(error_df: pd.DataFrame, reference_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        if column not in error_df.columns:
            continue
        error_rate = float(error_df[column].fillna(False).astype(bool).mean()) if len(error_df) else 0.0
        ref_rate = float(reference_df[column].fillna(False).astype(bool).mean()) if len(reference_df) else 0.0
        rows.append(
            {
                "signal": column,
                "error_rate": error_rate,
                "all_true_p2_rate": ref_rate,
                "difference": error_rate - ref_rate,
            }
        )
    return pd.DataFrame(rows).sort_values(["difference", "error_rate"], ascending=False)


def top_phrases(df: pd.DataFrame, max_features: int = 20) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["phrase", "count"])
    text = join_text(df)
    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_]{2,}\b",
    )
    matrix = vectorizer.fit_transform(text)
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    names = np.array(vectorizer.get_feature_names_out())
    order = np.argsort(-counts)[:max_features]
    return pd.DataFrame({"phrase": names[order], "count": counts[order].astype(int)})


def markdown_table(df: pd.DataFrame) -> str:
    view = df.copy()
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


def write_summary(path: str, errors: pd.DataFrame, all_true_p2: pd.DataFrame) -> None:
    signal_cols = [
        "severity_normal_or_lower",
        "severity_major_or_critical",
        "stack_exception_signal",
        "high_impact_signal",
        "regression_signal",
        "debug_signal",
        "ui_signal",
        "update_install_signal",
        "weak_text_signal",
        "product_platform_or_jdt",
        "component_ui_debug_core",
        "related_pull_to_p1",
        "related_pull_to_p2",
        "related_pull_to_p3",
    ]
    p2_to_p1 = errors[errors["error_direction"] == "P2_to_P1_too_severe"]
    p2_to_p3 = errors[errors["error_direction"] == "P2_to_P3_too_light"]
    lines = [
        "# P2 Second-Round Error Analysis",
        "",
        "This report inspects the remaining true-P2 rows predicted as P1 or P3 by the current best model.",
        "",
        "## Scope",
        "",
        f"- Remaining P2 boundary errors: `{len(errors)}`",
        f"- All true-P2 rows in natural_holdout: `{len(all_true_p2)}`",
        "",
        "## Error Direction",
        "",
        markdown_table(count_table(errors, "error_direction", "direction")),
        "",
        "## Main Second-Round Tags",
        "",
        markdown_table(count_table(errors, "second_round_tags", "tag_combo").head(12)),
        "",
        "## Signal Rates Compared With All True P2",
        "",
        markdown_table(bool_rate_table(errors, all_true_p2, signal_cols).head(16)),
        "",
        "## P2 -> P1 Top Phrases",
        "",
        markdown_table(top_phrases(p2_to_p1, 15)),
        "",
        "## P2 -> P3 Top Phrases",
        "",
        markdown_table(top_phrases(p2_to_p3, 15)),
        "",
        "## Direction x Severity",
        "",
        markdown_table(pd.crosstab(errors["error_direction"], errors["severity"]).reset_index()),
        "",
        "## Direction x Product",
        "",
        markdown_table(pd.crosstab(errors["error_direction"], errors["product"]).reset_index()),
        "",
        "## Interpretation",
        "",
        "- P2 -> P1 errors are mostly cases where text looks severe, especially stack traces, exceptions, UI/debug/core modules, or related reports leaning to P1.",
        "- P2 -> P3 errors are mostly normal/lower severity rows or rows without clear high-impact keywords, so they look closer to ordinary major bugs.",
        "- This suggests the next improvement should focus on a calibrated P1/P2/P3 boundary layer and better ambiguity handling, not simply adding more global P2 weight.",
        "",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main() -> None:
    args = build_parser().parse_args()
    errors = pd.read_csv(args.error_csv)
    meta = pd.read_csv(args.feature_meta)
    merged = errors.merge(meta, on="id", how="left", suffixes=("", "_meta"))
    for column in ["summary", "description", "product", "component", "severity", "creator", "creation_time", "status"]:
        meta_column = f"{column}_meta"
        if meta_column in merged.columns:
            merged = merged.drop(columns=[meta_column])

    all_true_p2 = meta[meta["priority_num"] == 2].copy() if "priority_num" in meta.columns else meta[meta["priority"] == "P2"].copy()
    all_true_p2 = add_flags(all_true_p2)
    analyzed = add_flags(merged)
    analyzed["second_round_tags"] = analyzed.apply(assign_tags, axis=1)

    preferred = [
        "id",
        "predicted_priority_name",
        "error_direction",
        "second_round_tags",
        "severity",
        "product",
        "component",
        "summary",
        "related_top1_priority",
        "related_top3_avg_priority",
        "related_top3_p1_rate",
        "related_top3_p2_rate",
        "related_top3_p3_rate",
        "stack_exception_signal",
        "high_impact_signal",
        "weak_text_signal",
        "severity_normal_or_lower",
        "product_platform_or_jdt",
        "component_ui_debug_core",
    ]
    output = analyzed[[column for column in preferred if column in analyzed.columns]].copy()
    output = output.sort_values(["error_direction", "second_round_tags", "id"])
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    output.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    write_summary(args.summary_md, analyzed, all_true_p2)

    print("=== P2 Second-Round Error Analysis ===")
    print(f"remaining P2 boundary errors: {len(analyzed)}")
    print(count_table(analyzed, "error_direction", "direction").to_string(index=False))
    print(f"saved csv -> {args.output_csv}")
    print(f"saved summary -> {args.summary_md}")


if __name__ == "__main__":
    main()
