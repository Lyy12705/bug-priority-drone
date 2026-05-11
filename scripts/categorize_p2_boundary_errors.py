import argparse
import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def project_path(relative_path: str) -> str:
    return os.path.join(PROJECT_ROOT, relative_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manually categorize true-P2 errors predicted as P1/P3."
    )
    parser.add_argument(
        "--error-csv",
        default=project_path("reports/p2_error_analysis_boundary_refined_improved_natural.csv"),
        help="CSV from analyze_priority_errors.py.",
    )
    parser.add_argument(
        "--feature-meta",
        default=project_path("data/processed/features_improved_dupe_natural_test/feature_meta.csv"),
        help="feature_meta.csv containing keyword/error-driven features.",
    )
    parser.add_argument(
        "--output-csv",
        default=project_path("reports/p2_error_manual_categories_boundary_refined.csv"),
        help="Output CSV with manual category flags.",
    )
    parser.add_argument(
        "--summary-md",
        default=project_path("reports/p2_error_manual_categories_boundary_refined.md"),
        help="Markdown summary output.",
    )
    parser.add_argument(
        "--model-name",
        default="boundary_refined_improved",
        help="Name shown in the Markdown report.",
    )
    return parser


def bool_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].fillna(0).astype(float) > 0


def norm_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()


def assign_primary_category(row: pd.Series) -> str:
    if row["error_direction"] == "P2 -> P1 判太嚴重":
        if row["has_stack_or_exception"]:
            return "P2->P1：stack/exception 訊號讓模型判太嚴重"
        if row["has_high_impact_terms"]:
            return "P2->P1：crash/block/data-loss 等高影響詞讓模型判太嚴重"
        if row["component_focus_ui_debug_core"]:
            return "P2->P1：ui/debug/core 模組歷史訊號偏高"
        return "P2->P1：一般判太嚴重"

    if row["error_direction"] == "P2 -> P3 判太輕":
        if row["severity_normal_or_lower"]:
            return "P2->P3：severity normal/enhancement/minor 讓模型判太輕"
        if row["related_pull_to_p3"]:
            return "P2->P3：related reports 偏向 P3"
        if not row["has_high_impact_terms"] and not row["has_stack_or_exception"]:
            return "P2->P3：缺少明顯 critical/high-impact 訊號"
        return "P2->P3：一般判太輕"

    return "其他"


def add_manual_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    predicted = norm_text(df["predicted_priority_name"])
    severity = norm_text(df["severity"])
    product = norm_text(df["product"])
    component = norm_text(df["component"])

    df["error_direction"] = np.where(
        predicted.str.contains("p1"),
        "P2 -> P1 判太嚴重",
        np.where(predicted.str.contains("p3"), "P2 -> P3 判太輕", "其他"),
    )
    df["predicted_too_high_p1"] = df["error_direction"].eq("P2 -> P1 判太嚴重")
    df["predicted_too_low_p3"] = df["error_direction"].eq("P2 -> P3 判太輕")

    df["severity_is_normal"] = severity.eq("normal")
    df["severity_is_major"] = severity.eq("major")
    df["severity_is_enhancement"] = severity.eq("enhancement")
    df["severity_normal_or_lower"] = severity.isin(["normal", "minor", "trivial", "enhancement"])

    df["product_is_platform"] = product.eq("platform")
    df["product_is_jdt"] = product.eq("jdt")
    df["product_focus_platform_or_jdt"] = product.isin(["platform", "jdt"])

    df["component_is_ui"] = component.eq("ui")
    df["component_is_debug"] = component.eq("debug")
    df["component_is_core"] = component.eq("core")
    df["component_focus_ui_debug_core"] = component.isin(["ui", "debug", "core"])

    df["has_stack_or_exception"] = (
        bool_col(df, "has_stack_trace")
        | bool_col(df, "summary_has_stack_trace")
        | bool_col(df, "has_exception_terms")
        | bool_col(df, "summary_has_exception_terms")
    )
    df["has_high_impact_terms"] = (
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
    df["has_install_update_terms_any"] = bool_col(df, "has_install_update_terms") | bool_col(df, "summary_has_install_update_terms")
    df["has_regression_terms_any"] = bool_col(df, "has_regression_terms") | bool_col(df, "summary_has_regression_terms")
    df["has_api_break_terms_any"] = bool_col(df, "has_api_break_terms") | bool_col(df, "summary_has_api_break_terms")

    related_top1 = pd.to_numeric(df.get("related_top1_priority", 0), errors="coerce").fillna(0)
    related_top3 = pd.to_numeric(df.get("related_top3_avg_priority", 0), errors="coerce").fillna(0)
    df["related_pull_to_p1_p2"] = (related_top1 <= 2) | (related_top3 <= 2.25)
    df["related_pull_to_p3"] = (related_top1 >= 3) | (related_top3 >= 2.75)

    df["manual_primary_category"] = df.apply(assign_primary_category, axis=1)
    return df


def count_table(df: pd.DataFrame, col: str, label: str) -> pd.DataFrame:
    counts = df[col].value_counts(dropna=False).rename_axis(label).reset_index(name="count")
    counts["percent"] = counts["count"] / len(df) if len(df) else 0.0
    return counts


def bool_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        count = int(df[col].fillna(False).astype(bool).sum())
        rows.append({"category": col, "count": count, "percent": count / len(df) if len(df) else 0.0})
    return pd.DataFrame(rows).sort_values(["count", "category"], ascending=[False, True])


def crosstab(df: pd.DataFrame, index: str, columns: str) -> pd.DataFrame:
    table = pd.crosstab(df[index], df[columns])
    table["total"] = table.sum(axis=1)
    return table.reset_index()


def markdown_table(df: pd.DataFrame) -> str:
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda value: f"{value:.4f}")
    try:
        return view.to_markdown(index=False)
    except ImportError:
        header = "| " + " | ".join(view.columns) + " |"
        divider = "| " + " | ".join(["---"] * len(view.columns)) + " |"
        rows = ["| " + " | ".join(str(row[col]) for col in view.columns) + " |" for _, row in view.iterrows()]
        return "\n".join([header, divider, *rows])


def write_summary(path: str, df: pd.DataFrame, model_name: str) -> None:
    flag_cols = [
        "predicted_too_high_p1",
        "predicted_too_low_p3",
        "severity_is_normal",
        "has_stack_or_exception",
        "product_is_platform",
        "product_is_jdt",
        "product_focus_platform_or_jdt",
        "component_is_ui",
        "component_is_debug",
        "component_is_core",
        "component_focus_ui_debug_core",
        "has_high_impact_terms",
        "has_install_update_terms_any",
        "has_regression_terms_any",
        "has_api_break_terms_any",
        "related_pull_to_p1_p2",
        "related_pull_to_p3",
    ]

    lines = [
        "# P2 Boundary Error Manual Categories",
        "",
        f"- Model: `{model_name}`",
        f"- Selected errors: `{len(df)}`",
        "- Scope: true P2 predicted as P1 or P3",
        "",
        "## Error Direction",
        "",
        markdown_table(count_table(df, "error_direction", "direction")),
        "",
        "## Manual Primary Categories",
        "",
        markdown_table(count_table(df, "manual_primary_category", "primary_category")),
        "",
        "## Multi-label Category Flags",
        "",
        markdown_table(bool_summary(df, flag_cols)),
        "",
        "## Direction x Severity",
        "",
        markdown_table(crosstab(df, "error_direction", "severity")),
        "",
        "## Direction x Product",
        "",
        markdown_table(crosstab(df, "error_direction", "product")),
        "",
        "## Direction x Component Focus",
        "",
        markdown_table(crosstab(df, "error_direction", "component_focus_ui_debug_core")),
        "",
        "## Direction x Stack/Exception",
        "",
        markdown_table(crosstab(df, "error_direction", "has_stack_or_exception")),
        "",
        "## Direction x Related-report Pull",
        "",
        markdown_table(crosstab(df, "error_direction", "related_pull_to_p3")),
        "",
        "## Interpretation",
        "",
        "- `P2 -> P1 判太嚴重` 表示模型把 critical bug 判成 blocker。",
        "- `P2 -> P3 判太輕` 表示模型把 critical bug 判成 major。",
        "- 多標籤 flags 不是互斥分類；同一筆 bug 可以同時是 `severity_is_normal`、`has_stack_or_exception`、`product_is_jdt`。",
        "- `manual_primary_category` 是為了報告方便而指定的一個主要錯誤原因。",
        "",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main() -> None:
    args = build_parser().parse_args()
    errors = pd.read_csv(args.error_csv)
    meta = pd.read_csv(args.feature_meta)

    merged = errors.merge(
        meta,
        on="id",
        how="left",
        suffixes=("", "_meta"),
    )
    # Prefer human-readable fields from the original error CSV while keeping keyword columns from feature_meta.
    for col in ["summary", "description", "product", "component", "severity", "creator", "creation_time", "status"]:
        meta_col = f"{col}_meta"
        if meta_col in merged.columns:
            merged = merged.drop(columns=[meta_col])

    categorized = add_manual_flags(merged)
    preferred_cols = [
        "id",
        "true_priority_name",
        "predicted_priority_name",
        "error_direction",
        "manual_primary_category",
        "severity",
        "product",
        "component",
        "summary",
        "description_snippet",
        "severity_is_normal",
        "has_stack_or_exception",
        "has_high_impact_terms",
        "product_focus_platform_or_jdt",
        "component_focus_ui_debug_core",
        "related_pull_to_p1_p2",
        "related_pull_to_p3",
        "related_top1_priority",
        "related_top3_avg_priority",
        "related_top5_avg_priority",
    ]
    output_cols = [col for col in preferred_cols if col in categorized.columns]
    output = categorized[output_cols].sort_values(["error_direction", "manual_primary_category", "id"])

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    output.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    write_summary(args.summary_md, categorized, args.model_name)

    print(f"saved categorized CSV -> {args.output_csv}")
    print(f"saved summary -> {args.summary_md}")
    print("\n=== Error Direction ===")
    print(count_table(categorized, "error_direction", "direction").to_string(index=False))
    print("\n=== Manual Primary Categories ===")
    print(count_table(categorized, "manual_primary_category", "primary_category").to_string(index=False))


if __name__ == "__main__":
    main()
