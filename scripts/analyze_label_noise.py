import argparse
import math
import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRIORITIES = ["P1", "P2", "P3", "P4", "P5"]


def project_path(relative_path: str) -> str:
    return os.path.join(PROJECT_ROOT, relative_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze potential label noise and ambiguous priority groups.",
    )
    parser.add_argument(
        "--input",
        default=project_path("data/processed/eclipse_bug_reports_clean_with_dupe.csv"),
        help="Clean CSV with priority labels and dupe_of.",
    )
    parser.add_argument(
        "--feature-meta",
        default=project_path("data/processed/features_bm25_p2_error_keywords_natural_test/feature_meta.csv"),
        help="Natural-holdout feature_meta.csv for keyword ambiguity analysis.",
    )
    parser.add_argument(
        "--ambiguous-groups-csv",
        default=project_path("reports/label_noise_ambiguous_groups.csv"),
        help="Output CSV for ambiguous groups.",
    )
    parser.add_argument(
        "--duplicate-disagreement-csv",
        default=project_path("reports/label_noise_duplicate_disagreements.csv"),
        help="Output CSV for duplicate groups with mixed priorities.",
    )
    parser.add_argument(
        "--summary-md",
        default=project_path("reports/label_noise_analysis.md"),
        help="Markdown summary output.",
    )
    parser.add_argument("--min-group-size", type=int, default=20)
    parser.add_argument("--dominant-share-threshold", type=float, default=0.60)
    return parser


def priority_counts(group: pd.DataFrame) -> dict:
    counts = group["priority"].value_counts().reindex(PRIORITIES, fill_value=0)
    total = int(counts.sum())
    dominant_priority = str(counts.idxmax()) if total else ""
    dominant_share = float(counts.max() / total) if total else 0.0
    probs = counts[counts > 0] / total if total else counts
    entropy = float(-(probs * np.log2(probs)).sum()) if total else 0.0
    normalized_entropy = entropy / math.log2(len(PRIORITIES)) if total else 0.0
    row = {
        "rows": total,
        "dominant_priority": dominant_priority,
        "dominant_share": dominant_share,
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "unique_priorities": int((counts > 0).sum()),
    }
    row.update({priority: int(counts[priority]) for priority in PRIORITIES})
    return row


def summarize_groups(df: pd.DataFrame, group_cols: list[str], min_group_size: int, dominant_share_threshold: float) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        stats = priority_counts(group)
        if stats["rows"] < min_group_size:
            continue
        if stats["dominant_share"] > dominant_share_threshold:
            continue
        row = {column: value for column, value in zip(group_cols, keys)}
        row["grouping"] = "+".join(group_cols)
        row.update(stats)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["normalized_entropy", "rows"], ascending=False)


def duplicate_disagreements(df: pd.DataFrame) -> pd.DataFrame:
    if "dupe_of" not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    work["dupe_of_clean"] = pd.to_numeric(work["dupe_of"], errors="coerce")
    work = work[work["dupe_of_clean"].notna()]
    rows = []
    for dupe_of, group in work.groupby("dupe_of_clean"):
        stats = priority_counts(group)
        if stats["rows"] < 2 or stats["unique_priorities"] < 2:
            continue
        row = {
            "dupe_of": int(dupe_of),
            "bug_ids": ",".join(group["id"].astype(str).head(12)),
            "products": ",".join(sorted(group["product"].fillna("unknown").astype(str).unique())[:8]),
            "components": ",".join(sorted(group["component"].fillna("unknown").astype(str).unique())[:8]),
        }
        row.update(stats)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["unique_priorities", "normalized_entropy", "rows"], ascending=False)


def bool_col(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return pd.to_numeric(df[column], errors="coerce").fillna(0) > 0


def keyword_signature_groups(meta: pd.DataFrame, min_group_size: int, dominant_share_threshold: float) -> pd.DataFrame:
    if "priority" not in meta.columns:
        return pd.DataFrame()
    work = meta.copy()
    work["severity_group"] = work.get("severity", "unknown").fillna("unknown").astype(str).str.lower()
    work["impact_keyword_group"] = np.select(
        [
            bool_col(work, "has_crash_terms") | bool_col(work, "summary_has_crash_terms"),
            bool_col(work, "has_stack_trace") | bool_col(work, "summary_has_stack_trace"),
            bool_col(work, "has_exception_terms") | bool_col(work, "summary_has_exception_terms"),
            bool_col(work, "has_regression_terms") | bool_col(work, "summary_has_regression_terms"),
            bool_col(work, "has_blocking_terms") | bool_col(work, "summary_has_blocking_terms"),
            bool_col(work, "has_data_loss_terms") | bool_col(work, "summary_has_data_loss_terms"),
        ],
        ["crash", "stack_trace", "exception", "regression", "blocking", "data_loss"],
        default="no_clear_impact_keyword",
    )
    return summarize_groups(
        work,
        ["severity_group", "impact_keyword_group"],
        min_group_size=min_group_size,
        dominant_share_threshold=dominant_share_threshold,
    )


def severity_priority_mismatch(df: pd.DataFrame) -> pd.DataFrame:
    severity = df["severity"].fillna("unknown").astype(str).str.lower()
    high_priority = df["priority"].isin(["P1", "P2"])
    low_priority = df["priority"].isin(["P4", "P5"])
    conditions = {
        "normal_or_lower_but_high_priority": severity.isin(["normal", "minor", "trivial", "enhancement"]) & high_priority,
        "major_critical_blocker_but_low_priority": severity.isin(["major", "critical", "blocker"]) & low_priority,
        "enhancement_but_not_low_priority": severity.eq("enhancement") & ~low_priority,
    }
    rows = []
    for name, mask in conditions.items():
        subset = df[mask]
        stats = priority_counts(subset) if len(subset) else {
            "rows": 0,
            "dominant_priority": "",
            "dominant_share": 0.0,
            "entropy": 0.0,
            "normalized_entropy": 0.0,
            "unique_priorities": 0,
            **{priority: 0 for priority in PRIORITIES},
        }
        stats["case"] = name
        rows.append(stats)
    return pd.DataFrame(rows)[["case", "rows", "dominant_priority", "dominant_share", *PRIORITIES]]


def markdown_table(df: pd.DataFrame, limit: int | None = None) -> str:
    view = df.copy()
    if limit is not None:
        view = view.head(limit)
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: f"{value:.4f}")
    if view.empty:
        return "_No rows._"
    try:
        return view.to_markdown(index=False)
    except ImportError:
        header = "| " + " | ".join(view.columns) + " |"
        divider = "| " + " | ".join(["---"] * len(view.columns)) + " |"
        rows = ["| " + " | ".join(str(row[column]) for column in view.columns) + " |" for _, row in view.iterrows()]
        return "\n".join([header, divider, *rows])


def write_summary(
    path: str,
    df: pd.DataFrame,
    ambiguous: pd.DataFrame,
    duplicates: pd.DataFrame,
    mismatches: pd.DataFrame,
) -> None:
    overall = priority_counts(df)
    lines = [
        "# Label Noise / Ambiguous Priority Analysis",
        "",
        "This report looks for dataset patterns where the same metadata context maps to mixed priorities.",
        "",
        "## Overall Priority Distribution",
        "",
        markdown_table(pd.DataFrame([overall])),
        "",
        "## Severity / Priority Mismatch Signals",
        "",
        markdown_table(mismatches),
        "",
        "## Top Ambiguous Groups",
        "",
        markdown_table(ambiguous, 20),
        "",
        "## Duplicate-Link Priority Disagreements",
        "",
        markdown_table(duplicates, 20),
        "",
        "## Interpretation",
        "",
        "- Ambiguous groups mean the same severity/product/component or keyword context has multiple priority labels, so some prediction errors may be label ambiguity rather than pure model failure.",
        "- Duplicate-link disagreements are especially important because duplicate reports should be semantically related; mixed priorities inside duplicate groups suggest historical triage inconsistency.",
        "- For reporting, this supports explaining why P2 is hard: P2 sits between P1 and P3, and many P2-like contexts are not labeled consistently.",
        "",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main() -> None:
    args = build_parser().parse_args()
    df = pd.read_csv(args.input)
    meta = pd.read_csv(args.feature_meta) if os.path.exists(args.feature_meta) else pd.DataFrame()

    group_frames = [
        summarize_groups(df, ["severity"], args.min_group_size, args.dominant_share_threshold),
        summarize_groups(df, ["product"], args.min_group_size, args.dominant_share_threshold),
        summarize_groups(df, ["component"], args.min_group_size, args.dominant_share_threshold),
        summarize_groups(df, ["product", "component"], args.min_group_size, args.dominant_share_threshold),
        summarize_groups(df, ["severity", "product"], args.min_group_size, args.dominant_share_threshold),
    ]
    if not meta.empty:
        group_frames.append(keyword_signature_groups(meta, max(8, args.min_group_size // 2), args.dominant_share_threshold))
    ambiguous = pd.concat([frame for frame in group_frames if not frame.empty], ignore_index=True)
    if not ambiguous.empty:
        ambiguous = ambiguous.sort_values(["normalized_entropy", "rows"], ascending=False)

    duplicates = duplicate_disagreements(df)
    mismatches = severity_priority_mismatch(df)

    os.makedirs(os.path.dirname(args.ambiguous_groups_csv), exist_ok=True)
    ambiguous.to_csv(args.ambiguous_groups_csv, index=False, encoding="utf-8-sig")
    duplicates.to_csv(args.duplicate_disagreement_csv, index=False, encoding="utf-8-sig")
    write_summary(args.summary_md, df, ambiguous, duplicates, mismatches)

    print("=== Label Noise / Ambiguous Priority Analysis ===")
    print(f"rows analyzed: {len(df)}")
    print(f"ambiguous groups: {len(ambiguous)}")
    print(f"duplicate disagreement groups: {len(duplicates)}")
    print("\nSeverity mismatch signals:")
    print(mismatches.to_string(index=False))
    print(f"saved ambiguous groups -> {args.ambiguous_groups_csv}")
    print(f"saved duplicate disagreements -> {args.duplicate_disagreement_csv}")
    print(f"saved summary -> {args.summary_md}")


if __name__ == "__main__":
    main()
