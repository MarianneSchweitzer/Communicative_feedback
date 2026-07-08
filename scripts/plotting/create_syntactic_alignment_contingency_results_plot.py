#!/usr/bin/env python3
"""
Create plots and statistical tests for syntactic caregiver alignment
by child utterance grammaticality.

This is the syntactic-only version split from the combined semantic/syntactic
alignment analysis script.

It does two complementary things:

1. Transcript-level stats used for the plot:
   - Computes mean syntactic alignment per transcript and grammaticality category.
   - Compares grammatical vs ungrammatical means within the same transcripts.

2. Raw-row mixed-effects model:
   - Fits alignment ~ grammar_label + (1 | transcript_file).
   - Uses all individual utterance pairs while accounting for transcript/session.

Expected grammaticality coding:
    -1 = ungrammatical
     0 = ambiguous
     1 = grammatical
"""

import argparse
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import seaborn as sns

from scipy.stats import ttest_rel, wilcoxon
from pymer4.models import Lmer


GRAMMAR_LABELS = {
    -1: "ungrammatical",
     0: "ambiguous",
     1: "grammatical",
}

DEFAULT_GRAMMAR_ORDER = ["grammatical", "ungrammatical"]
AMBIGUOUS_GRAMMAR_ORDER = ["grammatical", "ambiguous", "ungrammatical"]

DEFAULT_ALIGNMENT_COLUMN = "align_syntactic_mor_bigram"
DEFAULT_ALIGNMENT_TYPE = "syntactic"


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------


def validate_columns(data: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        raise ValueError(
            f"\nMissing required columns: {missing_columns}\n"
            f"Available columns are: {data.columns.tolist()}\n"
        )


def ensure_transcript_file(data: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure that transcript_file exists and is usable as a random-effect grouping factor.

    If transcript_file is missing or entirely empty, reconstruct it from subject + session.
    """
    data = data.copy()

    if "transcript_file" not in data.columns or data["transcript_file"].isna().all():
        if {"subject", "session"}.issubset(data.columns):
            print(
                "\ntranscript_file is missing or entirely empty. "
                "Creating transcript_file from subject + session."
            )
            data["transcript_file"] = (
                "subj_" + data["subject"].astype(str)
                + "_sess_" + data["session"].astype(str)
            )
        else:
            raise ValueError(
                "transcript_file is missing/empty, and subject/session are not available "
                "to reconstruct it."
            )

    data["transcript_file"] = data["transcript_file"].astype(str)
    return data


def standardize_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Rename common alternative column names to the expected names."""
    return data.rename(columns={
        "grammar": "is_grammatical",
        "grammaticality": "is_grammatical",
        "is_child_grammatical": "is_grammatical",
        "file": "transcript_file",
        "transcript": "transcript_file",
        "semantic_alignment": "align_semantic",
        "syntactic_alignment": "align_syntactic",
    })


def add_grammar_label(data: pd.DataFrame) -> pd.DataFrame:
    """Create a human-readable categorical grammaticality column."""
    data = data.copy()
    data = data.dropna(subset=["is_grammatical"]).copy()
    data["is_grammatical"] = data["is_grammatical"].astype(int)

    allowed_grammar_values = set(GRAMMAR_LABELS.keys())
    observed_grammar_values = set(data["is_grammatical"].unique())

    if not observed_grammar_values.issubset(allowed_grammar_values):
        raise ValueError(
            f"Unexpected values in is_grammatical: {observed_grammar_values}\n"
            "Expected only -1, 0, and 1."
        )

    data["grammar_label"] = data["is_grammatical"].map(GRAMMAR_LABELS)
    return data


def p_to_stars(p_value: float) -> str:
    if pd.isna(p_value):
        return "n/a"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def print_basic_diagnostics(data: pd.DataFrame, alignment_column: str) -> None:
    print("\nColumns found:")
    print(data.columns.tolist())

    diagnostic_cols = ["is_grammatical", "grammar_label", "transcript_file", alignment_column]

    print("\nMissing values:")
    print(data[diagnostic_cols].isna().sum())

    print("\nGrammaticality counts, utterance-pair level:")
    print(data["grammar_label"].value_counts(dropna=False))

    print("\nTranscript count:")
    print(data["transcript_file"].nunique())

    print(f"\nSummary for {alignment_column}:")
    print(data[alignment_column].describe())


# ---------------------------------------------------------------------------
# Transcript-level data, plot summaries, and paired tests
# ---------------------------------------------------------------------------


def build_transcript_level_alignment(
    data: pd.DataFrame,
    alignment_column: str,
    alignment_type: str,
    min_rows_per_transcript: int,
) -> pd.DataFrame:
    """
    Compute transcript-level mean alignment by grammaticality.

    This mirrors the plotting logic: instead of plotting raw rows directly, we first
    compute per-transcript means, then average those transcript-level values.
    """
    entries = []

    for transcript_file in tqdm(
        data["transcript_file"].unique(),
        desc="Building transcript-level means",
    ):
        data_transcript = data[data["transcript_file"] == transcript_file]

        if len(data_transcript) < min_rows_per_transcript:
            continue

        for grammar_value, grammar_label in GRAMMAR_LABELS.items():
            data_grammar = data_transcript[data_transcript["is_grammatical"] == grammar_value]

            if data_grammar.empty:
                continue

            mean_alignment = data_grammar[alignment_column].mean(skipna=True)

            if pd.isna(mean_alignment):
                continue

            entries.append({
                "transcript_file": transcript_file,
                "Grammaticality": grammar_label,
                "alignment_column": alignment_column,
                "alignment_type": alignment_type,
                "mean_alignment": mean_alignment,
                "n_rows": len(data_grammar),
            })

    transcript_df = pd.DataFrame(entries)

    if transcript_df.empty:
        raise ValueError(
            "No transcript-level entries were created. "
            "This may mean that no transcript has enough rows, "
            "or that alignment values are missing."
        )

    return transcript_df


def summarize_transcript_level_alignment(transcript_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        transcript_df
        .groupby(["alignment_type", "Grammaticality"], as_index=False)
        .agg(
            mean_alignment=("mean_alignment", "mean"),
            sd_alignment=("mean_alignment", "std"),
            sem_alignment=("mean_alignment", lambda x: x.std(ddof=1) / np.sqrt(x.count())),
            n_transcripts=("mean_alignment", "count"),
            total_rows=("n_rows", "sum"),
        )
    )
    return summary


def paired_transcript_tests(
    transcript_df: pd.DataFrame,
    grammar_order: list[str],
) -> pd.DataFrame:
    """
    Run paired transcript-level tests between grammaticality categories.

    Only transcripts that have both grammaticality categories being compared are used.
    """
    results = []
    comparisons = list(combinations(grammar_order, 2))

    alignment_type = transcript_df["alignment_type"].iloc[0]

    wide = transcript_df.pivot_table(
        index="transcript_file",
        columns="Grammaticality",
        values="mean_alignment",
        aggfunc="mean",
    )

    for condition_a, condition_b in comparisons:
        if condition_a not in wide.columns or condition_b not in wide.columns:
            continue

        pair_data = wide[[condition_a, condition_b]].dropna().copy()
        n_pairs = len(pair_data)

        if n_pairs < 2:
            results.append({
                "alignment_type": alignment_type,
                "comparison": f"{condition_a} vs {condition_b}",
                "condition_a": condition_a,
                "condition_b": condition_b,
                "n_paired_transcripts": n_pairs,
                "mean_a": np.nan,
                "mean_b": np.nan,
                "mean_difference_a_minus_b": np.nan,
                "cohens_dz": np.nan,
                "paired_t_statistic": np.nan,
                "paired_t_p": np.nan,
                "paired_t_sig": "n/a",
                "wilcoxon_statistic": np.nan,
                "wilcoxon_p": np.nan,
                "wilcoxon_sig": "n/a",
                "note": "Not enough paired transcripts for a test.",
            })
            continue

        a = pair_data[condition_a].astype(float)
        b = pair_data[condition_b].astype(float)
        diff = a - b

        t_stat, t_p = ttest_rel(a, b, nan_policy="omit")

        if np.allclose(diff, 0):
            w_stat, w_p = np.nan, 1.0
            note = "All paired differences are zero; Wilcoxon p set to 1.0."
        else:
            w_stat, w_p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            note = ""

        diff_sd = diff.std(ddof=1)
        cohens_dz = diff.mean() / diff_sd if diff_sd > 0 else np.nan

        results.append({
            "alignment_type": alignment_type,
            "comparison": f"{condition_a} vs {condition_b}",
            "condition_a": condition_a,
            "condition_b": condition_b,
            "n_paired_transcripts": n_pairs,
            "mean_a": a.mean(),
            "mean_b": b.mean(),
            "mean_difference_a_minus_b": diff.mean(),
            "cohens_dz": cohens_dz,
            "paired_t_statistic": t_stat,
            "paired_t_p": t_p,
            "paired_t_sig": p_to_stars(t_p),
            "wilcoxon_statistic": w_stat,
            "wilcoxon_p": w_p,
            "wilcoxon_sig": p_to_stars(w_p),
            "note": note,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Mixed-effects models on raw rows
# ---------------------------------------------------------------------------


def fit_alignment_model(
    data: pd.DataFrame,
    alignment_column: str,
    alignment_type: str,
    grammar_order: list[str],
    output_dir: Path,
) -> pd.DataFrame:
    """
    Fit a mixed-effects linear regression on raw utterance-pair rows:

        alignment ~ grammar_label + (1 | transcript_file)

    The reference category is grammatical.
    """
    model_data = data.dropna(
        subset=[alignment_column, "grammar_label", "transcript_file"]
    ).copy()

    model_data = model_data[model_data["grammar_label"].isin(grammar_order)].copy()

    if model_data.empty:
        raise ValueError(
            f"No usable rows for {alignment_column} after dropping missing values."
        )

    model_data["grammar_label"] = pd.Categorical(
        model_data["grammar_label"],
        categories=grammar_order,
        ordered=False,
    )
    model_data["transcript_file"] = model_data["transcript_file"].astype(str)

    print("\n" + "=" * 80)
    print(f"RAW-ROW MIXED-EFFECTS MODEL FOR {alignment_type} alignment")
    print("=" * 80)
    print(f"Rows used: {len(model_data):,}")
    print(f"Transcripts used: {model_data['transcript_file'].nunique():,}")
    print("Reference grammaticality category: grammatical")

    formula = f"{alignment_column} ~ grammar_label + (1 | transcript_file)"
    mod = Lmer(formula, data=model_data)
    fitted = mod.fit()

    print(fitted)

    fitted = fitted.reset_index().rename(columns={"index": "term"})
    fitted.insert(0, "alignment_type", alignment_type)
    fitted.insert(1, "model", formula)

    model_output_path = output_dir / f"mixed_effects_{alignment_type}_alignment.csv"
    fitted.to_csv(model_output_path, index=False)
    print(f"\nSaved mixed-effects coefficients to: {model_output_path}")

    cols_to_show = [
        col for col in ["term", "Estimate", "SE", "T-stat", "P-val", "Sig"]
        if col in fitted.columns
    ]
    if cols_to_show:
        print("\nKey coefficients:")
        print(fitted[cols_to_show])

    return fitted


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def create_alignment_plot(
    transcript_df: pd.DataFrame,
    paired_tests_df: pd.DataFrame,
    output_dir: Path,
    grammar_order: list[str],
    alignment_type: str,
) -> None:
    plot_data = transcript_df[transcript_df["Grammaticality"].isin(grammar_order)].copy()

    sns.set_theme(font_scale=1.2)
    sns.set_style("ticks")

    plt.figure(figsize=(7, 5.5))

    ax = sns.barplot(
        data=plot_data,
        x="Grammaticality",
        y="mean_alignment",
        order=grammar_order,
        errorbar="ci",
    )

    plt.ylabel(f"Mean caregiver {alignment_type} alignment")
    plt.xlabel("")
    plt.ylim((0, 1))
    plt.title(f"{alignment_type.capitalize()} alignment by grammaticality")

    if {"grammatical", "ungrammatical"}.issubset(set(grammar_order)) and not paired_tests_df.empty:
        main_tests = paired_tests_df[
            paired_tests_df["comparison"].isin([
                "grammatical vs ungrammatical",
                "ungrammatical vs grammatical",
            ])
        ].copy()

        if not main_tests.empty:
            row = main_tests.iloc[0]
            plt.figtext(
                0.5,
                -0.02,
                f"Transcript-level paired test, grammatical vs ungrammatical — "
                f"paired t p={row['paired_t_p']:.3g} ({row['paired_t_sig']})",
                ha="center",
                fontsize=9,
            )

    plt.tight_layout()
    sns.despine()

    output_path = output_dir / f"{alignment_type}_alignment_by_grammaticality.png"
    ax.get_figure().savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved plot to: {output_path}")


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------


def create_results(args) -> None:
    alignment_column = args.alignment_column
    alignment_type = args.alignment_type

    print(f"\nStats for {alignment_type} alignment:")

    data = pd.read_csv(args.data_path)

    print("\nLoaded file:")
    print(args.data_path)

    data = standardize_columns(data)

    required_columns = ["is_grammatical", alignment_column]
    validate_columns(data, required_columns)
    data = ensure_transcript_file(data)
    data = add_grammar_label(data)

    if args.include_ambiguous:
        grammar_order = AMBIGUOUS_GRAMMAR_ORDER
    else:
        grammar_order = DEFAULT_GRAMMAR_ORDER
        data = data[data["grammar_label"].isin(grammar_order)].copy()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print_basic_diagnostics(data, alignment_column)

    if not args.skip_models:
        fit_alignment_model(
            data=data,
            alignment_column=alignment_column,
            alignment_type=alignment_type,
            grammar_order=grammar_order,
            output_dir=output_dir,
        )

    transcript_df = build_transcript_level_alignment(
        data=data,
        alignment_column=alignment_column,
        alignment_type=alignment_type,
        min_rows_per_transcript=args.min_rows_per_transcript,
    )

    transcript_summary_path = output_dir / f"{alignment_type}_alignment_by_grammaticality_transcript_level.csv"
    transcript_df.to_csv(transcript_summary_path, index=False)
    print(f"\nSaved transcript-level data to: {transcript_summary_path}")

    summary_df = summarize_transcript_level_alignment(transcript_df)
    summary_path = output_dir / f"{alignment_type}_alignment_by_grammaticality_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\nMean transcript-level {alignment_type} alignment by grammaticality:")
    print(summary_df)
    print(f"\nSaved transcript-level summary to: {summary_path}")

    paired_tests_df = paired_transcript_tests(
        transcript_df=transcript_df,
        grammar_order=grammar_order,
    )

    paired_tests_path = output_dir / f"{alignment_type}_alignment_by_grammaticality_paired_tests.csv"
    paired_tests_df.to_csv(paired_tests_path, index=False)

    print("\nTranscript-level paired statistical tests:")
    print(paired_tests_df)
    print(f"\nSaved paired tests to: {paired_tests_path}")

    create_alignment_plot(
        transcript_df=transcript_df,
        paired_tests_df=paired_tests_df,
        output_dir=output_dir,
        grammar_order=grammar_order,
        alignment_type=alignment_type,
    )

    print("\nHow to read the main stats:")
    print(
        f"- Use {alignment_type}_alignment_by_grammaticality_paired_tests.csv "
        "for the stats that match the plot.\n"
        "- paired_t_p is the paired t-test p-value.\n"
        "- wilcoxon_p is the non-parametric paired test p-value.\n"
        "- cohens_dz is the paired-effect size: larger absolute values mean larger "
        "within-transcript differences.\n"
        f"- Use mixed_effects_{alignment_type}_alignment.csv as the raw-row mixed-effects model check."
    )


def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze syntactic caregiver alignment by child utterance "
            "grammaticality, including plots and statistical tests."
        )
    )

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to annotated conversation-pair CSV.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results_syntactic_alignment",
        help="Directory where plots, summaries, and stats will be saved.",
    )

    parser.add_argument(
        "--alignment_column",
        type=str,
        default=DEFAULT_ALIGNMENT_COLUMN,
        help="Alignment column to analyze.",
    )

    parser.add_argument(
        "--alignment_type",
        type=str,
        default=DEFAULT_ALIGNMENT_TYPE,
        help="Human-readable alignment name used in outputs.",
    )

    parser.add_argument(
        "--min_rows_per_transcript",
        type=int,
        default=100,
        help=(
            "Minimum number of rows required for a transcript/session to be included. "
            "This affects the plot and transcript-level paired tests."
        ),
    )

    parser.add_argument(
        "--include_ambiguous",
        action="store_true",
        help="Include ambiguous utterances, is_grammatical == 0, in plots and tests.",
    )

    parser.add_argument(
        "--skip_models",
        action="store_true",
        help="Skip raw-row mixed-effects models and only run transcript-level plot/tests.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    create_results(args)
