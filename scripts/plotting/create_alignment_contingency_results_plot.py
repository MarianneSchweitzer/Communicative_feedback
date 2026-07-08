#!/usr/bin/env python3

"""
Create plots and statistical tests for semantic/syntactic caregiver alignment
by child utterance grammaticality.

This script does two complementary things:

1. Plot-level / transcript-level stats:
   - First computes mean alignment per transcript and grammaticality category.
   - Then compares grammatical vs ungrammatical means within the same transcripts.
   - This is the statistical test that corresponds most directly to the barplot.

2. Raw-row mixed-effects models:
   - Fits alignment ~ grammaticality_group + (1 | transcript_file).
   - This uses all individual utterance pairs while accounting for transcript/session.

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

ALIGNMENT_LABELS = {
    "align_semantic": "semantic",
    "align_syntactic": "syntactic",
}

DEFAULT_GRAMMAR_ORDER = ["grammatical", "ungrammatical"]
AMBIGUOUS_GRAMMAR_ORDER = ["grammatical", "ambiguous", "ungrammatical"]


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


def benjamini_hochberg(p_values: Iterable[float]) -> list[float]:
    """
    Benjamini-Hochberg FDR correction.

    Implemented directly to avoid depending on statsmodels.
    """
    p_values = np.asarray(list(p_values), dtype=float)
    adjusted = np.full_like(p_values, np.nan, dtype=float)

    valid_mask = ~np.isnan(p_values)
    valid_p = p_values[valid_mask]

    if len(valid_p) == 0:
        return adjusted.tolist()

    order = np.argsort(valid_p)
    ranked_p = valid_p[order]
    n = len(ranked_p)

    ranked_adjusted = ranked_p * n / np.arange(1, n + 1)
    ranked_adjusted = np.minimum.accumulate(ranked_adjusted[::-1])[::-1]
    ranked_adjusted = np.clip(ranked_adjusted, 0, 1)

    corrected_valid = np.empty_like(valid_p)
    corrected_valid[order] = ranked_adjusted
    adjusted[valid_mask] = corrected_valid

    return adjusted.tolist()


def print_basic_diagnostics(data: pd.DataFrame, alignment_columns: list[str]) -> None:
    print("\nColumns found:")
    print(data.columns.tolist())

    diagnostic_cols = ["is_grammatical", "grammar_label", "transcript_file"] + alignment_columns

    print("\nMissing values:")
    print(data[diagnostic_cols].isna().sum())

    print("\nGrammaticality counts, utterance-pair level:")
    print(data["grammar_label"].value_counts(dropna=False))

    print("\nTranscript count:")
    print(data["transcript_file"].nunique())

    for col in alignment_columns:
        print(f"\nSummary for {col}:")
        print(data[col].describe())


# ---------------------------------------------------------------------------
# Transcript-level data, plot summaries, and paired tests
# ---------------------------------------------------------------------------


def build_transcript_level_alignment(
    data: pd.DataFrame,
    alignment_columns: list[str],
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

            for alignment_column in alignment_columns:
                mean_alignment = data_grammar[alignment_column].mean(skipna=True)

                if pd.isna(mean_alignment):
                    continue

                entries.append({
                    "transcript_file": transcript_file,
                    "Grammaticality": grammar_label,
                    "alignment_column": alignment_column,
                    "alignment_type": ALIGNMENT_LABELS.get(alignment_column, alignment_column),
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

    For each alignment type, only transcripts that have both grammaticality categories
    being compared are used. This avoids treating transcript/session means as if they
    were independent when they come from the same transcript.
    """
    results = []
    comparisons = list(combinations(grammar_order, 2))

    for alignment_type in sorted(transcript_df["alignment_type"].unique()):
        sub = transcript_df[transcript_df["alignment_type"] == alignment_type].copy()

        wide = sub.pivot_table(
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
                    "wilcoxon_statistic": np.nan,
                    "wilcoxon_p": np.nan,
                    "note": "Not enough paired transcripts for a test.",
                })
                continue

            a = pair_data[condition_a].astype(float)
            b = pair_data[condition_b].astype(float)
            diff = a - b

            # Paired t-test: tests whether the mean within-transcript difference is non-zero.
            t_stat, t_p = ttest_rel(a, b, nan_policy="omit")

            # Wilcoxon signed-rank test: non-parametric paired test.
            # It fails if all differences are zero, so handle that case explicitly.
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
                "wilcoxon_statistic": w_stat,
                "wilcoxon_p": w_p,
                "note": note,
            })

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        results_df["paired_t_p_fdr"] = benjamini_hochberg(results_df["paired_t_p"])
        results_df["paired_t_sig_fdr"] = results_df["paired_t_p_fdr"].apply(p_to_stars)
        results_df["wilcoxon_p_fdr"] = benjamini_hochberg(results_df["wilcoxon_p"])
        results_df["wilcoxon_sig_fdr"] = results_df["wilcoxon_p_fdr"].apply(p_to_stars)

    return results_df


# ---------------------------------------------------------------------------
# Mixed-effects models on raw rows
# ---------------------------------------------------------------------------


def fit_alignment_model(
    data: pd.DataFrame,
    alignment_column: str,
    grammar_order: list[str],
    output_dir: Path,
) -> pd.DataFrame:
    """
    Fit a mixed-effects linear regression on raw utterance-pair rows:

        alignment ~ grammar_label + (1 | transcript_file)

    The reference category is grammatical, so coefficients tell you whether
    ambiguous/ungrammatical utterances differ from grammatical utterances.
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

    alignment_type = ALIGNMENT_LABELS.get(alignment_column, alignment_column)

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
) -> None:
    plot_data = transcript_df[transcript_df["Grammaticality"].isin(grammar_order)].copy()

    sns.set_theme(font_scale=1.2)
    sns.set_style("ticks")
    sns.set_palette("Set2")

    plt.figure(figsize=(9, 5.5))

    ax = sns.barplot(
        data=plot_data,
        x="Grammaticality",
        y="mean_alignment",
        hue="alignment_type",
        order=grammar_order,
        errorbar="ci",
    )

    plt.ylabel("Mean caregiver alignment")
    plt.xlabel("")
    plt.ylim((0, 1))
    plt.legend(title="Alignment type")

    # Add a compact caption with the main grammatical vs ungrammatical p-values.
    if {"grammatical", "ungrammatical"}.issubset(set(grammar_order)) and not paired_tests_df.empty:
        main_tests = paired_tests_df[
            paired_tests_df["comparison"].isin([
                "grammatical vs ungrammatical",
                "ungrammatical vs grammatical",
            ])
        ].copy()

        if not main_tests.empty:
            caption_parts = []
            for _, row in main_tests.iterrows():
                caption_parts.append(
                    f"{row['alignment_type']}: paired t FDR p={row['paired_t_p_fdr']:.3g} "
                    f"({row['paired_t_sig_fdr']})"
                )
            plt.figtext(
                0.5,
                -0.02,
                "Main transcript-level paired test, grammatical vs ungrammatical — "
                + "; ".join(caption_parts),
                ha="center",
                fontsize=9,
            )

    plt.tight_layout()
    sns.despine()

    output_path = output_dir / "alignment_by_grammaticality.png"
    ax.get_figure().savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved plot to: {output_path}")


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------


def create_results(args) -> None:
    print("\nStats for annotated data:")

    data = pd.read_csv(args.data_path)

    print("\nLoaded file:")
    print(args.data_path)

    data = standardize_columns(data)

    alignment_columns = ["align_semantic", "align_syntactic"]
    required_columns = ["is_grammatical", *alignment_columns]

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

    print_basic_diagnostics(data, alignment_columns)

    # 1. Raw-row mixed-effects models.
    all_model_results = []
    if not args.skip_models:
        for alignment_column in alignment_columns:
            fitted = fit_alignment_model(
                data=data,
                alignment_column=alignment_column,
                grammar_order=grammar_order,
                output_dir=output_dir,
            )
            all_model_results.append(fitted)

        if all_model_results:
            mixed_effects_all = pd.concat(all_model_results, ignore_index=True)
            mixed_effects_all_path = output_dir / "mixed_effects_all_alignment_models.csv"
            mixed_effects_all.to_csv(mixed_effects_all_path, index=False)
            print(f"\nSaved all mixed-effects coefficients to: {mixed_effects_all_path}")

    # 2. Transcript-level data used for plotting and paired tests.
    transcript_df = build_transcript_level_alignment(
        data=data,
        alignment_columns=alignment_columns,
        min_rows_per_transcript=args.min_rows_per_transcript,
    )

    transcript_summary_path = output_dir / "alignment_by_grammaticality_transcript_level.csv"
    transcript_df.to_csv(transcript_summary_path, index=False)
    print(f"\nSaved transcript-level data to: {transcript_summary_path}")

    summary_df = summarize_transcript_level_alignment(transcript_df)
    summary_path = output_dir / "alignment_by_grammaticality_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nMean transcript-level alignment by grammaticality and alignment type:")
    print(summary_df)
    print(f"\nSaved transcript-level summary to: {summary_path}")

    paired_tests_df = paired_transcript_tests(
        transcript_df=transcript_df,
        grammar_order=grammar_order,
    )

    paired_tests_path = output_dir / "alignment_by_grammaticality_paired_tests.csv"
    paired_tests_df.to_csv(paired_tests_path, index=False)

    print("\nTranscript-level paired statistical tests:")
    print(paired_tests_df)
    print(f"\nSaved paired tests to: {paired_tests_path}")

    # 3. Plot.
    create_alignment_plot(
        transcript_df=transcript_df,
        paired_tests_df=paired_tests_df,
        output_dir=output_dir,
        grammar_order=grammar_order,
    )

    print("\nHow to read the main stats:")
    print(
        "- Use alignment_by_grammaticality_paired_tests.csv for the stats that match the plot.\n"
        "- paired_t_p_fdr is the FDR-corrected paired t-test p-value.\n"
        "- wilcoxon_p_fdr is the FDR-corrected non-parametric paired test p-value.\n"
        "- cohens_dz is the paired-effect size: larger absolute values mean larger within-transcript differences.\n"
        "- Use mixed_effects_all_alignment_models.csv as the raw-row mixed-effects model check."
    )


def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze semantic/syntactic caregiver alignment by child utterance "
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
        default="results_alignment",
        help="Directory where plots, summaries, and stats will be saved.",
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
