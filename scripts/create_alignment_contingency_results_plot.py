import argparse
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import seaborn as sns

from pymer4.models import Lmer


GRAMMAR_LABELS = {
    -1: "ungrammatical",
     0: "ambiguous",
     1: "grammatical",
}


def ensure_transcript_file(data: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure that transcript_file exists and is usable as a random-effect grouping factor.

    If transcript_file is missing or entirely empty, reconstruct it from subject + session.
    This matches the fix used for the clarification-request contingency analysis.
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


def validate_columns(data: pd.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        raise ValueError(
            f"\nMissing required columns: {missing_columns}\n"
            f"Available columns are: {data.columns.tolist()}\n"
        )


def print_basic_diagnostics(data: pd.DataFrame, alignment_columns: list[str]) -> None:
    print("\nColumns found:")
    print(data.columns.tolist())

    diagnostic_cols = ["is_grammatical", "transcript_file"] + alignment_columns

    print("\nMissing values:")
    print(data[diagnostic_cols].isna().sum())

    print("\nGrammaticality counts:")
    print(data["is_grammatical"].value_counts(dropna=False).sort_index())

    for col in alignment_columns:
        print(f"\nSummary for {col}:")
        print(data[col].describe())


def fit_alignment_model(data: pd.DataFrame, alignment_column: str) -> None:
    """
    Fit a mixed-effects linear regression:

        alignment ~ is_grammatical + (1 | transcript_file)

    This tests whether caregiver alignment varies with child utterance
    grammaticality, while accounting for transcript/session-specific baseline
    differences.
    """
    model_data = data.dropna(
        subset=[alignment_column, "is_grammatical", "transcript_file"]
    ).copy()

    if model_data.empty:
        raise ValueError(
            f"No usable rows for {alignment_column} after dropping missing values."
        )

    model_data["is_grammatical"] = model_data["is_grammatical"].astype(int)
    model_data["transcript_file"] = model_data["transcript_file"].astype(str)

    print("\n" + "=" * 70)
    print(f"MIXED-EFFECTS MODEL FOR {alignment_column}")
    print("=" * 70)
    print(f"Rows used: {len(model_data):,}")
    print(f"Transcripts used: {model_data['transcript_file'].nunique():,}")

    mod = Lmer(
        f"{alignment_column} ~ is_grammatical + (1 | transcript_file)",
        data=model_data,
    )

    fitted = mod.fit()

    print(fitted)

    cols_to_show = [
        col for col in ["Estimate", "SE", "T-stat", "P-val", "Sig"]
        if col in fitted.columns
    ]
    if cols_to_show:
        print("\nKey coefficients:")
        print(fitted[cols_to_show])


def build_transcript_level_alignment(
    data: pd.DataFrame,
    alignment_columns: list[str],
    min_rows_per_transcript: int,
) -> pd.DataFrame:
    """
    Compute transcript-level mean alignment by grammaticality.

    This mirrors the clarification-request plotting logic: instead of plotting
    raw rows directly, we first compute per-transcript means, then average those
    transcript-level values.
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
            data_grammar = data_transcript[
                data_transcript["is_grammatical"] == grammar_value
            ]

            if data_grammar.empty:
                continue

            for alignment_column in alignment_columns:
                mean_alignment = data_grammar[alignment_column].mean(skipna=True)

                if pd.isna(mean_alignment):
                    continue

                if alignment_column == "align_semantic":
                    alignment_type = "semantic"
                elif alignment_column == "align_syntactic":
                    alignment_type = "syntactic"
                else:
                    alignment_type = alignment_column

                entries.append({
                    "transcript_file": transcript_file,
                    "Grammaticality": grammar_label,
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


def create_alignment_plot(
    transcript_df: pd.DataFrame,
    output_dir: Path,
    include_ambiguous: bool,
) -> None:
    if include_ambiguous:
        grammar_order = ["grammatical", "ambiguous", "ungrammatical"]
    else:
        grammar_order = ["grammatical", "ungrammatical"]

    plot_data = transcript_df[
        transcript_df["Grammaticality"].isin(grammar_order)
    ].copy()

    print("\nMean transcript-level alignment by grammaticality and alignment type:")
    print(
        plot_data
        .groupby(["alignment_type", "Grammaticality"])
        .aggregate({"mean_alignment": ["mean", "std", "count"]})
    )

    sns.set_theme(font_scale=1.2)
    sns.set_style("ticks")
    sns.set_palette("Set2")

    plt.figure(figsize=(8, 5))

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
    plt.tight_layout()
    sns.despine()

    output_path = output_dir / "alignment_by_grammaticality.png"
    ax.get_figure().savefig(output_path, dpi=300)
    plt.close()

    print(f"\nSaved plot to: {output_path}")


def create_results(args):
    print("\nStats for annotated data:")

    data = pd.read_csv(args.data_path)

    print("\nLoaded file:")
    print(args.data_path)

    # Optional: rename common alternative column names
    data = data.rename(columns={
        "grammar": "is_grammatical",
        "grammaticality": "is_grammatical",
        "is_child_grammatical": "is_grammatical",
        "file": "transcript_file",
        "transcript": "transcript_file",
        "semantic_alignment": "align_semantic",
        "syntactic_alignment": "align_syntactic",
    })

    alignment_columns = ["align_semantic", "align_syntactic"]

    required_columns = [
        "is_grammatical",
        *alignment_columns,
    ]

    validate_columns(data, required_columns)
    data = ensure_transcript_file(data)

    print_basic_diagnostics(data, alignment_columns)

    # Keep only valid grammaticality values
    data = data.dropna(subset=["is_grammatical"]).copy()
    data["is_grammatical"] = data["is_grammatical"].astype(int)

    allowed_grammar_values = {-1, 0, 1}
    observed_grammar_values = set(data["is_grammatical"].unique())

    if not observed_grammar_values.issubset(allowed_grammar_values):
        raise ValueError(
            f"Unexpected values in is_grammatical: {observed_grammar_values}\n"
            "Expected only -1, 0, and 1."
        )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fit one model per alignment type
    if not args.skip_models:
        for alignment_column in alignment_columns:
            fit_alignment_model(data, alignment_column)

    # Build transcript-level data for plotting
    transcript_df = build_transcript_level_alignment(
        data=data,
        alignment_columns=alignment_columns,
        min_rows_per_transcript=args.min_rows_per_transcript,
    )

    transcript_summary_path = output_dir / "alignment_by_grammaticality_transcript_level.csv"
    transcript_df.to_csv(transcript_summary_path, index=False)
    print(f"\nSaved transcript-level summary to: {transcript_summary_path}")

    # Plot
    create_alignment_plot(
        transcript_df=transcript_df,
        output_dir=output_dir,
        include_ambiguous=args.include_ambiguous,
    )


def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the contingency of semantic/syntactic caregiver alignment "
            "on child utterance grammaticality."
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
        help="Directory where plots and summaries will be saved.",
    )

    parser.add_argument(
        "--min_rows_per_transcript",
        type=int,
        default=100,
        help=(
            "Minimum number of rows required for a transcript/session to be "
            "included in the transcript-level plot."
        ),
    )

    parser.add_argument(
        "--include_ambiguous",
        action="store_true",
        help="Include ambiguous utterances, is_grammatical == 0, in the plot.",
    )

    parser.add_argument(
        "--skip_models",
        action="store_true",
        help="Skip mixed-effects models and only create the transcript-level plot.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    create_results(args)
