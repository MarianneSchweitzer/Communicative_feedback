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


COLUMN_RENAMES = {
    "grammar": "is_grammatical",
    "grammaticality": "is_grammatical",
    "is_child_grammatical": "is_grammatical",
    "cr": "is_cr",
    "clarification_request": "is_cr",
    "file": "transcript_file",
    "transcript": "transcript_file",
}


def print_basic_diagnostics(data: pd.DataFrame) -> None:
    """Print useful checks before dropping missing rows."""
    diagnostic_cols = [
        col for col in ["is_cr", "is_grammatical", "transcript_file", "subject", "session"]
        if col in data.columns
    ]

    print("\nRows before filtering:", len(data))

    if diagnostic_cols:
        print("\nMissing values before filtering:")
        print(data[diagnostic_cols].isna().sum())

    if "is_cr" in data.columns:
        print("\nValue counts for is_cr before filtering:")
        print(data["is_cr"].value_counts(dropna=False))

    if "is_grammatical" in data.columns:
        print("\nValue counts for is_grammatical before filtering:")
        print(data["is_grammatical"].value_counts(dropna=False))

    if "transcript_file" in data.columns:
        print("\nFirst transcript_file values before filtering:")
        print(data["transcript_file"].head(10))
        print("\nNumber of distinct transcript_file values before filtering:")
        print(data["transcript_file"].nunique(dropna=True))


def ensure_transcript_file(data: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure there is a usable transcript_file column for the random intercept.

    In some prepared CSVs, transcript_file was created from the original 'tape'
    column. If that column was empty, transcript_file is empty too. In that case,
    we reconstruct a transcript/session identifier from subject + session.
    """
    data = data.copy()

    transcript_missing_or_empty = (
        "transcript_file" not in data.columns
        or data["transcript_file"].isna().all()
        or data["transcript_file"].astype(str).str.strip().replace({"nan": ""}).eq("").all()
    )

    if transcript_missing_or_empty:
        if {"subject", "session"}.issubset(data.columns):
            print(
                "\ntranscript_file is missing or entirely empty. "
                "Creating transcript_file from subject + session."
            )

            subject_str = data["subject"].astype("Int64").astype(str)
            session_str = data["session"].astype("Int64").astype(str)

            data["transcript_file"] = "subj_" + subject_str + "_sess_" + session_str
        else:
            raise ValueError(
                "transcript_file is missing/empty, and subject/session columns are not "
                "available to reconstruct it."
            )
    else:
        # Normalize existing transcript IDs to strings.
        data["transcript_file"] = data["transcript_file"].astype(str).str.strip()

    return data


def clean_and_validate_data(data: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, reconstruct transcript_file if needed, drop unusable rows, and validate values."""
    data = data.rename(columns=COLUMN_RENAMES)
    data = ensure_transcript_file(data)

    required_columns = [
        "is_cr",
        "is_grammatical",
        "transcript_file",
    ]

    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        raise ValueError(
            f"\nMissing required columns: {missing_columns}\n"
            f"Available columns are: {data.columns.tolist()}\n\n"
            "This script expects an annotated CSV with at least:\n"
            "- is_cr: 1 for clarification request, 0 otherwise\n"
            "- is_grammatical: -1 ungrammatical, 0 ambiguous, 1 grammatical\n"
            "- transcript_file: transcript/session identifier\n"
            "  If transcript_file is empty, subject + session can be used to rebuild it.\n"
        )

    print_basic_diagnostics(data)

    before_drop = len(data)
    data = data.dropna(subset=["is_cr", "is_grammatical", "transcript_file"]).copy()
    after_drop = len(data)

    if before_drop != after_drop:
        print(f"\nDropped {before_drop - after_drop} rows with missing values.")

    if data.empty:
        raise ValueError(
            "\nNo rows left after dropping missing values in "
            "is_cr, is_grammatical, and transcript_file.\n"
            "The model cannot be fitted. Check the diagnostics printed above."
        )

    # Convert labels to numeric safely.
    data["is_cr"] = pd.to_numeric(data["is_cr"], errors="coerce")
    data["is_grammatical"] = pd.to_numeric(data["is_grammatical"], errors="coerce")

    before_numeric_drop = len(data)
    data = data.dropna(subset=["is_cr", "is_grammatical"]).copy()
    after_numeric_drop = len(data)

    if before_numeric_drop != after_numeric_drop:
        print(
            f"\nDropped {before_numeric_drop - after_numeric_drop} rows "
            "whose is_cr/is_grammatical values could not be converted to numbers."
        )

    data["is_cr"] = data["is_cr"].astype(int)
    data["is_grammatical"] = data["is_grammatical"].astype(int)
    data["transcript_file"] = data["transcript_file"].astype(str)

    allowed_cr_values = {0, 1}
    observed_cr_values = set(data["is_cr"].unique())

    if not observed_cr_values.issubset(allowed_cr_values):
        raise ValueError(
            f"Unexpected values in is_cr: {observed_cr_values}\n"
            "Expected only 0 and 1."
        )

    allowed_grammar_values = {-1, 0, 1}
    observed_grammar_values = set(data["is_grammatical"].unique())

    if not observed_grammar_values.issubset(allowed_grammar_values):
        raise ValueError(
            f"Unexpected values in is_grammatical: {observed_grammar_values}\n"
            "Expected only -1, 0, and 1."
        )

    n_transcripts = data["transcript_file"].nunique()
    print("\nRows after filtering:", len(data))
    print("Number of transcript_file groups:", n_transcripts)

    if n_transcripts < 2:
        raise ValueError(
            "The mixed-effects model needs at least two transcript_file groups. "
            f"Only found {n_transcripts}."
        )

    return data


def print_annotation_stats(data: pd.DataFrame) -> None:
    print("\nClarification requests:")
    print("Mean grammaticality:", data[data["is_cr"] == 1]["is_grammatical"].mean())
    print(data[data["is_cr"] == 1]["is_grammatical"].value_counts().sort_index())

    print("\nOther responses:")
    print("Mean grammaticality:", data[data["is_cr"] == 0]["is_grammatical"].mean())
    print(data[data["is_cr"] == 0]["is_grammatical"].value_counts().sort_index())

    print("\nSamples of annotated data:")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", 1000)

    n_samples = min(50, len(data))
    print(data.sample(n_samples, random_state=42))


def fit_mixed_model(data: pd.DataFrame):
    """Fit mixed-effects logistic regression."""
    model_data = data.copy()

    # pymer4/lme4 binomial models work cleanly with boolean or 0/1 outcomes.
    model_data["is_cr"] = model_data["is_cr"].astype(bool)

    mod = Lmer(
        "is_cr ~ is_grammatical + (1 | transcript_file)",
        family="binomial",
        data=model_data,
    )

    print("=" * 50 + "\nGLM\n" + "=" * 50)

    fitted = mod.fit()

    print(fitted)

    cols_to_print = [col for col in ["Estimate", "SE", "Sig"] if col in fitted.columns]
    if cols_to_print:
        print(fitted[cols_to_print])
    else:
        print(fitted)

    return fitted


def build_transcript_level_proportions(data: pd.DataFrame, min_transcript_rows: int) -> pd.DataFrame:
    """Build transcript-level CR proportions for each grammaticality class."""
    entries = []

    for transcript_file in tqdm(data["transcript_file"].unique()):
        data_transcript = data[data["transcript_file"] == transcript_file]

        # Keep only sufficiently large transcripts.
        if len(data_transcript) <= min_transcript_rows:
            continue

        for is_grammatical in [-1, 0, 1]:
            data_filtered = data_transcript[
                data_transcript["is_grammatical"] == is_grammatical
            ]

            if len(data_filtered) == 0:
                continue

            counts = data_filtered["is_cr"].value_counts(normalize=True)

            for is_cr, proportion in counts.items():
                entries.append({
                    "is_cr": "clarification request" if is_cr == 1 else "other",
                    "grammaticality": GRAMMAR_LABELS[is_grammatical],
                    "proportion": proportion,
                    "transcript_file": transcript_file,
                    "n_rows_transcript": len(data_transcript),
                    "n_rows_condition": len(data_filtered),
                })

    df = pd.DataFrame(entries)

    if df.empty:
        transcript_sizes = data["transcript_file"].value_counts()
        raise ValueError(
            "No transcript-level entries were created. "
            f"This may mean that no transcript has more than {min_transcript_rows} rows.\n\n"
            "Largest transcript sizes:\n"
            f"{transcript_sizes.head(20)}"
        )

    return df


def create_plot(data_filtered: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(font_scale=1.2)
    sns.set_style("ticks")
    sns.set_palette("Set2")

    plt.figure(figsize=(6, 5))

    grammar_order = ["grammatical", "ungrammatical"]

    fig = sns.barplot(
        data=data_filtered,
        x="Grammaticality",
        y="proportion",
        hue="Grammaticality",
        order=grammar_order,
        hue_order=grammar_order,
        errorbar="ci",
        legend=False,
    )

    plt.ylabel("Proportion of Responses that are\nClarification Requests")
    plt.xlabel("")
    plt.ylim((0, 0.24))
    plt.tight_layout()
    sns.despine()

    output_path = output_dir / "grammaticality.png"
    fig.get_figure().savefig(output_path, dpi=300)

    print(f"\nSaved plot to: {output_path}")

    # Print CI bounds from seaborn plot.
    ax = plt.gca()
    lower = [line.get_ydata().min() for line in ax.lines]
    upper = [line.get_ydata().max() for line in ax.lines]

    print("CI lower bounds:", lower)
    print("CI upper bounds:", upper)


def create_results_plot(args):
    print("\nStats for annotated data:")

    data = pd.read_csv(args.data_path)

    print("\nLoaded file:")
    print(args.data_path)

    print("\nColumns found:")
    print(data.columns.tolist())

    data = clean_and_validate_data(data)

    print_annotation_stats(data)

    fit_mixed_model(data)

    df = build_transcript_level_proportions(
        data=data,
        min_transcript_rows=args.min_transcript_rows,
    )

    grammar_order = ["grammatical", "ungrammatical"]

    data_filtered = df[df["grammaticality"].isin(grammar_order)]
    data_filtered = data_filtered[
        data_filtered["is_cr"] == "clarification request"
    ].copy()

    if data_filtered.empty:
        raise ValueError(
            "No data left for the grammatical vs ungrammatical clarification-request plot. "
            "Check whether both grammatical and ungrammatical rows exist after filtering."
        )

    data_filtered = data_filtered.rename(
        columns={"grammaticality": "Grammaticality"}
    )

    print("\nMean proportion of clarification requests by grammaticality:")
    print(
        data_filtered
        .groupby("Grammaticality")
        .aggregate({"proportion": "mean"})
    )

    print("\nStandard deviation by grammaticality:")
    print(
        data_filtered
        .groupby("Grammaticality")
        .aggregate({"proportion": "std"})
    )

    create_plot(
        data_filtered=data_filtered,
        output_dir=Path(args.output_dir),
    )


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to annotated conversation-pair CSV.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory where plots will be saved.",
    )

    parser.add_argument(
        "--min_transcript_rows",
        type=int,
        default=100,
        help="Minimum number of rows required for a transcript to be included in the plot.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    create_results_plot(args)
