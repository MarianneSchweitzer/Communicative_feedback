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
        col for col in ["is_cr", "is_grammatical", "subject", "session", "age"]
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


def clean_and_validate_data(data: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, drop unusable rows, and validate values."""
    data = data.rename(columns=COLUMN_RENAMES)

    required_columns = [
        "is_cr",
        "is_grammatical",
        "subject",
        "session",
        "age",
    ]

    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        raise ValueError(
            f"\nMissing required columns: {missing_columns}\n"
            f"Available columns are: {data.columns.tolist()}\n"
        )

    print_basic_diagnostics(data)

    before_drop = len(data)
    data = data.dropna(subset=["is_cr", "is_grammatical", "subject", "session", "age"]).copy()
    after_drop = len(data)

    if before_drop != after_drop:
        print(f"\nDropped {before_drop - after_drop} rows with missing values.")

    if data.empty:
        raise ValueError(
            "\nNo rows left after dropping missing values.\n"
            "The model cannot be fitted. Check the diagnostics printed above."
        )

    data["is_cr"] = pd.to_numeric(data["is_cr"], errors="coerce")
    data["is_grammatical"] = pd.to_numeric(data["is_grammatical"], errors="coerce")
    data["age"] = pd.to_numeric(data["age"], errors="coerce")

    before_numeric_drop = len(data)
    data = data.dropna(subset=["is_cr", "is_grammatical", "age"]).copy()
    after_numeric_drop = len(data)

    if before_numeric_drop != after_numeric_drop:
        print(
            f"\nDropped {before_numeric_drop - after_numeric_drop} rows "
            "whose is_cr/is_grammatical/age values could not be converted to numbers."
        )

    data["is_cr"] = data["is_cr"].astype(int)
    data["is_grammatical"] = data["is_grammatical"].astype(int)
    data["subject"] = data["subject"].astype(str)
    data["session"] = data["session"].astype(str)

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

    n_subjects = data["subject"].nunique()
    n_sessions = data.groupby("subject")["session"].nunique().sum()
    print("\nRows after filtering:", len(data))
    print("Number of subjects:", n_subjects)
    print("Number of sessions (total):", n_sessions)

    if n_subjects < 2:
        raise ValueError(
            "The mixed-effects model needs at least two subjects. "
            f"Only found {n_subjects}."
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


def fit_mixed_model(data: pd.DataFrame, output_dir: Path):
    """Fit mixed-effects logistic regression and save results."""
    model_data = data.copy()

    model_data["is_cr"] = model_data["is_cr"].astype(bool)

    formula = "is_cr ~ is_grammatical + age + (1 | subject/session)"

    mod = Lmer(
        formula,
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

    output_dir.mkdir(parents=True, exist_ok=True)

    results = fitted.copy()
    results.insert(0, "model", formula)
    stats_path = output_dir / "mixed_effects_cr.csv"
    results.to_csv(stats_path, index=True)

    print(f"\nSaved stats to: {stats_path}")

    return fitted


def build_session_level_proportions(data: pd.DataFrame, min_session_rows: int) -> pd.DataFrame:
    """Build session-level CR proportions for each grammaticality class."""
    entries = []

    session_groups = data.groupby(["subject", "session"])

    for (subject, session), data_session in tqdm(session_groups, total=len(session_groups)):
        if len(data_session) <= min_session_rows:
            continue

        for is_grammatical in [-1, 0, 1]:
            data_filtered = data_session[
                data_session["is_grammatical"] == is_grammatical
            ]

            if len(data_filtered) == 0:
                continue

            counts = data_filtered["is_cr"].value_counts(normalize=True)

            for is_cr, proportion in counts.items():
                entries.append({
                    "is_cr": "clarification request" if is_cr == 1 else "other",
                    "grammaticality": GRAMMAR_LABELS[is_grammatical],
                    "proportion": proportion,
                    "subject": subject,
                    "session": session,
                    "n_rows_session": len(data_session),
                    "n_rows_condition": len(data_filtered),
                })

    df = pd.DataFrame(entries)

    if df.empty:
        session_sizes = session_groups.size().sort_values(ascending=False)
        raise ValueError(
            "No session-level entries were created. "
            f"This may mean that no session has more than {min_session_rows} rows.\n\n"
            "Largest session sizes:\n"
            f"{session_sizes.head(20)}"
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

    fit_mixed_model(data, output_dir=Path(args.output_dir))

    df = build_session_level_proportions(
        data=data,
        min_session_rows=args.min_session_rows,
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
        "--min_session_rows",
        type=int,
        default=100,
        help="Minimum number of rows required for a session to be included in the plot.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    create_results_plot(args)
