#!/usr/bin/env python3

from pathlib import Path
import pandas as pd


INPUT_FILES = {
    "typical": "conversation_pairs_no_lesion_annotated_grammar_syntactic_semantic_alignment.csv",
    "lesion_left": "conversation_pairs_lesion_left_annotated_grammar_syntactic_semantic_alignment.csv",
    "lesion_right": "conversation_pairs_lesion_right_annotated_grammar_syntactic_semantic_alignment.csv",
}

OUTPUT_FILE = "conversation_pairs_all_groups_annotated.csv"


def load_and_label(path: str, group3: str) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Could not find file: {path}")

    df = pd.read_csv(path)

    # Add clean group labels for later analyses
    df["group3"] = group3

    if group3 == "typical":
        df["group2"] = "typical"
        df["has_brain_lesion_clean"] = False
        df["lesion_side_clean"] = "none"
    elif group3 == "lesion_left":
        df["group2"] = "lesion"
        df["has_brain_lesion_clean"] = True
        df["lesion_side_clean"] = "left"
    elif group3 == "lesion_right":
        df["group2"] = "lesion"
        df["has_brain_lesion_clean"] = True
        df["lesion_side_clean"] = "right"
    else:
        raise ValueError(f"Unexpected group: {group3}")

    return df


def main():
    dfs = []

    for group3, file_path in INPUT_FILES.items():
        df = load_and_label(file_path, group3)
        print(f"{group3}: {len(df):,} rows, {df['subject'].nunique()} subjects")
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)

    print("\nMerged file:")
    print(f"Rows: {len(merged):,}")
    print(f"Subjects: {merged['subject'].nunique()}")

    print("\nRows by group3:")
    print(merged["group3"].value_counts())

    print("\nSubjects by group3:")
    print(merged.groupby("group3")["subject"].nunique())

    # Optional sanity check: show important columns if they exist
    important_cols = [
        "subject",
        "group3",
        "group2",
        "has_brain_lesion_clean",
        "lesion_side_clean",
        "transcript_file",
        "utt_transcript_clean",
        "response_transcript_clean",
        "is_cr",
        "is_grammatical",
        "align_syntactic",
        "align_semantic",
    ]

    existing_important_cols = [col for col in important_cols if col in merged.columns]
    print("\nImportant columns found:")
    for col in existing_important_cols:
        print(f"- {col}")

    # Save
    merged.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved merged file to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
