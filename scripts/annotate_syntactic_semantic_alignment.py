#!/usr/bin/env python3
"""Syntactic and semantic alignment annotation for child-adult interactions.

This script takes a CSV/Excel file containing child utterance -> adult response
pairs and adds two alignment columns:

    - align_syntactic
    - align_semantic

It is a simplified version of the original sentiment/alignment script:
    - no sentiment annotation
    - no lexical unigram/bigram output
    - only syntactic and semantic alignment are saved
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd

warnings.filterwarnings("ignore")

from pkg import settings
from pkg.annotator.alignment import LinguisticAlignmentSuite
from pkg.settings import get_torch_device


##############################
# Argument parsing
##############################


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Annotate syntactic and semantic alignment for child-adult interaction pairs."
    )

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help=(
            "Path to input CSV/Excel file, relative to settings.PATH.dataset_root. "
            "The file should contain child-adult utterance pairs."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="annotated_alignment/",
        help="Directory where the annotated file will be saved, relative to settings.PATH.dataset_root.",
    )

    parser.add_argument(
        "--child_column",
        type=str,
        default="utt_transcript_clean",
        help="Column name containing child utterances.",
    )

    parser.add_argument(
        "--adult_column",
        type=str,
        default="response_transcript_clean",
        help="Column name containing adult/caregiver responses.",
    )

    parser.add_argument(
        "--spacy_model",
        type=str,
        default="en_core_web_sm",
        help="spaCy model used for syntactic analysis.",
    )

    parser.add_argument(
        "--semantic_model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model used for semantic alignment.",
    )

    parser.add_argument(
        "--exclude_stopwords",
        action="store_true",
        help="Exclude stopwords when the alignment suite computes lexical features internally.",
    )

    parser.add_argument(
        "--exclude_interjections",
        action="store_true",
        help="Exclude interjections when the alignment suite computes lexical features internally.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: process only the first 100 rows.",
    )

    return parser.parse_args(argv)


##############################
# Helpers
##############################


def resolve_dataset_path(path_str: str) -> Path:
    """Resolve an input/output path relative to the configured dataset root.

    If the user provides an absolute path, keep it as-is.
    Otherwise, interpret it relative to settings.PATH.dataset_root.
    """
    path = Path(path_str)
    if path.is_absolute():
        return path
    return settings.PATH.dataset_root / path


def load_table(input_path: Path) -> pd.DataFrame:
    """Load a CSV or Excel file into a DataFrame."""
    suffix = input_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(input_path)

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(input_path)

    raise ValueError(
        f"Unsupported input file format: {input_path.suffix}\n"
        "Expected a .csv, .xlsx, or .xls file."
    )


def check_required_columns(df: pd.DataFrame, child_column: str, adult_column: str) -> None:
    """Check that the required child/adult utterance columns are present."""
    required_cols = [child_column, adult_column]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}\n"
            f"Available columns are: {df.columns.tolist()}"
        )


def build_output_filename(input_path: Path, args) -> str:
    """Build an informative output filename."""
    suffix = "_syntactic_semantic_alignment"

    if args.exclude_stopwords:
        suffix += "_no_stopwords"
    if args.exclude_interjections:
        suffix += "_no_intjs"
    if args.debug:
        suffix += "_debug"

    return f"{input_path.stem}{suffix}.csv"


##############################
# Alignment annotation
##############################


def annotate_syntactic_semantic_alignment(
    df: pd.DataFrame,
    child_column: str,
    adult_column: str,
    spacy_model: str,
    semantic_model: str,
    exclude_stopwords: bool,
    exclude_interjections: bool,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """Annotate child-adult pairs with syntactic and semantic alignment scores."""
    print("\n" + "=" * 70)
    print("SYNTACTIC & SEMANTIC ALIGNMENT ANNOTATION")
    print("=" * 70)

    suite = LinguisticAlignmentSuite(
        spacy_model=spacy_model,
        semantic_model=semantic_model,
        exclude_stopwords=exclude_stopwords,
        exclude_interjections=exclude_interjections,
        device=device,
    )

    child_texts = df[child_column].fillna("").astype(str).tolist()
    adult_texts = df[adult_column].fillna("").astype(str).tolist()

    print(f"Processing {len(child_texts)} child-adult pairs...")

    alignment_results = suite.compute_batch(child_texts, adult_texts)

    missing_result_keys = [
        key
        for key in ["syntactic_alignment", "semantic_alignment"]
        if key not in alignment_results
    ]
    if missing_result_keys:
        raise KeyError(
            f"The alignment suite did not return expected keys: {missing_result_keys}\n"
            f"Available keys are: {list(alignment_results.keys())}"
        )

    alignment_df = pd.DataFrame(
        {
            "align_syntactic": alignment_results["syntactic_alignment"],
            "align_semantic": alignment_results["semantic_alignment"],
        }
    )

    return alignment_df


##############################
# Main
##############################


def main(argv):
    args = parse_args(argv)
    device = get_torch_device()

    input_path = resolve_dataset_path(args.input_path)
    output_dir = resolve_dataset_path(args.output_dir)

    print("=" * 70)
    print("SYNTACTIC & SEMANTIC ALIGNMENT ANNOTATION")
    print("=" * 70)
    print(f"Input file: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Child column: {args.child_column}")
    print(f"Adult column: {args.adult_column}")
    print(f"spaCy model: {args.spacy_model}")
    print(f"Semantic model: {args.semantic_model}")
    print(f"Device: {device}")

    print("\nLoading data...")
    df = load_table(input_path)

    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")

    check_required_columns(
        df=df,
        child_column=args.child_column,
        adult_column=args.adult_column,
    )

    if args.debug:
        print("\n" + "!" * 70)
        print("DEBUG MODE: Processing only the first 100 rows")
        print("!" * 70)
        df = df.head(100).copy()

    alignment_df = annotate_syntactic_semantic_alignment(
        df=df,
        child_column=args.child_column,
        adult_column=args.adult_column,
        spacy_model=args.spacy_model,
        semantic_model=args.semantic_model,
        exclude_stopwords=args.exclude_stopwords,
        exclude_interjections=args.exclude_interjections,
        device=device,
    )

    print("✓ Alignment annotation complete")
    print(f"  Added columns: {alignment_df.columns.tolist()}")

    print("\nCombining annotations...")
    result_df = pd.concat([df.reset_index(drop=True), alignment_df.reset_index(drop=True)], axis=1)
    result_df = result_df.loc[:, ~result_df.columns.duplicated()]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file_name = build_output_filename(input_path, args)
    output_path = output_dir / output_file_name

    result_df.to_csv(output_path, index=False)

    print("\n" + "=" * 70)
    print("ANNOTATION COMPLETE")
    print("=" * 70)
    print(f"Output saved to: {output_path}")
    print(f"Total rows: {len(result_df)}")
    print(f"Total columns: {len(result_df.columns)}")


if __name__ == "__main__":
    main(sys.argv[1:])
