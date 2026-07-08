#!/usr/bin/env python3
"""
Compute syntactic alignment from existing CHAT/MOR tag annotations.

This script is designed to mirror the syntactic alignment formula used in
alignment.py as closely as possible, except for one deliberate change:
utterance pairs with no POS/MOR bigrams are not treated as perfect alignment.

Original alignment.py logic:
    1. spaCy parses raw child/adult text.
    2. POS tags are extracted, excluding punctuation.
    3. POS bigram types are created.
    4. Alignment = |shared POS bigram types| / max(|child bigram types|, |adult bigram types|)
    5. If both bigram sets are empty, original code returns 1.0.

This script:
    1. Reads already extracted MOR/POS tag columns, e.g. child_mor_tags and parent_mor_tags.
    2. Creates MOR/POS bigram types from those tag sequences.
    3. Uses the same overlap/max-denominator formula.
    4. By default returns NaN when both sides have no bigrams, because one-token
       utterances should not be treated as perfect syntactic alignment.

Expected input columns by default:
    - child_mor_tags
    - parent_mor_tags

Default output column:
    - align_syntactic_mor_bigram
"""

import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


PUNCT_TAGS = {".", ",", "?", "!", ";", ":", "...", "--", "-"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute syntactic alignment from MOR/POS tag columns using the "
            "same POS-bigram overlap formula as alignment.py."
        )
    )

    parser.add_argument(
        "--input_path",
        required=True,
        help="Input CSV file containing paired child-parent utterances and MOR tag columns.",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help=(
            "Output CSV path. If omitted, saves next to the input file with "
            "'_syntactic_alignment_mor_bigram.csv' appended."
        ),
    )
    parser.add_argument(
        "--child_tags_column",
        default="child_mor_tags",
        help="Column containing child MOR/POS tag sequences.",
    )
    parser.add_argument(
        "--parent_tags_column",
        default="parent_mor_tags",
        help="Column containing parent/caregiver MOR/POS tag sequences.",
    )
    parser.add_argument(
        "--output_column",
        default="align_syntactic_mor_bigram",
        help="Name of the output syntactic alignment column.",
    )
    parser.add_argument(
        "--empty_bigram_policy",
        choices=["nan", "zero", "original"],
        default="nan",
        help=(
            "What to do when both utterances have no MOR/POS bigrams, usually because "
            "both have fewer than two tags. 'nan' means alignment is undefined; "
            "'zero' means no evidence of alignment; 'original' reproduces alignment.py "
            "and returns 1.0. Default: nan."
        ),
    )
    parser.add_argument(
        "--missing_policy",
        choices=["nan", "zero"],
        default="nan",
        help=(
            "What to do if one or both tag columns are missing/empty before bigram creation. "
            "Default: nan."
        ),
    )
    parser.add_argument(
        "--add_diagnostics",
        action="store_true",
        help="Add tag counts, bigram counts, and shared bigram counts for inspection.",
    )

    return parser.parse_args()


def clean_cell(value) -> Optional[str]:
    """Return a stripped string, or None for missing/empty placeholder values."""
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s:
        return None
    if s.lower() in {"nan", "none", "null"}:
        return None
    return s


def parse_tag_sequence(value) -> list[str]:
    """
    Parse a whitespace-separated MOR/POS tag sequence.

    The expected input is already tag-only, e.g.:
        'pro v n'
        'aux pro v inf v'

    The function also tolerates accidental full MOR tokens, e.g.:
        'pro|you v|want n|cookie .'
    by extracting the part before '|'.
    """
    s = clean_cell(value)
    if s is None:
        return []

    tags: list[str] = []

    for raw_token in s.split():
        # A token may still contain CHAT/MOR alternatives or compounds separated by ^.
        for part in raw_token.split("^"):
            part = part.strip()
            if not part:
                continue

            # If a full MOR token slipped through, keep only the tag before '|'.
            if "|" in part:
                tag = part.split("|", 1)[0].strip()
            else:
                tag = part

            if not tag or tag in PUNCT_TAGS:
                continue

            tags.append(tag)

    return tags


def bigrams(seq: list[str]) -> set[tuple[str, str]]:
    """Return unique adjacent bigram types, matching alignment.py's set() behavior."""
    return {(seq[i], seq[i + 1]) for i in range(len(seq) - 1)}


def compute_original_like_alignment(
    child_tags_value,
    parent_tags_value,
    empty_bigram_policy: str = "nan",
    missing_policy: str = "nan",
) -> float:
    """
    Compute MOR/POS-bigram syntactic alignment.

    Formula mirrors alignment.py:
        shared = len(child_bigrams & parent_bigrams)
        denom = max(len(child_bigrams), len(parent_bigrams))
        score = shared / denom

    Deliberate difference by default:
        If both bigram sets are empty, return NaN rather than 1.0.
    """
    child_tags = parse_tag_sequence(child_tags_value)
    parent_tags = parse_tag_sequence(parent_tags_value)

    if not child_tags or not parent_tags:
        return np.nan if missing_policy == "nan" else 0.0

    child_bigrams = bigrams(child_tags)
    parent_bigrams = bigrams(parent_tags)

    if not child_bigrams and not parent_bigrams:
        if empty_bigram_policy == "original":
            return 1.0
        if empty_bigram_policy == "zero":
            return 0.0
        return np.nan

    if not child_bigrams or not parent_bigrams:
        return 0.0

    shared = len(child_bigrams & parent_bigrams)
    denom = max(len(child_bigrams), len(parent_bigrams))
    return shared / denom


def build_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_syntactic_alignment_mor_bigram.csv")


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path) if args.output_path else build_output_path(input_path)

    df = pd.read_csv(input_path, low_memory=False)

    required = [args.child_tags_column, args.parent_tags_column]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns are: {df.columns.tolist()}"
        )

    scores = []
    diagnostics = {
        "child_tag_count": [],
        "parent_tag_count": [],
        "child_bigram_type_count": [],
        "parent_bigram_type_count": [],
        "shared_bigram_type_count": [],
    }

    for _, row in df.iterrows():
        child_value = row[args.child_tags_column]
        parent_value = row[args.parent_tags_column]

        score = compute_original_like_alignment(
            child_value,
            parent_value,
            empty_bigram_policy=args.empty_bigram_policy,
            missing_policy=args.missing_policy,
        )
        scores.append(score)

        if args.add_diagnostics:
            child_tags = parse_tag_sequence(child_value)
            parent_tags = parse_tag_sequence(parent_value)
            child_bi = bigrams(child_tags)
            parent_bi = bigrams(parent_tags)
            diagnostics["child_tag_count"].append(len(child_tags))
            diagnostics["parent_tag_count"].append(len(parent_tags))
            diagnostics["child_bigram_type_count"].append(len(child_bi))
            diagnostics["parent_bigram_type_count"].append(len(parent_bi))
            diagnostics["shared_bigram_type_count"].append(len(child_bi & parent_bi))

    df[args.output_column] = scores

    if args.add_diagnostics:
        for col, values in diagnostics.items():
            df[col] = values

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    valid_scores = pd.Series(scores).dropna()

    print("Done.")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Added column: {args.output_column}")
    print(f"Rows: {len(df):,}")
    print(f"Non-NaN alignment scores: {len(valid_scores):,}")
    if len(valid_scores):
        print(f"Mean alignment: {valid_scores.mean():.4f}")
        print(f"Min alignment:  {valid_scores.min():.4f}")
        print(f"Max alignment:  {valid_scores.max():.4f}")


if __name__ == "__main__":
    main()
