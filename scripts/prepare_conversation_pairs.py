#!/usr/bin/env python3
"""
Prepare child→parent utterance-pair CSVs for the CR/feedback classifier.

Input CSV columns expected:
  - subject
  - session
  - row
  - line
  - p_utts
  - c_utts
Optional but preserved when present:
  - p_utts_orig, c_utts_orig, context, task, tape, path

Output files:
  - conversation_pairs_prepared_all.csv
  - conversation_pairs_brain_lesion.csv
  - conversation_pairs_typical.csv

The important classifier-compatible columns are:
  - utt_transcript_clean              # child utterance
  - response_transcript_clean         # immediately following parent utterance
"""

import argparse
import csv
import os
import re
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd

DEFAULT_LESION_SUBJECTS = {30, 32, 34, 35, 46, 54, 55, 56, 57, 58, 68, 69, 70, 72}

BASE_COLS = [
    "subject", "session", "row", "line", "p_utts", "c_utts",
    "p_utts_orig", "c_utts_orig", "context", "task", "tape", "path",
]

OUTPUT_COLS = [
    "subject", "group", "has_brain_lesion", "session", "transcript_file", "path",
    "child_row", "parent_row", "child_line", "parent_line",
    "child_orig_index", "parent_orig_index",
    "utt_transcript_clean", "response_transcript_clean",
    "c_utts_orig", "p_utts_orig",
    "child_context", "parent_context", "child_task", "parent_task",
]


def parse_lesion_subjects(value: Optional[str]) -> set[int]:
    if not value:
        return set(DEFAULT_LESION_SUBJECTS)
    return {int(x.strip()) for x in value.split(",") if x.strip()}


def is_missing(x) -> bool:
    return pd.isna(x)


def clean_str(x) -> Optional[str]:
    if is_missing(x):
        return None
    return str(x).strip()


def valid_utt(x) -> bool:
    """Filter out empty cells and obvious transcription placeholders."""
    s = clean_str(x)
    if not s:
        return False
    if re.fullmatch(r"[-–—]+", s):
        return False
    if re.fullmatch(r"#+", s):
        return False
    if s.lower() in {"nan", "xxx", "yyy", "www"}:
        return False
    return True


def subject_to_int(x) -> Optional[int]:
    if is_missing(x):
        return None
    try:
        return int(float(x))
    except Exception:
        return None


def group_key(row) -> Tuple:
    """Transcript-like key used to avoid pairing across subjects/sessions/tapes."""
    return (row.get("subject"), row.get("session"), row.get("tape"))


def make_event(row: Dict, speaker: str, orig_index: int) -> Dict:
    if speaker == "parent":
        utt_col, orig_col = "p_utts", "p_utts_orig"
    else:
        utt_col, orig_col = "c_utts", "c_utts_orig"

    return {
        "speaker": speaker,
        "utt": clean_str(row.get(utt_col)),
        "utt_orig": clean_str(row.get(orig_col)),
        "subject": row.get("subject"),
        "session": row.get("session"),
        "row": row.get("row"),
        "line": row.get("line"),
        "context": row.get("context"),
        "task": row.get("task"),
        "tape": row.get("tape"),
        "path": row.get("path"),
        "orig_index": orig_index,
    }


def row_to_events(row: Dict, orig_index: int) -> Iterable[Dict]:
    """
    Convert a source row to ordered utterance events.

    If both parent and child are present on the same row, parent is emitted first.
    In this corpus these rows often encode a parent prompt and child production
    on the same line, so parent→child ordering is the safer default.
    """
    if valid_utt(row.get("p_utts")):
        yield make_event(row, "parent", orig_index)
    if valid_utt(row.get("c_utts")):
        yield make_event(row, "child", orig_index)


def pair_child_parent(child_ev: Dict, parent_ev: Dict, lesion_subjects: set[int]) -> Dict:
    subj_int = subject_to_int(child_ev["subject"])
    has_lesion = subj_int in lesion_subjects if subj_int is not None else False
    return {
        "subject": subj_int,
        "group": "brain_lesion" if has_lesion else "typical",
        "has_brain_lesion": int(has_lesion),
        "session": child_ev["session"],
        "transcript_file": child_ev["tape"],
        "path": child_ev["path"],
        "child_row": child_ev["row"],
        "parent_row": parent_ev["row"],
        "child_line": child_ev["line"],
        "parent_line": parent_ev["line"],
        "child_orig_index": child_ev["orig_index"],
        "parent_orig_index": parent_ev["orig_index"],
        "utt_transcript_clean": child_ev["utt"],
        "response_transcript_clean": parent_ev["utt"],
        "c_utts_orig": child_ev["utt_orig"],
        "p_utts_orig": parent_ev["utt_orig"],
        "child_context": child_ev["context"],
        "parent_context": parent_ev["context"],
        "child_task": child_ev["task"],
        "parent_task": parent_ev["task"],
    }


def open_writer(path: str):
    f = open(path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=OUTPUT_COLS)
    writer.writeheader()
    return f, writer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Path to the large source CSV")
    parser.add_argument("--output_dir", default=".", help="Directory where output CSVs will be written")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument(
        "--lesion_subjects",
        default=None,
        help="Comma-separated lesion subject IDs. Defaults to the list from the project.",
    )
    args = parser.parse_args()

    lesion_subjects = parse_lesion_subjects(args.lesion_subjects)
    os.makedirs(args.output_dir, exist_ok=True)

    out_all = os.path.join(args.output_dir, "conversation_pairs_prepared_all.csv")
    out_lesion = os.path.join(args.output_dir, "conversation_pairs_brain_lesion.csv")
    out_typical = os.path.join(args.output_dir, "conversation_pairs_typical.csv")

    # Discover available columns so the script also works if optional columns are missing.
    available_cols = pd.read_csv(args.data_path, nrows=0).columns.tolist()
    usecols = [c for c in BASE_COLS if c in available_cols]
    required = {"subject", "session", "row", "p_utts", "c_utts"}
    missing = required - set(usecols)
    if missing:
        raise ValueError(f"Missing required input columns: {sorted(missing)}")

    # pending_child stores the latest child event whose immediately following event
    # has not been observed yet, separately for each transcript-like group.
    pending_child: Dict[Tuple, Optional[Dict]] = {}

    counts = {"all": 0, "brain_lesion": 0, "typical": 0}
    subjects_seen = {"brain_lesion": set(), "typical": set()}
    global_row_offset = 0

    f_all, w_all = open_writer(out_all)
    f_lesion, w_lesion = open_writer(out_lesion)
    f_typical, w_typical = open_writer(out_typical)

    try:
        reader = pd.read_csv(
            args.data_path,
            usecols=usecols,
            chunksize=args.chunksize,
            low_memory=False,
        )

        for chunk_idx, chunk in enumerate(reader):
            # Add missing optional columns as empty so downstream code is simpler.
            for col in BASE_COLS:
                if col not in chunk.columns:
                    chunk[col] = None

            for local_i, row in enumerate(chunk.to_dict(orient="records")):
                orig_index = global_row_offset + local_i
                key = group_key(row)

                for ev in row_to_events(row, orig_index):
                    if ev["speaker"] == "child":
                        # If two child utterances occur in a row, the first one has no immediate
                        # parent response. Keep only the latest child event.
                        pending_child[key] = ev
                    else:  # parent event
                        child_ev = pending_child.get(key)
                        if child_ev is not None:
                            pair = pair_child_parent(child_ev, ev, lesion_subjects)
                            w_all.writerow(pair)
                            counts["all"] += 1

                            if pair["has_brain_lesion"]:
                                w_lesion.writerow(pair)
                                counts["brain_lesion"] += 1
                                subjects_seen["brain_lesion"].add(pair["subject"])
                            else:
                                w_typical.writerow(pair)
                                counts["typical"] += 1
                                subjects_seen["typical"].add(pair["subject"])

                            # Clear because this child has received its immediate parent response.
                            pending_child[key] = None

            global_row_offset += len(chunk)
            print(
                f"Processed chunk {chunk_idx + 1}; total pairs so far: {counts['all']:,}",
                flush=True,
            )

    finally:
        f_all.close()
        f_lesion.close()
        f_typical.close()

    print("\nDone.")
    print(f"All pairs:          {counts['all']:,} -> {out_all}")
    print(f"Brain lesion pairs: {counts['brain_lesion']:,} -> {out_lesion}")
    print(f"Typical pairs:      {counts['typical']:,} -> {out_typical}")
    print("Brain lesion subjects present:", sorted(x for x in subjects_seen["brain_lesion"] if x is not None))
    print("Typical subjects present count:", len(subjects_seen["typical"]))


if __name__ == "__main__":
    main()
