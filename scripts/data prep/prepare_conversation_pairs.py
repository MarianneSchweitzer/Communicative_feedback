#!/usr/bin/env python3
"""
Prepare child→parent utterance-pair CSVs for the CR/feedback classifier,
while preserving morphosyntactic annotations from c_mor/p_mor.

This version derives lesion groups from a subjects CSV instead of using a
hardcoded subject list.

Expected conversation CSV columns:
  - subject
  - session
  - row
  - line
  - p_utts
  - c_utts
  - p_mor      # parent MOR annotation
  - c_mor      # child MOR annotation

Optional conversation columns preserved when present:
  - p_utts_orig, c_utts_orig, context, task, tape, path

Expected subjects CSV columns by default:
  - id        # subject ID, matching conversation_csv["subject"]
  - lesion    # NaN/empty = no lesion, L = left lesion, R = right lesion

Output files:
  - conversation_pairs_prepared_all.csv
  - conversation_pairs_lesion.csv
  - conversation_pairs_no_lesion.csv
  - conversation_pairs_lesion_left.csv
  - conversation_pairs_lesion_right.csv

The important classifier-compatible columns are:
  - utt_transcript_clean              # child utterance
  - response_transcript_clean         # immediately following parent utterance

The added syntax-related columns are:
  - child_mor                         # original child c_mor annotation
  - parent_mor                        # original parent p_mor annotation
  - child_mor_tags                    # tags extracted before | from child_mor
  - parent_mor_tags                   # tags extracted before | from parent_mor

Example:
  c_mor = "pro|I v|want n|cookie ."
  child_mor_tags = "pro v n"
"""

import argparse
import csv
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


BASE_COLS = [
    "subject", "session", "row", "line", "p_utts", "c_utts",
    "p_utts_orig", "c_utts_orig", "p_mor", "c_mor",
    "context", "task", "tape", "path",
]

OUTPUT_COLS = [
    "subject",
    "group",
    "has_brain_lesion",
    "lesion_side",
    "lesion_raw",
    "subject_active",
    "subject_control",
    "session",
    "transcript_file",
    "path",
    "child_row",
    "parent_row",
    "child_line",
    "parent_line",
    "child_orig_index",
    "parent_orig_index",
    "utt_transcript_clean",
    "response_transcript_clean",
    "c_utts_orig",
    "p_utts_orig",
    "child_mor",
    "parent_mor",
    "child_mor_tags",
    "parent_mor_tags",
    "child_context",
    "parent_context",
    "child_task",
    "parent_task",
]


MOR_TAG_RE = re.compile(r"([^\s^|]+)\|")


def is_missing(x) -> bool:
    return pd.isna(x)


def clean_str(x) -> Optional[str]:
    if is_missing(x):
        return None
    s = str(x).strip()
    return s if s else None


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


def extract_mor_tags(mor) -> List[str]:
    """
    Extract morphosyntactic tags from a CHAT/CHILDES-style MOR string.

    The rule is deliberately simple and faithful to the user's request:
    keep the part before each pipe character '|'.

    Examples:
      "v|play prep|with n|hand ." -> ["v", "prep", "n"]
      "pro:poss:det|my n|phone ." -> ["pro:poss:det", "n"]
      "n:prop|Mommy^n:prop|Mommy-POSS^" -> ["n:prop", "n:prop"]

    Missing values, empty strings, and placeholder-only values return [].
    """
    s = clean_str(mor)
    if not s:
        return []
    if re.fullmatch(r"[-–—]+", s):
        return []
    if s.lower() in {"nan", "xxx", "yyy", "www"}:
        return []
    return MOR_TAG_RE.findall(s)


def mor_tags_to_string(mor, sep: str = " ") -> Optional[str]:
    tags = extract_mor_tags(mor)
    if not tags:
        return None
    return sep.join(tags)


def subject_to_int(x) -> Optional[int]:
    if is_missing(x):
        return None
    try:
        return int(float(x))
    except Exception:
        return None


def normalize_lesion_side(value) -> Optional[str]:
    """
    Convert the subjects-file lesion column to a normalized lesion side.

    Returns:
      - "left" for L/left
      - "right" for R/right
      - None for missing/no lesion
      - "unknown" for unexpected non-empty values
    """
    s = clean_str(value)
    if s is None:
        return None

    s_norm = s.strip().lower()
    if s_norm in {"nan", "none", "no", "n", "0", "false", "control", "typical", "no_lesion"}:
        return None
    if s_norm in {"l", "left", "left_lesion", "lesion_left"}:
        return "left"
    if s_norm in {"r", "right", "right_lesion", "lesion_right"}:
        return "right"
    return "unknown"


def group_from_lesion_side(lesion_side: Optional[str]) -> str:
    if lesion_side == "left":
        return "lesion_left"
    if lesion_side == "right":
        return "lesion_right"
    if lesion_side == "unknown":
        return "unknown_lesion"
    return "no_lesion"


def load_subject_metadata(
    subjects_path: str,
    subject_id_col: str = "id",
    lesion_col: str = "lesion",
    active_col: str = "active",
    control_col: str = "control",
) -> Dict[int, Dict]:
    subjects = pd.read_csv(subjects_path)

    required = {subject_id_col, lesion_col}
    missing = required - set(subjects.columns)
    if missing:
        raise ValueError(f"Missing required subjects CSV columns: {sorted(missing)}")

    meta: Dict[int, Dict] = {}

    for _, row in subjects.iterrows():
        subj = subject_to_int(row.get(subject_id_col))
        if subj is None:
            continue

        lesion_raw = clean_str(row.get(lesion_col))
        lesion_side = normalize_lesion_side(row.get(lesion_col))
        group = group_from_lesion_side(lesion_side)

        meta[subj] = {
            "lesion_raw": lesion_raw,
            "lesion_side": lesion_side,
            "group": group,
            "has_brain_lesion": int(lesion_side in {"left", "right"}),
            "subject_active": row.get(active_col) if active_col in subjects.columns else None,
            "subject_control": row.get(control_col) if control_col in subjects.columns else None,
        }

    return meta


def group_key(row) -> Tuple:
    """Transcript-like key used to avoid pairing across subjects/sessions/tapes."""
    return (row.get("subject"), row.get("session"), row.get("tape"))


def make_event(row: Dict, speaker: str, orig_index: int, mor_tag_sep: str = " ") -> Dict:
    if speaker == "parent":
        utt_col = "p_utts"
        orig_col = "p_utts_orig"
        mor_col = "p_mor"
    else:
        utt_col = "c_utts"
        orig_col = "c_utts_orig"
        mor_col = "c_mor"

    mor = clean_str(row.get(mor_col))

    return {
        "speaker": speaker,
        "utt": clean_str(row.get(utt_col)),
        "utt_orig": clean_str(row.get(orig_col)),
        "mor": mor,
        "mor_tags": mor_tags_to_string(mor, sep=mor_tag_sep),
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


def row_to_events(row: Dict, orig_index: int, mor_tag_sep: str = " ") -> Iterable[Dict]:
    """
    Convert a source row to ordered utterance events.

    If both parent and child are present on the same row, parent is emitted first.
    In this corpus these rows often encode a parent prompt and child production
    on the same line, so parent→child ordering is the safer default.
    """
    if valid_utt(row.get("p_utts")):
        yield make_event(row, "parent", orig_index, mor_tag_sep=mor_tag_sep)
    if valid_utt(row.get("c_utts")):
        yield make_event(row, "child", orig_index, mor_tag_sep=mor_tag_sep)


def pair_child_parent(child_ev: Dict, parent_ev: Dict, subject_meta: Dict[int, Dict]) -> Dict:
    subj_int = subject_to_int(child_ev["subject"])

    meta = subject_meta.get(
        subj_int,
        {
            "lesion_raw": None,
            "lesion_side": None,
            "group": "unknown_subject",
            "has_brain_lesion": 0,
            "subject_active": None,
            "subject_control": None,
        },
    )

    return {
        "subject": subj_int,
        "group": meta["group"],
        "has_brain_lesion": meta["has_brain_lesion"],
        "lesion_side": meta["lesion_side"],
        "lesion_raw": meta["lesion_raw"],
        "subject_active": meta["subject_active"],
        "subject_control": meta["subject_control"],
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
        "child_mor": child_ev["mor"],
        "parent_mor": parent_ev["mor"],
        "child_mor_tags": child_ev["mor_tags"],
        "parent_mor_tags": parent_ev["mor_tags"],
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
    parser.add_argument("--data_path", required=True, help="Path to the large source conversation CSV")
    parser.add_argument("--subjects_path", required=True, help="Path to subjects CSV containing subject IDs and lesion side")
    parser.add_argument("--output_dir", default=".", help="Directory where output CSVs will be written")
    parser.add_argument("--chunksize", type=int, default=100_000)

    parser.add_argument("--subject_id_col", default="id", help="Subject ID column in subjects CSV")
    parser.add_argument("--lesion_col", default="lesion", help="Lesion side column in subjects CSV")
    parser.add_argument("--active_col", default="active", help="Optional active column in subjects CSV")
    parser.add_argument("--control_col", default="control", help="Optional control column in subjects CSV")
    parser.add_argument(
        "--strict_subjects",
        action="store_true",
        help="Fail if a conversation subject is missing from the subjects CSV",
    )
    parser.add_argument(
        "--mor_tag_sep",
        default=" ",
        help="Separator used when saving extracted MOR tags. Default: a single space.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    subject_meta = load_subject_metadata(
        subjects_path=args.subjects_path,
        subject_id_col=args.subject_id_col,
        lesion_col=args.lesion_col,
        active_col=args.active_col,
        control_col=args.control_col,
    )

    out_all = os.path.join(args.output_dir, "conversation_pairs_prepared_all.csv")
    out_lesion = os.path.join(args.output_dir, "conversation_pairs_lesion.csv")
    out_no_lesion = os.path.join(args.output_dir, "conversation_pairs_no_lesion.csv")
    out_left = os.path.join(args.output_dir, "conversation_pairs_lesion_left.csv")
    out_right = os.path.join(args.output_dir, "conversation_pairs_lesion_right.csv")

    # Discover available columns so the script also works if optional columns are missing.
    available_cols = pd.read_csv(args.data_path, nrows=0).columns.tolist()
    usecols = [c for c in BASE_COLS if c in available_cols]
    required = {"subject", "session", "row", "p_utts", "c_utts"}
    missing = required - set(usecols)
    if missing:
        raise ValueError(f"Missing required input columns: {sorted(missing)}")

    if "p_mor" not in available_cols:
        print("WARNING: p_mor column not found; parent_mor and parent_mor_tags will be empty.", flush=True)
    if "c_mor" not in available_cols:
        print("WARNING: c_mor column not found; child_mor and child_mor_tags will be empty.", flush=True)

    # pending_child stores the latest child event whose immediately following event
    # has not been observed yet, separately for each transcript-like group.
    pending_child: Dict[Tuple, Optional[Dict]] = {}

    counts = {
        "all": 0,
        "lesion": 0,
        "no_lesion": 0,
        "lesion_left": 0,
        "lesion_right": 0,
        "unknown_subject": 0,
        "unknown_lesion": 0,
    }
    subjects_seen = {k: set() for k in counts}
    global_row_offset = 0

    f_all, w_all = open_writer(out_all)
    f_lesion, w_lesion = open_writer(out_lesion)
    f_no_lesion, w_no_lesion = open_writer(out_no_lesion)
    f_left, w_left = open_writer(out_left)
    f_right, w_right = open_writer(out_right)

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

                subj = subject_to_int(row.get("subject"))
                if args.strict_subjects and subj is not None and subj not in subject_meta:
                    raise ValueError(
                        f"Subject {subj} from conversation CSV is missing from subjects CSV"
                    )

                for ev in row_to_events(row, orig_index, mor_tag_sep=args.mor_tag_sep):
                    if ev["speaker"] == "child":
                        # If two child utterances occur in a row, the first one has no immediate
                        # parent response. Keep only the latest child event.
                        pending_child[key] = ev
                    else:  # parent event
                        child_ev = pending_child.get(key)
                        if child_ev is not None:
                            pair = pair_child_parent(child_ev, ev, subject_meta)
                            w_all.writerow(pair)
                            counts["all"] += 1
                            subjects_seen["all"].add(pair["subject"])

                            if pair["group"] == "lesion_left":
                                w_lesion.writerow(pair)
                                w_left.writerow(pair)
                                counts["lesion"] += 1
                                counts["lesion_left"] += 1
                                subjects_seen["lesion"].add(pair["subject"])
                                subjects_seen["lesion_left"].add(pair["subject"])

                            elif pair["group"] == "lesion_right":
                                w_lesion.writerow(pair)
                                w_right.writerow(pair)
                                counts["lesion"] += 1
                                counts["lesion_right"] += 1
                                subjects_seen["lesion"].add(pair["subject"])
                                subjects_seen["lesion_right"].add(pair["subject"])

                            elif pair["group"] == "no_lesion":
                                w_no_lesion.writerow(pair)
                                counts["no_lesion"] += 1
                                subjects_seen["no_lesion"].add(pair["subject"])

                            elif pair["group"] == "unknown_subject":
                                counts["unknown_subject"] += 1
                                subjects_seen["unknown_subject"].add(pair["subject"])

                            elif pair["group"] == "unknown_lesion":
                                counts["unknown_lesion"] += 1
                                subjects_seen["unknown_lesion"].add(pair["subject"])

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
        f_no_lesion.close()
        f_left.close()
        f_right.close()

    print("\nDone.")
    print(f"All pairs:          {counts['all']:,} -> {out_all}")
    print(f"Lesion pairs:       {counts['lesion']:,} -> {out_lesion}")
    print(f"No-lesion pairs:    {counts['no_lesion']:,} -> {out_no_lesion}")
    print(f"Left lesion pairs:  {counts['lesion_left']:,} -> {out_left}")
    print(f"Right lesion pairs: {counts['lesion_right']:,} -> {out_right}")

    if counts["unknown_subject"]:
        print(
            f"WARNING: {counts['unknown_subject']:,} pairs had subjects missing from the subjects CSV."
        )
        print("Unknown subjects:", sorted(x for x in subjects_seen["unknown_subject"] if x is not None))

    if counts["unknown_lesion"]:
        print(
            f"WARNING: {counts['unknown_lesion']:,} pairs had unexpected non-empty lesion values."
        )
        print("Unknown-lesion subjects:", sorted(x for x in subjects_seen["unknown_lesion"] if x is not None))

    print("Lesion subjects present:", sorted(x for x in subjects_seen["lesion"] if x is not None))
    print("Left lesion subjects present:", sorted(x for x in subjects_seen["lesion_left"] if x is not None))
    print("Right lesion subjects present:", sorted(x for x in subjects_seen["lesion_right"] if x is not None))
    print("No-lesion subjects present count:", len(subjects_seen["no_lesion"]))


if __name__ == "__main__":
    main()
