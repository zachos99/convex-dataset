import pandas as pd
import os
from pathlib import Path


def _load(src):
    """Accept either a file path or an already-loaded DataFrame."""
    if isinstance(src, pd.DataFrame):
        return src.copy()
    return pd.read_csv(src, dtype=str, low_memory=False)


def check_csv_merge_safety(
    left_csv,
    right_csv,
    note_id_col="noteId",
    tweet_id_col="tweet_id",
    max_rows=None,
):
    print("\n" + "=" * 80)
    print("CSV MERGE SAFETY REPORT (pandas)")
    print("=" * 80)

    left  = _load(left_csv)
    right = _load(right_csv)

    if max_rows:
        left  = left.head(max_rows)
        right = right.head(max_rows)

    def section(title):
        print("\n" + title)
        print("-" * len(title))

    # --- FILE OVERVIEW ---
    section("FILES")

    for name, df, path in [
        ("LEFT", left, left_csv),
        ("RIGHT", right, right_csv),
    ]:
        print(f"\n{name}:")
        print(f"  path: {path}")
        print(f"  rows: {len(df)}")
        print(f"  columns: {len(df.columns)}")
        print(f"  has noteId: {note_id_col in df.columns}")
        print(f"  has tweet_id: {tweet_id_col in df.columns}")

    # --- SCHEMA CHECK ---
    section("SCHEMA CHECK")

    same_order = list(left.columns) == list(right.columns)
    same_set = set(left.columns) == set(right.columns)

    print(f"  same columns & order: {same_order}")
    print(f"  same columns (ignore order): {same_set}")

    if not same_set:
        only_left = sorted(set(left.columns) - set(right.columns))
        only_right = sorted(set(right.columns) - set(left.columns))
        print("\n  column differences:")
        if only_left:
            print(f"    only in LEFT: {only_left}")
        if only_right:
            print(f"    only in RIGHT: {only_right}")

    # --- noteId CHECKS ---
    section("NOTE ID CHECKS")

    if note_id_col in left.columns and note_id_col in right.columns:
        left_note = left[note_id_col].dropna().astype(str).str.strip()
        right_note = right[note_id_col].dropna().astype(str).str.strip()

        left_note = left_note[left_note != ""]
        right_note = right_note[right_note != ""]

        left_dups = left_note.duplicated().sum()
        right_dups = right_note.duplicated().sum()
        overlap_note = set(left_note) & set(right_note)

        print(f"  LEFT duplicated noteIds: {left_dups}")
        print(f"  RIGHT duplicated noteIds: {right_dups}")
        print(f"  common noteIds across files: {len(overlap_note)}")

        if overlap_note:
            print(f"    sample: {sorted(overlap_note)[:20]}")
    else:
        print("  noteId column missing in one of the files")

    # --- tweet_id CHECKS (informational) ---
    section("TWEET ID CHECKS (informational)")

    if tweet_id_col in left.columns and tweet_id_col in right.columns:
        left_tweet = left[tweet_id_col].dropna().astype(str).str.strip()
        right_tweet = right[tweet_id_col].dropna().astype(str).str.strip()

        left_tweet = left_tweet[left_tweet != ""]
        right_tweet = right_tweet[right_tweet != ""]

        overlap_tweet = set(left_tweet) & set(right_tweet)

        print(f"  LEFT unique tweet_ids: {left_tweet.nunique()}")
        print(f"  RIGHT unique tweet_ids: {right_tweet.nunique()}")
        print(f"  common tweet_ids: {len(overlap_tweet)}")

        if overlap_tweet:
            print(f"    sample: {sorted(overlap_tweet)[:20]}")
    else:
        print("  tweet_id column missing in one of the files")

    # --- FINAL VERDICT ---
    section("FINAL VERDICT")

    safe_concat = same_order and (note_id_col in left.columns) and (note_id_col in right.columns)

    if safe_concat:
        print("  ✔ SAFE for row-wise concat (same schema & order)")
    else:
        print("  ✖ NOT SAFE for row-wise concat")

    print("=" * 80 + "\n")

    return {
        "same_columns_same_order": same_order,
        "same_columns_ignore_order": same_set,
        "noteId_overlap": len(overlap_note) if note_id_col in left.columns and note_id_col in right.columns else None,
        "tweet_id_overlap": len(overlap_tweet) if tweet_id_col in left.columns and tweet_id_col in right.columns else None,
        "safe_to_concat": safe_concat,
    }

def align_new_data_for_integration(existing_dataset_path, new_data_path):
    """
    Align the new data for integration with the existing dataset.
    Returns the aligned DataFrame (does not save to disk).

    Steps:
      1) Rewrite iteration_id to continue from max(existing)
      2) Drop confidence_rerun if present
      3) Add empty misinfo_type if missing
      4) Reorder columns to exactly match the existing dataset (raises if schema differs)
    """
    old = _load(existing_dataset_path)
    new = _load(new_data_path)

    # 1) Rewrite iteration_id
    if "iteration_id" not in old.columns or "iteration_id" not in new.columns:
        raise ValueError("iteration_id column missing in one of the datasets")

    old_ids = pd.to_numeric(old["iteration_id"], errors="coerce")
    if old_ids.isna().any():
        raise ValueError("Found non-numeric iteration_id values in existing dataset")

    max_old_id = int(old_ids.max())
    n_new = len(new)
    new_start = max_old_id + 1
    new["iteration_id"] = [str(i) for i in range(new_start, new_start + n_new)]

    print(f"Rewriting iteration_id in new data:")
    print(f"  max old iteration_id = {max_old_id}")
    print(f"  new iteration_id range = [{new_start}, {new_start + n_new - 1}]")

    # 2) Drop confidence_rerun if present
    if "confidence_rerun" in new.columns:
        new = new.drop(columns=["confidence_rerun"])

    # 3) Add empty misinfo_type if missing
    if "misinfo_type" not in new.columns:
        new["misinfo_type"] = ""

    # 4) Reorder columns to match existing.
    # Columns in old but not in new → add as empty (warn).
    # Columns in new but not in old → unexpected, raise.
    missing_in_new = [c for c in old.columns if c not in new.columns]
    extra_in_new   = [c for c in new.columns if c not in old.columns]

    if missing_in_new:
        print(f"Warning: columns present in existing dataset but missing in new data — "
              f"adding as empty: {missing_in_new}")
        for col in missing_in_new:
            new[col] = ""

    if extra_in_new:
        raise ValueError(
            f"Schema mismatch: new data has unexpected columns not in existing dataset: {extra_in_new}"
        )

    new = new[old.columns]
    print("Columns aligned to existing dataset.")
    return new

def append_new_data_to_existing_dataset(existing_dataset_path, new_data_aligned, out_merged_path):
    """
    Concat existing dataset with the aligned new data and save to out_merged_path.
    new_data_aligned can be a file path or an already-loaded/aligned DataFrame.
    """
    old    = _load(existing_dataset_path)
    new    = _load(new_data_aligned)
    merged = pd.concat([old, new], ignore_index=True)
    merged.to_csv(out_merged_path, index=False)
    print("Merged saved:", out_merged_path)
    print("Rows old:", len(old))
    print("Rows new:", len(new))
    print("Rows merged:", len(merged))






def run_integration_pipeline(
    image_new_path,
    video_new_path,
    image_existing_path,
    video_existing_path,
):
    """
    Run the full integration pipeline for both modalities.

    For each modality:
      1) Align new data to existing dataset schema (iteration_id, column order, …)
      2) Safety-check the merge
      3) Append and save next to the existing dataset file as *_NEW.csv

    Returns a dict with the output paths for image and video.
    """
    results = {}

    pairs = [
        ("image", image_new_path,  image_existing_path),
        ("video", video_new_path,  video_existing_path),
    ]

    for modality, new_path, existing_path in pairs:
        print(f"\n{'='*60}")
        print(f"  Integration pipeline — {modality}")
        print(f"{'='*60}")

        # 1) Align
        aligned_df = align_new_data_for_integration(existing_path, new_path)

        # 2) Safety check
        safety = check_csv_merge_safety(existing_path, aligned_df)
        if not safety["safe_to_concat"]:
            raise RuntimeError(
                f"[{modality}] Merge safety check failed — aborting. "
                f"Review the report above before proceeding."
            )

        # 3) Append
        out_path = str(Path(existing_path).with_suffix("")) + "_NEW.csv"
        append_new_data_to_existing_dataset(existing_path, aligned_df, out_path)

        results[modality] = out_path

    return results


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    IMAGE_NEW      = "notes-tweets-data-image-set-new_misinfo_final.csv"
    VIDEO_NEW      = "notes-tweets-data-video-set-new_misinfo_final.csv"
    IMAGE_EXISTING = "dataset-image-set.csv"
    VIDEO_EXISTING = "dataset-video-set.csv"

    outputs = run_integration_pipeline(
        image_new_path=IMAGE_NEW,
        video_new_path=VIDEO_NEW,
        image_existing_path=IMAGE_EXISTING,
        video_existing_path=VIDEO_EXISTING,
    )

    print("\n=== Integration outputs ===")
    print("Image:", outputs["image"])
    print("Video:", outputs["video"])





