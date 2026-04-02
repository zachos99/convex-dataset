import os
import re
import pandas as pd
import csv
from pathlib import Path


def merge_csvs_numerical(folder=".", text_cols=("text", "full_text")):
    """
    Merge twikit "start-end run" CSVs written by `renew_data/tweet_extraction.py`.

    This function is intended ONLY for the case where the user ran tweet extraction
    multiple times with START_ID/END_ID, producing files like:

      notes-data-image-set-new_0-100_tweet_data.csv
      notes-data-image-set-new_100-200_tweet_data.csv

    It will:
    - run separately for image and video
    - merge ONLY those run files (skips per-batch shard/debug files like *_tweet_data_1-145.csv)
    - merge in numerical order by the {start} number (not alphabetical)
    - write:
        notes-data-image-set-new_merged_tweet_data.csv
        notes-data-video-set-new_merged_tweet_data.csv
    """

    def _merge_one_modality(modality):
        # Match run files produced by tweet_extraction.run_tweet_extraction_sync when start/end are set.
        # IMPORTANT: anchor pattern so shard/debug files are excluded.
        run_re = re.compile(
            r"^notes-data-"
            + re.escape(modality)
            + r"-set-new_(\d+)-(\d+)_tweet_data\.csv$"
        )

        files = []
        for fname in os.listdir(folder):
            m = run_re.match(fname)
            if not m:
                continue
            start = int(m.group(1))
            end = int(m.group(2))
            files.append((start, end, fname))

        if not files:
            print(f"[merge_csvs_numerical] No run files found for modality={modality!r} in {folder!r}. Skipping.")
            return None

        files.sort(key=lambda t: (t[0], t[1]))
        ordered_fnames = [t[2] for t in files]

        dfs = []
        for fname in ordered_fnames:
            path = os.path.join(folder, fname)
            df = pd.read_csv(
                path,
                dtype=str,
                keep_default_na=False,
                na_filter=False,
                low_memory=False,
            )

            # Sanitize text columns if present (don't enforce).
            for col in text_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace(r"[\r\n]+", " ", regex=True)

            dfs.append(df)

        merged = pd.concat(dfs, ignore_index=True)
        # Canonical output name (no start-end suffix) so downstream can
        # always consume the same path, whether merging happened or not.
        output_path = os.path.join(
            folder, f"notes-data-{modality}-set-new_tweet_data.csv"
        )
        merged.to_csv(
            output_path,
            index=False,
            encoding="utf-8",
            quoting=csv.QUOTE_MINIMAL,
        )
        print(f"[merge_csvs_numerical] Merged {len(ordered_fnames)} files → {output_path}")
        return output_path

    out_image = _merge_one_modality("image")
    out_video = _merge_one_modality("video")
    return {"image": out_image, "video": out_video}


def validate_media_paths(csv_path, parent_folder):
    """
    Validates that all media paths in a CSV file exist.
    
    Args:
        csv_path (str): Path to the CSV file containing a 'media' column
        parent_folder (str): Parent folder path where media files should be located
        
    Returns:
        dict: A dictionary containing:
            - 'total_rows': Total number of rows checked
            - 'rows_with_media': Number of rows that have media paths
            - 'total_paths': Total number of media paths found
            - 'valid_paths': Number of valid paths
            - 'invalid_paths': Number of invalid paths
            - 'invalid_paths_list': List of tuples (row_index, path) for invalid paths
            - 'all_valid': Boolean indicating if all paths are valid
    """
    # Read the CSV file
    df = pd.read_csv(
        csv_path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        low_memory=False,
    )

    
    
    # Check if 'media' column exists
    if 'media' not in df.columns:
        raise KeyError(f"Column 'media' not found in {csv_path}")
    
    # Initialize counters
    total_rows = len(df)
    print(f"Checking media paths in CSV file: {csv_path} with {total_rows} rows")
    col_to_check = "full_text"  
    successfully_scraped_rows = df[col_to_check].astype(str).str.strip().ne("").sum()
    rows_with_media = 0
    total_paths = 0
    valid_paths = 0
    invalid_paths = 0
    invalid_paths_list = []
    
    # Process each row
    for idx, row in df.iterrows():
        media_value = str(row['media']).strip()
        
        # Skip empty media values
        if not media_value or media_value == 'nan' or media_value == '':
            continue
        
        rows_with_media += 1
        
        # Split by comma to handle multiple paths
        paths = [p.strip() for p in media_value.split(',')]
        
        # Validate each path
        for path in paths:
            if not path or path == 'nan':
                continue
                
            total_paths += 1
            
            # Construct full path
            full_path = os.path.join(parent_folder, path)
            
            # Check if path exists
            if os.path.exists(full_path) and os.path.isfile(full_path):
                valid_paths += 1
            else:
                invalid_paths += 1
                invalid_paths_list.append((idx + 2, path))  # +2 because CSV rows are 1-indexed and header is row 1
    
    # Prepare result
    result = {
        'total_rows': total_rows,
        'rows_with_media': rows_with_media,
        'total_paths': total_paths,
        'valid_paths': valid_paths,
        'invalid_paths': invalid_paths,
        'invalid_paths_list': invalid_paths_list,
        'all_valid': invalid_paths == 0
    }
    
    # Pretty print the results
    print("\n" + "="*70)
    print("MEDIA PATH VALIDATION RESULTS")
    print("="*70)
    print(f"CSV File:           {csv_path}")
    print(f"Parent Folder:      {parent_folder}")
    print("-"*70)
    print(f"Total Rows:                 {total_rows:,}")
    print(f"Successfully Scraped Rows:  {successfully_scraped_rows:,}")
    print(f"Rows with Media:            {rows_with_media:,}")
    print(f"Total Media Paths:          {total_paths:,}")
    print(f"Valid Paths:                {valid_paths:,} ({100*valid_paths/total_paths:.1f}%)" if total_paths > 0 else "Valid Paths:        0 (N/A)")
    print(f"Invalid Paths:              {invalid_paths:,} ({100*invalid_paths/total_paths:.1f}%)" if total_paths > 0 else "Invalid Paths:      0 (N/A)")
    print("-"*70)
    
    if invalid_paths == 0:
        print("✓ ALL PATHS ARE VALID")
    else:
        print(f"✗ {invalid_paths} INVALID PATH(S) FOUND")
        print("\nInvalid Paths (showing first 20):")
        print("-"*70)
        for i, (row_idx, path) in enumerate(invalid_paths_list[:20], 1):
            full_path = os.path.join(parent_folder, path)
            print(f"  {i:2d}. Row {row_idx:6d}: {path}")
            print(f"      Full path: {full_path}")
        if len(invalid_paths_list) > 20:
            print(f"\n  ... and {len(invalid_paths_list) - 20} more invalid paths")
    
    print("="*70 + "\n")
    
    return result


"""
def fix_media_extensions_and_paths(
    csv_path,
    base_dir=".",
    media_col="media",
    out_suffix="_fixed",
    allowed_formats=("jpeg", "png", "webp", "gif", "bmp", "tiff"),
    dry_run=False,
):
    
    # For each media path in a CSV 'media' column:
    #   - Detect the real image format from file bytes
    #   - If extension mismatches, rename file to correct extension
    #   - Update the CSV media path accordingly
    # Saves a new CSV: <stem>_fixed.csv
    

    csv_path = Path(csv_path)
    base_dir = Path(base_dir)

    df = pd.read_csv(
        csv_path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        low_memory=False,
    )

    if media_col not in df.columns:
        raise KeyError(f"Column '{media_col}' not found in {csv_path}")

    # ---------- format detection ----------
    def detect_format(abs_path: Path):
        try:
            from PIL import Image
            with Image.open(abs_path) as im:
                fmt = (im.format or "").lower()
        except Exception:
            fmt = ""

        if not fmt:
            try:
                import imghdr
                fmt = (imghdr.what(abs_path) or "").lower()
            except Exception:
                fmt = ""

        if not fmt:
            return None

        if fmt == "jpeg":
            return "jpg"
        if fmt in ("tif", "tiff"):
            return "tiff"
        return fmt

    # ---------- safe rename ----------
    def safe_rename(old_abs: Path, new_abs: Path):
        if new_abs.exists():
            if old_abs.resolve() == new_abs.resolve():
                return True, "already_same"
            return False, "collision_target_exists"

        if dry_run:
            return True, "dry_run"

        try:
            old_abs.rename(new_abs)
            return True, "renamed"
        except Exception as e:
            return False, f"rename_failed:{e}"

    # ---------- counters ----------
    examined = renamed = updated_paths = 0
    missing_files = unreadable = collisions = skipped_unknown_format = 0
    issues = []

    rename_map = {}  # old_rel_posix -> new_rel_posix

    # ---------- main ----------
    def repair_cell(cell_val, row_idx):
        nonlocal examined, renamed, updated_paths
        nonlocal missing_files, unreadable, collisions, skipped_unknown_format

        s = str(cell_val).strip()
        if not s:
            return cell_val

        parts = [p.strip() for p in s.split(",") if p.strip()]
        new_parts = []

        for rel_str in parts:
            rel_path = Path(rel_str)
            key = rel_path.as_posix()

            # 0) rewrite via earlier rename
            if key in rename_map:
                new_parts.append(rename_map[key])
                updated_paths += 1
                examined += 1
                continue

            abs_path = base_dir / rel_path
            examined += 1

            # 1) missing → try alternate extensions
            if not abs_path.exists() or not abs_path.is_file():
                stem = abs_path.with_suffix("")
                found = None
                for ext in (".png", ".jpg", ".jpeg", ".webp"):
                    cand = stem.with_suffix(ext)
                    if cand.exists() and cand.is_file():
                        found = cand.relative_to(base_dir).as_posix()
                        break

                if found:
                    rename_map[key] = found
                    new_parts.append(found)
                    updated_paths += 1
                    continue

                missing_files += 1
                issues.append({"row": row_idx + 2, "path": rel_str, "issue": "missing_file"})
                new_parts.append(rel_str)
                continue

            # 2) detect format
            fmt = detect_format(abs_path)
            if fmt is None:
                unreadable += 1
                issues.append({"row": row_idx + 2, "path": rel_str, "issue": "unreadable_or_unknown_format"})
                new_parts.append(rel_str)
                continue

            if fmt not in allowed_formats and fmt != "jpg":
                skipped_unknown_format += 1
                issues.append({"row": row_idx + 2, "path": rel_str, "issue": f"format_not_allowed:{fmt}"})
                new_parts.append(rel_str)
                continue

            current_ext = rel_path.suffix.lower().lstrip(".")
            if current_ext == "jpeg":
                current_ext = "jpg"

            # 3) correct extension already
            if current_ext == fmt:
                new_parts.append(key)
                continue

            # 4) rename file
            new_rel = rel_path.with_suffix("." + fmt)
            new_abs = base_dir / new_rel

            ok, status = safe_rename(abs_path, new_abs)
            if not ok:
                if status == "collision_target_exists":
                    collisions += 1
                else:
                    unreadable += 1
                issues.append({
                    "row": row_idx + 2,
                    "path": rel_str,
                    "issue": status,
                    "detected_format": fmt,
                    "suggested_path": new_rel.as_posix(),
                })
                new_parts.append(key)
                continue

            if status in ("renamed", "dry_run"):
                renamed += 1

            new_posix = new_rel.as_posix()
            rename_map[key] = new_posix
            new_parts.append(new_posix)
            updated_paths += 1

        return ", ".join(new_parts)

    df[media_col] = [repair_cell(v, i) for i, v in enumerate(df[media_col].tolist())]

    # ---------- report ----------
    print("—— Media Fix Summary ——")
    print(f"Rows processed:      {len(df):,}")
    print(f"Media refs examined: {examined:,}")
    print(f"Files renamed:       {renamed:,}")
    print(f"CSV paths updated:   {updated_paths:,}")
    print(f"Missing files:       {missing_files:,}")
    print(f"Unreadable/unknown:  {unreadable:,}")
    print(f"Collisions:          {collisions:,}")
    print(f"Skipped formats:     {skipped_unknown_format:,}")
    print(f"Issues total:        {len(issues):,}")

    if issues:
        print("\nFirst issues (up to 10):")
        for it in issues[:10]:
            print(f"  Row {it['row']}: {it['issue']} — {it['path']}")

    if not dry_run:
        out_path = csv_path.with_name(csv_path.stem + out_suffix + ".csv")
        df.to_csv(out_path, index=False)
        print(f"\n✅ Saved fixed CSV → {out_path}")

    return out_path

"""



def merge_notes_with_tweet_data_one(
    fixed_path,
    notes_path,
    keys,  # "image" or "video"
    out_filtered_path = None,
    media_col = "media",
    full_text_col = "full_text",
    tweet_id_col = "tweet_id",
    created_col = "created_at_datetime",
):
    """
    - Reads tweets + notes
    - Early check: duplicated tweet_ids with mixed empty/non-empty full_text (optionally saves those rows)
    - Prints tweet emptiness stats
    - Dedupes tweets on tweet_id, preferring non-empty full_text (ONLY criterion)
    - Merges notes (LEFT) with deduped tweets; enforces merged rows == notes rows
    - Filters merged based on keys:
        image -> require non-empty media
        video -> require non-empty full_text
    - Diagnostics after filtering (incl. duplicate noteId check)
    - Saves ONLY the filtered df
    """

    if keys not in {"image", "video"}:
        raise ValueError("keys must be either 'image' or 'video'")

    def is_empty(s):
        return s.isna() | (s.astype(str).str.strip() == "")

    # ------------------ read tweets ------------------
    tweets = pd.read_csv(
        fixed_path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        low_memory=False,
    )

    req_tweet = {tweet_id_col, media_col, full_text_col, created_col}
    miss = req_tweet - set(tweets.columns)
    if miss:
        raise KeyError(f"Tweet CSV missing required columns: {sorted(miss)}")

    #  EARLY duplicate tweet_id full_text consistency check 
    dup_mask = tweets[tweet_id_col].duplicated(keep=False)
    mixed_ids = []
    if dup_mask.any():
        ft_empty = is_empty(tweets[full_text_col])
        ft_mixed = (
            tweets.loc[dup_mask]
            .assign(_ft_empty=ft_empty[dup_mask])
            .groupby(tweet_id_col)["_ft_empty"]
            .nunique()
        )
        mixed_ids = ft_mixed[ft_mixed > 1].index.tolist()
        
        print("\n—— Tweet DF: empty-field stats (before dedupe/merge) ——")
        print(f"Duplicated tweet_ids: {tweets.loc[dup_mask, tweet_id_col].nunique():,}")
        

    # tweet emptiness stats
    n_tweets = len(tweets)
    empty_media = is_empty(tweets[media_col])
    empty_full = is_empty(tweets[full_text_col])
    empty_created = is_empty(tweets[created_col])
    empty_both = empty_media & empty_full

   
    print(f"Total rows: {n_tweets:,}")
    print(f"Empty {media_col:<20} {int(empty_media.sum()):,} ({100*empty_media.mean():.1f}%)")
    print(f"Empty {full_text_col:<20} {int(empty_full.sum()):,} ({100*empty_full.mean():.1f}%)")
    print(f"Empty {created_col:<20} {int(empty_created.sum()):,} ({100*empty_created.mean():.1f}%)")
    # print(f"Empty BOTH {full_text_col} AND {media_col:<7} {int(empty_both.sum()):,} ({100*empty_both.mean():.1f}%)")
    # print()

    #  dedupe tweets to avoid m*n (prefer non-empty full_text only) 
    tweets["_pref_full_text"] = (~is_empty(tweets[full_text_col])).astype(int)

    # Stable sort: within each tweet_id, put non-empty full_text first, then keep first row
    tweets_sorted = tweets.sort_values(
        by=[tweet_id_col, "_pref_full_text"],
        ascending=[True, False],
        kind="mergesort",
    )

    tweets_one = tweets_sorted.drop_duplicates(subset=[tweet_id_col], keep="first").copy()
    tweets_one = tweets_one.drop(columns=["_pref_full_text"], errors="ignore")

    # ------------------ read notes ------------------
    notes = pd.read_csv(
        notes_path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        low_memory=False,
    )

    req_notes = {"tweetId", "noteId", "noteText", "createdAtMillis"}
    missn = req_notes - set(notes.columns)
    if missn:
        raise KeyError(f"Notes CSV missing required columns: {sorted(missn)}")

    notes_keep = notes[["tweetId", "noteId", "noteText", "createdAtMillis"]].copy()
    notes_keep = notes_keep.rename(columns={"tweetId": tweet_id_col, "createdAtMillis": "noteDate"})

    # ------------------ merge (notes LEFT) ------------------
    merged = notes_keep.merge(tweets_one, on=tweet_id_col, how="left")

    # Step 4: enforce merged rows == notes rows
    if len(merged) != len(notes_keep):
        print("❌ ERROR: Merge produced unexpected row count (many-to-many merge likely).")
        print(f"Notes rows:  {len(notes_keep):,}")
        print(f"Merged rows: {len(merged):,}")
        raise RuntimeError("Row count mismatch after merge. tweet_id was expected to be unique after dedupe.")

    # Reorder columns: tweet columns first, then note columns
    note_cols = ["noteId", "noteText", "noteDate"]
    tweet_cols = [c for c in tweets_one.columns if c != tweet_id_col]
    merged = merged[[tweet_id_col] + tweet_cols + note_cols]

    # ------------------ filter AFTER merge ------------------
    keycol = media_col if keys == "image" else full_text_col
    key_empty = is_empty(merged[keycol])
    filtered = merged[~key_empty].copy()

    # ------------------ diagnostics AFTER filtering ------------------
    print("—— Post-merge filtering diagnostics ——")
    print(f"Notes rows (input):           {len(notes_keep):,}")
    print(f"Merged rows (pre-filter):     {len(merged):,} ✓ (matches notes)")
    print(f"Filtered rows kept ({keys}):   {len(filtered):,}")
    print(f"Filtered rows dropped:        {len(merged) - len(filtered):,} ({100*(1 - len(filtered)/len(merged)):.2f}%)")

    dup_note_ids = int(filtered["noteId"].duplicated().sum()) if len(filtered) else 0
    if len(filtered):
        print(f"Duplicate noteId rows (filtered): {dup_note_ids:,} ({100*dup_note_ids/len(filtered):.2f}%)")
    else:
        print("Duplicate noteId rows (filtered): N/A (0 rows)")

    still_empty = int(is_empty(filtered[keycol]).sum()) if len(filtered) else 0
    print(f"Empty {keycol} in filtered:   {still_empty:,}")

    if created_col in filtered.columns and len(filtered):
        ec = int(is_empty(filtered[created_col]).sum())
        print(f"Empty {created_col} in filtered: {ec:,} ({100*ec/len(filtered):.2f}%)")
    print("—— End diagnostics ——\n")

    # ------------------ save ONLY filtered ------------------
    if out_filtered_path is None:
        out_filtered_path = str(Path(notes_path).with_suffix("").as_posix()) + f"_with_tweets_{keys}_filtered.csv"

    filtered.to_csv(out_filtered_path, index=False)
    print(f"✅ Saved filtered df → {out_filtered_path}")

    return filtered


def merge_notes_with_tweet_data(
    fixed_image_path,
    fixed_video_path,
    notes_image_path,
    notes_video_path,
):
    """
    Merge notes + tweet/media for BOTH modalities (image + video).

    Inputs:
      - `fixed_image_path`, `fixed_video_path`: tweet CSV paths
      - `notes_image_path`, `notes_video_path`: notes CSV paths

    Output (per modality):
      - `notes-tweets-data-image-set-new.csv`
      - `notes-tweets-data-video-set-new.csv`

    Filtering is handled by the underlying `merge_notes_with_tweet_data_one()`:
      - image: keeps only rows where `media` is non-empty
      - video: keeps only rows where `full_text` is non-empty
    """

    out_image = Path(notes_image_path).parent / "notes-tweets-data-image-set-new.csv"
    out_video = Path(notes_video_path).parent / "notes-tweets-data-video-set-new.csv"

    merge_notes_with_tweet_data_one(
        fixed_path=fixed_image_path,
        notes_path=notes_image_path,
        keys="image",
        out_filtered_path=str(out_image),
    )

    merge_notes_with_tweet_data_one(
        fixed_path=fixed_video_path,
        notes_path=notes_video_path,
        keys="video",
        out_filtered_path=str(out_video),
    )

    return {"image": str(out_image), "video": str(out_video)}








"""
Example usage:

If you ran `tweet_extraction.py` multiple times with START_ID/END_ID, you will have
multiple "run CSVs" in the current folder, e.g.:

  notes-data-image-set-new_0-100_tweet_data.csv
  notes-data-image-set-new_100-200_tweet_data.csv

and similarly for video.

In that case, call `merge_csvs_numerical(folder=".")` to merge runs separately for
image and video (it intentionally skips per-batch shard/debug files like
`*_tweet_data_1-145.csv`).
"""


if __name__ == "__main__":
    # Ensure `folder="."` refers to this file's directory (renew_data/).
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    """
        It produces files:
        - notes-data-{image/video}-set-new_tweet_data.csv
    """
    merged_paths = merge_csvs_numerical(folder=".")
    print(f"Image path: {merged_paths['image']}")
    print(f"Video path: {merged_paths['video']}")

    validate_media_paths(merged_paths['image'], "tweet_media/image_set")
    validate_media_paths(merged_paths['video'], "tweet_media/video_set")

    notes_image_csv = "notes-data-image-set-new.csv"
    notes_video_csv = "notes-data-video-set-new.csv"

    merged_paths_with_tweets = merge_notes_with_tweet_data(
        fixed_image_path=merged_paths['image'],
        fixed_video_path=merged_paths['video'],
        notes_image_path=notes_image_csv,
        notes_video_path=notes_video_csv,
    )








