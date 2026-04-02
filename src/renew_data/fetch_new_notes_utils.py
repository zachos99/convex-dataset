import pandas as pd
import os
import re
from datetime import datetime

def process_tsv(input_path):
    """
    Reads a TSV file, keeps only specific columns and returns DataFrame.
    """
    try:
        header = pd.read_csv(input_path, sep="\t", nrows=0)
        all_columns = [c.strip() for c in header.columns.tolist()]

        # Prefer selecting by name (stable across TSV schema changes).
        # If some columns are missing, we fall back to reading all columns.
        desired = [
            "noteId",
            "createdAtMillis",
            "tweetId",
            "classification",
            "trustworthySources",
            "isMediaNote",
            "summary",
            # misleading / not misleading tag columns are also useful (combine_tags drops them later)
            "misleadingOther",
            "misleadingFactualError",
            "misleadingManipulatedMedia",
            "misleadingOutdatedInformation",
            "misleadingMissingImportantContext",
            "misleadingUnverifiedClaimAsFact",
            "misleadingSatire",
            "notMisleadingOther",
            "notMisleadingFactuallyCorrect",
            "notMisleadingOutdatedButNotWhenWritten",
            "notMisleadingClearlySatire",
            "notMisleadingPersonalOpinion",
        ]

        available = set(all_columns)
        usecols = [c for c in desired if c in available]
        if not usecols:
            # If schema is unexpected, still allow pipeline to attempt running.
            df = pd.read_csv(input_path, sep="\t", dtype={"tweetId": str}, low_memory=False)
        else:
            df = pd.read_csv(input_path, sep="\t", dtype={"tweetId": str}, usecols=usecols, low_memory=False)

        df.columns = df.columns.str.strip()
        return df

    except Exception as e:
        print(f"❌ Error: {e}")
        return None



def searchMediaKeywords(file_or_df, media_type='image'):
    """
        Filters Community Notes based on keywords indicating image or video cases
    """

    media_keywords_pattern_images = re.compile(
        r"\b(?:photo|photos|image|images|photograph|photographs|screenshot|screenshots|pic|pics|photoshop|photoshopped|picture|pictures|snapshot|snapshots|visual|visuals|jpg|graphic|graphics|thumbnail|thumbnails|logo|logos|png|jpeg)\b",
        flags=re.IGNORECASE
    ) 
    
    media_keywords_pattern_videos = re.compile(
        r"\b(?:video|videos|clip|footage|deepfake|deepfakes|recording)\b",
        flags=re.IGNORECASE
    ) 

    # Accept either file path or DataFrame
    if isinstance(file_or_df, str):
        df = pd.read_csv(file_or_df, dtype=str)
    else:
        df = file_or_df.copy()
    
    df.columns = df.columns.str.strip()
    df["summary"] = df["summary"].fillna("")

    # First filter: contains media-related keywords based on media_type
    if media_type == 'image':
        media_mask = df["summary"].str.contains(media_keywords_pattern_images)
    elif media_type == 'video':
        media_mask = df["summary"].str.contains(media_keywords_pattern_videos)
    else:
        raise ValueError(f"media_type must be 'image' or 'video', got '{media_type}'")

    # (OPTIONAL) Second filter: classification == MISINFORMED_OR_POTENTIALLY_MISLEADING
    classification_mask = df["classification"].str.upper() == "MISINFORMED_OR_POTENTIALLY_MISLEADING"

    # Apply both filters
    df_filtered = df[media_mask & classification_mask].copy()


    # Fix tweetID and convert to string type (can do it also for notes IDs)
    df_filtered["tweetId"] = df_filtered["tweetId"].astype(str).str.replace(r"\.0$", "", regex=True)

    # You can also build the URLs of tweets using their ID 
    df_filtered["tweetUrl"] = "https://x.com/i/web/status/" + df_filtered["tweetId"]

    # You can convert tweet creation time from ms to datetime 
    df_filtered["createdAtMillis"] = pd.to_datetime(df_filtered["createdAtMillis"], unit='ms', utc=True) 

    return df_filtered



def combine_tags(df):

    """
        Combine misleading and not misleading tags in two columns to make it more clear 
    """
    def combine_tags(row, cols):
        return [col for col in cols if row.get(col) == 1]

    misleading_cols = [
        "misleadingOther", "misleadingFactualError", "misleadingManipulatedMedia", 
        "misleadingOutdatedInformation", "misleadingMissingImportantContext", 
        "misleadingUnverifiedClaimAsFact", "misleadingSatire"
    ]

    not_misleading_cols = [
        "notMisleadingOther", "notMisleadingFactuallyCorrect", 
        "notMisleadingOutdatedButNotWhenWritten", "notMisleadingClearlySatire", 
        "notMisleadingPersonalOpinion"
    ]

    df["misleadingTags"] = df.apply(lambda row: combine_tags(row, misleading_cols), axis=1)
    df["notMisleadingTags"] = df.apply(lambda row: combine_tags(row, not_misleading_cols), axis=1)

    df.drop(columns=misleading_cols + not_misleading_cols, inplace=True)

    return df




def find_new_notes(
    df_old,
    df_new,
    out_path,
    save=False,
):
    print()
    print("=" * 50)
    print("COMPARING NOTE CHECKPOINTS")
    print("=" * 50)

    df_old = df_old.copy()
    df_new = df_new.copy()

    # ---------- CSV SIZES ----------
    print("\nCSV SIZES:")
    print(f" Old CSV: {len(df_old):,} rows")
    print(f" New CSV: {len(df_new):,} rows")
    print(f" Difference: {len(df_new) - len(df_old):+,} rows")

    """
    # ---------- DUPLICATE ANALYSIS ----------
    print("\nDUPLICATE ANALYSIS:")

    tweet_id_col_old = "tweetId" if "tweetId" in df_old.columns else ("tweet_id" if "tweet_id" in df_old.columns else None)
    tweet_id_col_new = "tweetId" if "tweetId" in df_new.columns else ("tweet_id" if "tweet_id" in df_new.columns else None)

    if tweet_id_col_old:
        print(f"   Old CSV {tweet_id_col_old} duplicates: {df_old[tweet_id_col_old].duplicated().sum():,}")
    else:
        print("   Old CSV: No tweetId/tweet_id column found")

    if tweet_id_col_new:
        print(f"   New CSV {tweet_id_col_new} duplicates: {df_new[tweet_id_col_new].duplicated().sum():,}")
    else:
        print("   New CSV: No tweetId/tweet_id column found")

    old_noteid_dups = df_old["noteId"].duplicated().sum()
    new_noteid_dups = df_new["noteId"].duplicated().sum()
    if old_noteid_dups > 0 or new_noteid_dups > 0:
        print(f"\n      WARNING: noteId duplicates detected!")
        print(f"      Old CSV noteId duplicates: {old_noteid_dups:,}")
        print(f"      New CSV noteId duplicates: {new_noteid_dups:,}")
    """
    # ---------- NOTE ID COMPARISON ----------
    print("\nNOTE ID COMPARISON:")
    print(f" Old CSV: {len(df_old):,} rows, {df_old['noteId'].nunique(dropna=True):,} unique noteIds")
    print(f" New CSV: {len(df_new):,} rows, {df_new['noteId'].nunique(dropna=True):,} unique noteIds")

    # Clean noteIds
    old_ids = set(df_old["noteId"].astype(str).str.strip())
    new_ids = set(df_new["noteId"].astype(str).str.strip())
    old_ids.discard("nan")
    new_ids.discard("nan")

    new_by_id = new_ids - old_ids

    # ---------- DATE ANALYSIS ----------
    print("\nDATE ANALYSIS:")

    # Parse old checkpoint timestamp (prefer createdAtMillis if present; else noteDate)
    latest_old_datetime = None

    if "createdAtMillis" in df_old.columns:
        if pd.api.types.is_numeric_dtype(df_old["createdAtMillis"]):
            old_ts = pd.to_datetime(df_old["createdAtMillis"], unit="ms", errors="coerce", utc=True)
        else:
            old_ts = pd.to_datetime(df_old["createdAtMillis"], errors="coerce", utc=True)
        if old_ts.notna().any():
            latest_old_datetime = old_ts.max()
            print(f" Old CSV latest entry (createdAtMillis): {latest_old_datetime.strftime('%Y-%m-%d %H:%M:%S.%f+00:00')}")
        else:
            print(" Old CSV latest entry (createdAtMillis): could not parse any values")
    elif "noteDate" in df_old.columns:
        old_ts = pd.to_datetime(df_old["noteDate"], errors="coerce", utc=True)
        if old_ts.notna().any():
            latest_old_datetime = old_ts.max()
            # show the raw value at max too (optional)
            idx = old_ts.idxmax()
            raw = df_old.loc[idx, "noteDate"]
            print(f" Old CSV latest entry (noteDate): {raw} (parsed: {latest_old_datetime.strftime('%B %d,%Y')})")
        else:
            print(" Old CSV latest entry (noteDate): could not parse any values")
    else:
        print(" Old CSV: No date column found (checked createdAtMillis, noteDate)")

    # Parse new createdAtMillis
    if "createdAtMillis" not in df_new.columns:
        print(" New CSV: No createdAtMillis column found")
        print("=" * 50)
        return None

    if pd.api.types.is_numeric_dtype(df_new["createdAtMillis"]):
        new_ts = pd.to_datetime(df_new["createdAtMillis"], unit="ms", errors="coerce", utc=True)
    else:
        new_ts = pd.to_datetime(df_new["createdAtMillis"], errors="coerce", utc=True)

    df_new["_createdAtMillis_parsed"] = new_ts

    if new_ts.notna().any():
        newest_new = new_ts.max()
        print(f" New CSV newest entry (createdAtMillis): {newest_new.strftime('%Y-%m-%d %H:%M:%S.%f+00:00')}")
    else:
        print(" New CSV newest entry (createdAtMillis): could not parse any values")

    # Parsing coverage
    # n_total = len(df_new)
    # n_parsed = new_ts.notna().sum()
    # print(f"\n  createdAtMillis parsing coverage: {n_parsed:,}/{n_total:,} ({n_parsed/n_total:.2%}) parsed, {(n_total-n_parsed):,} NaT")

    # Compute incremental set WITH inclusive boundary (>=)
    if latest_old_datetime is not None and pd.notna(latest_old_datetime):
        newer_or_equal_mask = df_new["_createdAtMillis_parsed"] >= latest_old_datetime  # inclusive
        newer_or_equal_count = newer_or_equal_mask.sum()
        print(f"\nNotes in new CSV newer OR EQUAL to latest old entry: {newer_or_equal_count:,}")

        ids_newer_or_equal = set(df_new.loc[newer_or_equal_mask, "noteId"].astype(str).str.strip())
        ids_newer_or_equal.discard("nan")

        ids_to_process = new_by_id & ids_newer_or_equal
        print(f"(new noteIds) ∩ (newer-or-equal-to-checkpoint): {len(ids_to_process):,}")
    else:
        ids_to_process = new_by_id
        print("\n   Could not compute checkpoint time; defaulting to (new noteIds) only.")
        print(f"   new noteIds (new − old): {len(ids_to_process):,}")

    # ---------- INTERSECTION INSIGHTS (ID ONLY) ----------
    print("\nNOTE ID INTERSECTION INSIGHTS:")
    common = old_ids & new_ids
    old_only = old_ids - new_ids
    print(f" Common noteIds (old ∩ new): {len(common):,} ({len(common)/len(old_ids):.2%} of old, {len(common)/len(new_ids):.2%} of new)")
    print(f" New-only noteIds (new − old): {len(new_by_id):,} ({len(new_by_id)/len(new_ids):.2%} of new)")
    print(f" Old-only noteIds (old − new): {len(old_only):,} ({len(old_only)/len(old_ids):.2%} of old)")

    # Extract rows to save/return
    df_out = df_new[df_new["noteId"].astype(str).str.strip().isin(ids_to_process)].copy()

    if save:
        df_out.to_csv(out_path, index=False)
        # print(f"\nSaved incremental new notes to: {out_path}")

    print("=" * 50)

    return out_path

    


def inspect_timestamps(path, time_col="createdAtMillis"):
    df = pd.read_csv(path, low_memory=False)

    if time_col not in df.columns:
        raise ValueError(f"{time_col} column not found")

    # Parse createdAtMillis → UTC datetime
    if pd.api.types.is_numeric_dtype(df[time_col]):
        ts = pd.to_datetime(df[time_col], unit="ms", errors="coerce", utc=True)
    else:
        ts = pd.to_datetime(df[time_col], errors="coerce", utc=True)

    n_total = len(ts)
    n_parsed = ts.notna().sum()

    print("=" * 50)
    print(f"Parsed timestamps: {n_parsed}/{n_total} ({n_parsed/n_total:.2%})")
    if n_parsed == 0:
        print("No valid timestamps parsed.")
        return

    # Oldest & newest
    print(f"Oldest entry: {ts.min().strftime('%Y-%m-%d %H:%M:%S.%f+00:00')}")
    print(f"Newest entry: {ts.max().strftime('%Y-%m-%d %H:%M:%S.%f+00:00')}")

    # Monthly counts
    monthly_counts = (
        ts.dropna()
          .dt.to_period("M")
          .value_counts()
          .sort_index()
    )

    print("\nEntries per month:")
    for period, count in monthly_counts.items():
        print(f"  {period}: {count:,}")

    print("=" * 50)





















