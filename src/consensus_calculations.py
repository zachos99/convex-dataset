import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def merge_fixed_dataset_with_history(keys, notes_all_path, notes_history_path, output_path=None):
    
    print("=" * 80)
    print(f"MERGING FIXED DATASET WITH NOTES_ALL AND STATUS HISTORY - {keys.upper()} SET")
    print("=" * 80)
    
    # STEP 1: Load fixed dataset and extract tweet IDs
    print("\n[STEP 1] Loading fixed dataset and extracting tweet IDs...")
    if keys == "image":
        dataset_pth = "notes_tweets_features_image_keywords_misinfo_final_NEW.csv"
    elif keys == "video":
        dataset_pth = "notes_tweets_features_video_keywords_misinfo_final_NEW.csv"
    else:
        print("Wrong Input!"); return

    df_fixed = pd.read_csv(dataset_pth, low_memory=False, dtype={"noteId": str, "tweet_id": str})
    print(f"  ✓ Loaded {len(df_fixed):,} rows from fixed dataset")
    print(f"  ✓ Unique noteIds in fixed dataset: {df_fixed['noteId'].nunique():,}")
    
    # Extract unique tweet IDs (use tweet_id_norm if available, otherwise tweet_id)
    tweet_col = None
    for col in ["tweet_id_norm", "tweet_id", "tweetId"]:
        if col in df_fixed.columns:
            tweet_col = col
            break
    
    if tweet_col:
        # Get unique tweet IDs and deduplicate (keep first)
        tweets = df_fixed[[tweet_col]].drop_duplicates(tweet_col, keep="first").copy()
        tweets = tweets.rename(columns={tweet_col: "tweet_id_norm"})
        tweets = tweets.dropna(subset=["tweet_id_norm"])
        
        # Rename tweet_col to tweet_id_norm for consistency
        if tweet_col != "tweet_id_norm":
            df_fixed_meta = df_fixed.rename(columns={tweet_col: "tweet_id_norm"})
        else:
            df_fixed_meta = df_fixed.copy()
        
        # Get tweet creation time
        if "created_at_datetime" in df_fixed_meta.columns:
            df_fixed_meta["tweet_creation_ts"] = pd.to_datetime(df_fixed_meta["created_at_datetime"], utc=True, errors="coerce")
        elif "tweet_creation_ts" in df_fixed_meta.columns:
            pass  # Already exists
        else:
            df_fixed_meta["tweet_creation_ts"] = None
        
        # Deduplicate by tweet_id_norm, keeping first
        df_fixed_meta = df_fixed_meta.drop_duplicates("tweet_id_norm", keep="first")
        
        # Get tweet metadata columns from fixed dataset (after renaming)
        tweet_meta_cols = [
            "tweet_id_norm",  # Always use tweet_id_norm after renaming
            "event", "topic", "language", "misinfo_type_final", "misinfo_type",
            "user_followers", "view_count", "retweet_count", "favorite_count",
            "reply_count", "bookmark_count", "full_text", "media"
        ]
        present_cols = [c for c in tweet_meta_cols if c in df_fixed_meta.columns]
        if "tweet_creation_ts" in df_fixed_meta.columns and "tweet_creation_ts" not in present_cols:
            present_cols.append("tweet_creation_ts")
        
        unique_tweet_ids = set(tweets["tweet_id_norm"].dropna().unique())
        print(f"  ✓ Unique tweets in dataset: {len(unique_tweet_ids):,}")
    else:
        raise ValueError("No tweet ID column found in fixed dataset")
    
    # STEP 2: Load notes_all.tsv and filter to matching tweets
    print("\n[STEP 2] Loading notes_all.tsv and filtering to dataset tweets...")
    notes_all = pd.read_csv(notes_all_path, sep="\t", low_memory=False, 
                           dtype={"tweetId": str, "noteId": str})
    notes_all["note_creation_ts"] = pd.to_datetime(notes_all["createdAtMillis"], 
                                                     unit="ms", utc=True, errors="coerce")
    
    # Filter to notes that reference tweets in our dataset
    tweet_notes = notes_all[notes_all["tweetId"].isin(unique_tweet_ids)].copy()
    print(f"  ✓ Notes from notes_all.tsv matching dataset tweets: {len(tweet_notes):,}")
    print(f"  ✓ Unique noteIds from notes_all: {tweet_notes['noteId'].nunique():,}")
    
    # STEP 3: Load note status history
    print("\n[STEP 3] Loading note status history...")
    hist = pd.read_csv(notes_history_path, sep="\t", low_memory=False, dtype={"noteId": str})
    hist.columns = hist.columns.str.strip()
    print(f"  ✓ Loaded {len(hist):,} rows from status history")
    print(f"  ✓ Unique noteIds in history: {hist['noteId'].nunique():,}")
    
    # Normalize column names
    rename_map = {
        "timestampMillisOfFirstNonNMRStatus": "ts_first_nonNMR",
        "firstNonNMRStatus": "first_nonNMR_status",
        "timestampMillisOfCurrentStatus": "ts_current_status",
        "currentStatus": "current_status",
        "timestampMillisOfLatestNonNMRStatus": "ts_latest_nonNMR",
        "mostRecentNonNMRStatus": "latest_nonNMR_status",
        "timestampMillisOfStatusLock": "ts_status_lock",
        "lockedStatus": "locked_status",
        "createdAtMillis": "createdAtMillis",
    }
    hist = hist.rename(columns=rename_map)
    
    # Convert millis → datetime
    def to_dt_ms(s): 
        return pd.to_datetime(s, unit="ms", utc=True, errors="coerce")
    
    hist["note_creation_ts_hist"] = to_dt_ms(hist.get("createdAtMillis"))
    hist["first_nonNMR_ts"] = to_dt_ms(hist.get("ts_first_nonNMR"))
    hist["current_status_ts"] = to_dt_ms(hist.get("ts_current_status"))
    hist["latest_nonNMR_ts"] = to_dt_ms(hist.get("ts_latest_nonNMR"))
    hist["lock_ts"] = to_dt_ms(hist.get("ts_status_lock"))
    
    # STEP 4: Merge notes_all with status history
    print("\n[STEP 4] Merging notes_all with status history...")
    enriched = tweet_notes.merge(
        hist[[
            "noteId", "note_creation_ts_hist",
            "first_nonNMR_ts", "first_nonNMR_status",
            "current_status_ts", "current_status",
            "latest_nonNMR_ts", "latest_nonNMR_status",
            "lock_ts", "locked_status"
        ]],
        on="noteId",
        how="left"
    )
    
    print(f"  ✓ Merged: {len(enriched):,} rows")
    print(f"  ✓ Notes with status history: {enriched['note_creation_ts_hist'].notna().sum():,} ({enriched['note_creation_ts_hist'].notna().mean()*100:.1f}%)")
    
    # Prefer note creation from history, otherwise use notes_all
    enriched["note_creation_ts_final"] = enriched["note_creation_ts_hist"].combine_first(enriched["note_creation_ts"])
    
    # STEP 5: Merge back with fixed dataset to get tweet metadata
    print("\n[STEP 5] Merging with fixed dataset for tweet metadata...")
    enriched = enriched.merge(
        df_fixed_meta[present_cols],
        left_on="tweetId",
        right_on="tweet_id_norm",
        how="left",
        suffixes=("", "_from_fixed")
    )
    
    # Ensure tweet_id_norm exists
    if "tweet_id_norm" not in enriched.columns:
        enriched["tweet_id_norm"] = enriched["tweetId"]
    
    print(f"  ✓ Notes with tweet metadata from fixed dataset: {enriched['tweet_id_norm'].notna().sum():,}")
    
    # STEP 6: Compute latencies (hours)
    print("\n[STEP 6] Computing time latencies...")
    def hours_since(start, end):
        start = pd.to_datetime(start, utc=True, errors="coerce").dt.tz_localize(None)
        end = pd.to_datetime(end, utc=True, errors="coerce").dt.tz_localize(None)
        return (end - start).dt.total_seconds() / 3600
    
    enriched["time_to_first_nonNMR_h"] = hours_since(enriched["note_creation_ts_final"], enriched["first_nonNMR_ts"])
    enriched["time_to_latest_nonNMR_h"] = hours_since(enriched["note_creation_ts_final"], enriched["latest_nonNMR_ts"])
    enriched["time_to_current_status_h"] = hours_since(enriched["note_creation_ts_final"], enriched["current_status_ts"])
    enriched["time_to_lock_h"] = hours_since(enriched["note_creation_ts_final"], enriched["lock_ts"])
    
    # STEP 7: Volatility/stability flags
    print("\n[STEP 7] Computing volatility and stability flags...")
    for c in ["first_nonNMR_status", "latest_nonNMR_status", "current_status", "locked_status"]:
        if c in enriched.columns:
            enriched[c] = enriched[c].replace({"": np.nan, "None": np.nan, "none": np.nan, "NONE": np.nan})
    
    enriched["status_flipped"] = (
        enriched["first_nonNMR_status"].notna()
        & enriched["latest_nonNMR_status"].notna()
        & (enriched["first_nonNMR_status"] != enriched["latest_nonNMR_status"])
    )
    
    enriched["left_NMR_at_least_once"] = enriched["first_nonNMR_ts"].notna()
    enriched["is_locked_final"] = (
        enriched["locked_status"].notna()
        & enriched["current_status"].notna()
        & (enriched["locked_status"] == enriched["current_status"])
    )
    
    # Tweet→Note lag
    if enriched["tweet_creation_ts"].notna().any():
        enriched["time_from_tweet_to_note_creation_h"] = hours_since(
            enriched["tweet_creation_ts"], enriched["note_creation_ts_final"]
        )
    else:
        enriched["time_from_tweet_to_note_creation_h"] = None
    
    # Rename tweet_id to tweet_id_norm for consistency
    if "tweet_id" in enriched.columns and "tweet_id_norm" not in enriched.columns:
        enriched["tweet_id_norm"] = enriched["tweet_id"]
    
    # Save output
    if output_path is None:
        output_path = f"responsiveness/tweet_notes_lifecycle_{keys}_set.csv"
    
    print(f"\n[STEP 6] Saving merged dataset...")
    enriched.to_csv(output_path, index=False)
    print(f"  ✓ Saved to: {output_path}")
    print(f"  ✓ Total columns: {len(enriched.columns)}")
    print(f"  ✓ Total rows: {len(enriched):,}")
    print("\n" + "=" * 80)
    print("MERGE COMPLETE")
    print("=" * 80)
    
    return enriched



def lifecycle_file_analysis(keys):  
    
    if keys == "image":
        path = "responsiveness/tweet_notes_lifecycle_image_set.csv"
    elif keys == "video":
        path = "responsiveness/tweet_notes_lifecycle_video_set.csv"
    else:
        print("Wrong Input!")
        return

    df = pd.read_csv(path, low_memory=False)
    print(f"Rows: {len(df):,}")

    def col_info(col):
        if col not in df.columns:
            print(f"[missing] {col}")
            return
        non_null = df[col].notna().sum()
        null = df[col].isna().sum()
        print(f"[{col}] non-null: {non_null:,} | null: {null:,}")
        print(df[col].value_counts(dropna=False).head(10), "\n")

    # Core IDs
    for col in ["noteId", "tweet_id_norm", "tweetId"]:
        col_info(col)
    if "noteId" in df.columns:
        print(f"Unique noteId: {df['noteId'].nunique():,}")
    if "tweet_id_norm" in df.columns:
        print(f"Unique tweet_id_norm: {df['tweet_id_norm'].nunique():,}")
    if "tweetId" in df.columns:
        print(f"Unique tweetId: {df['tweetId'].nunique():,}")
    print()

    # Tweet-level metadata
    for col in ["misinfo_type_final", "misinfo_type", "topic", "language"]:
        col_info(col)

    # Status/history columns
    for col in [
        "first_nonNMR_status", "latest_nonNMR_status", "current_status", "locked_status",
        "first_nonNMR_ts", "latest_nonNMR_ts", "current_status_ts", "lock_ts",
    ]:
        col_info(col)

    # Timing fields
    for col in [
        "note_creation_ts_final", "note_creation_ts_hist", "note_creation_ts",
        "time_to_first_nonNMR_h", "time_to_latest_nonNMR_h", "time_to_current_status_h",
        "time_to_lock_h", "time_from_tweet_to_note_creation_h",
    ]:
        col_info(col)


def compute_whole_dataset_statistics(notes_all_path, notes_history_path):
    """
    Compute statistics on the ENTIRE dataset (notes_all.tsv + noteStatusHistory.tsv)
    without any filtering. This allows comparison with filtered subsets.
    
    Computes:
    - Unique notes, unique tweets, notes per tweet
    - pct_left_nmr, pct_helpful, pct_not_helpful (using latest_nonNMR_status)
    - median_time_to_first_nonNMR_h
    - median_reaction_time_h (per tweet - earliest note)
    
    Parameters:
    -----------
    notes_all_path : str
        Path to notes_all.tsv file
    notes_history_path : str
        Path to noteStatusHistory.tsv file
    
    Returns:
    --------
    dict
        Dictionary with all computed statistics
    """
    
    print("=" * 80)
    print("WHOLE DATASET STATISTICS (no filtering)")
    print("=" * 80)
    
    # Load notes_all.tsv
    print("\n[STEP 1] Loading notes_all.tsv...")
    notes_all = pd.read_csv(notes_all_path, sep="\t", low_memory=False, 
                           dtype={"tweetId": str, "noteId": str})
    notes_all["note_creation_ts"] = pd.to_datetime(notes_all["createdAtMillis"], 
                                                     unit="ms", utc=True, errors="coerce")
    print(f"  ✓ Loaded {len(notes_all):,} rows")
    print(f"  ✓ Unique noteIds: {notes_all['noteId'].nunique():,}")
    print(f"  ✓ Unique tweetIds: {notes_all['tweetId'].nunique():,}")
    
    # Load note status history
    print("\n[STEP 2] Loading note status history...")
    hist = pd.read_csv(notes_history_path, sep="\t", low_memory=False, dtype={"noteId": str})
    hist.columns = hist.columns.str.strip()
    print(f"  ✓ Loaded {len(hist):,} rows")
    print(f"  ✓ Unique noteIds: {hist['noteId'].nunique():,}")
    
    # Normalize column names
    rename_map = {
        "timestampMillisOfFirstNonNMRStatus": "ts_first_nonNMR",
        "firstNonNMRStatus": "first_nonNMR_status",
        "timestampMillisOfCurrentStatus": "ts_current_status",
        "currentStatus": "current_status",
        "timestampMillisOfLatestNonNMRStatus": "ts_latest_nonNMR",
        "mostRecentNonNMRStatus": "latest_nonNMR_status",
        "timestampMillisOfStatusLock": "ts_status_lock",
        "lockedStatus": "locked_status",
        "createdAtMillis": "createdAtMillis",
    }
    hist = hist.rename(columns=rename_map)
    
    # Convert millis → datetime
    def to_dt_ms(s): 
        return pd.to_datetime(s, unit="ms", utc=True, errors="coerce")
    
    hist["note_creation_ts_hist"] = to_dt_ms(hist.get("createdAtMillis"))
    hist["first_nonNMR_ts"] = to_dt_ms(hist.get("ts_first_nonNMR"))
    hist["current_status_ts"] = to_dt_ms(hist.get("ts_current_status"))
    hist["latest_nonNMR_ts"] = to_dt_ms(hist.get("ts_latest_nonNMR"))
    hist["lock_ts"] = to_dt_ms(hist.get("ts_status_lock"))
    
    # Merge notes_all with status history
    print("\n[STEP 3] Merging notes_all with status history...")
    df = notes_all.merge(
        hist[[
            "noteId", "note_creation_ts_hist",
            "first_nonNMR_ts", "first_nonNMR_status",
            "current_status_ts", "current_status",
            "latest_nonNMR_ts", "latest_nonNMR_status",
            "lock_ts", "locked_status"
        ]],
        on="noteId",
        how="left"
    )
    
    print(f"  ✓ Merged: {len(df):,} rows")
    print(f"  ✓ Notes with status history: {df['note_creation_ts_hist'].notna().sum():,} ({df['note_creation_ts_hist'].notna().mean()*100:.1f}%)")
    
    # Prefer note creation from history, otherwise use notes_all
    df["note_creation_ts_final"] = df["note_creation_ts_hist"].combine_first(df["note_creation_ts"])
    
    # Clean status columns
    for c in ["first_nonNMR_status", "latest_nonNMR_status", "current_status", "locked_status"]:
        if c in df.columns:
            df[c] = df[c].replace({"": np.nan, "None": np.nan, "none": np.nan, "NONE": np.nan})
    
    # Compute time latencies
    print("\n[STEP 4] Computing time latencies...")
    def hours_since(start, end):
        start = pd.to_datetime(start, utc=True, errors="coerce").dt.tz_localize(None)
        end = pd.to_datetime(end, utc=True, errors="coerce").dt.tz_localize(None)
        return (end - start).dt.total_seconds() / 3600
    
    df["time_to_first_nonNMR_h"] = hours_since(df["note_creation_ts_final"], df["first_nonNMR_ts"])
    
    # Compute statistics
    print("\n[STEP 5] Computing statistics...")
    
    n_total = len(df)
    n_unique_notes = df["noteId"].nunique()
    n_unique_tweets = df["tweetId"].nunique()
    notes_per_tweet = (n_total / n_unique_tweets) if n_unique_tweets > 0 else 0
    
    # pct_left_nmr, pct_helpful, pct_not_helpful (using latest_nonNMR_status)
    helpful_status = "CURRENTLY_RATED_HELPFUL"
    not_helpful_status = "CURRENTLY_RATED_NOT_HELPFUL"
    
    if "latest_nonNMR_status" in df.columns:
        latest_has_status = df["latest_nonNMR_status"].notna()
        latest_helpful = (df["latest_nonNMR_status"] == helpful_status)
        latest_not_helpful = (df["latest_nonNMR_status"] == not_helpful_status)
        
        n_latest_with_status = latest_has_status.sum()
        n_latest_helpful = latest_helpful.sum()
        n_latest_not_helpful = latest_not_helpful.sum()
        
        pct_left_nmr = (n_latest_with_status / n_total * 100) if n_total > 0 else 0
        pct_helpful = (n_latest_helpful / n_latest_with_status * 100) if n_latest_with_status > 0 else 0
        pct_not_helpful = (n_latest_not_helpful / n_latest_with_status * 100) if n_latest_with_status > 0 else 0
    else:
        n_latest_with_status = n_latest_helpful = n_latest_not_helpful = 0
        pct_left_nmr = pct_helpful = pct_not_helpful = 0
    
    # median_time_to_first_nonNMR_h
    if "time_to_first_nonNMR_h" in df.columns:
        valid_times = df["time_to_first_nonNMR_h"].dropna()
        median_time_to_first_nonNMR = valid_times.median() if len(valid_times) > 0 else np.nan
        n_with_first_nonNMR = len(valid_times)
    else:
        median_time_to_first_nonNMR = np.nan
        n_with_first_nonNMR = 0
    
    # median_reaction_time_h (per tweet - earliest note only)
    # Note: We don't have tweet_creation_ts in notes_all, so we can't compute reaction time
    # We'll set it to NaN and note this limitation
    median_reaction_time = np.nan
    n_with_reaction_time = 0
    
    # Print results
    print("\n" + "=" * 80)
    print("📊 WHOLE DATASET STATISTICS")
    print("=" * 80)
    print(f"\n  Total notes: {n_total:,}")
    print(f"  Unique notes: {n_unique_notes:,}")
    print(f"  Unique tweets: {n_unique_tweets:,}")
    print(f"  Notes per tweet: {notes_per_tweet:.2f}")
    print(f"\n  Percentage that left NMR (latest_nonNMR_status): {pct_left_nmr:.2f}%")
    print(f"  Percentage Helpful (latest_nonNMR_status): {pct_helpful:.2f}%")
    print(f"  Percentage Not Helpful (latest_nonNMR_status): {pct_not_helpful:.2f}%")
    print(f"\n  Median time to first non-NMR status: {median_time_to_first_nonNMR:.2f} hours" if pd.notna(median_time_to_first_nonNMR) else f"\n  Median time to first non-NMR status: N/A")
    print(f"    (based on {n_with_first_nonNMR:,} notes)")
    print(f"\n  Median reaction time: N/A (tweet creation time not available in notes_all.tsv)")
    print("\n" + "=" * 80)
    
    results = {
        "n_total": n_total,
        "n_unique_notes": n_unique_notes,
        "n_unique_tweets": n_unique_tweets,
        "notes_per_tweet": round(notes_per_tweet, 2),
        "pct_left_nmr": round(pct_left_nmr, 2),
        "pct_helpful": round(pct_helpful, 2),
        "pct_not_helpful": round(pct_not_helpful, 2),
        "median_time_to_first_nonNMR_h": round(median_time_to_first_nonNMR, 2) if pd.notna(median_time_to_first_nonNMR) else np.nan,
        "n_with_first_nonNMR": n_with_first_nonNMR,
        "median_reaction_time_h": np.nan,  # Not available without tweet creation time
        "n_with_reaction_time": n_with_reaction_time,
    }
    
    return results


def compute_misinfo_comparison_metrics(keys):
    """
    Compute key metrics for comparative analysis:
    a) Helpful/Not Helpful percentages
    b) Percentage that left NMR (have non-NMR status)
    c) Median time to first non-NMR status
    d) Median reaction time (time_from_tweet_to_note_creation_h)
    """
    
    print("=" * 80)
    print(f"MISINFO TYPE COMPARISON METRICS - {keys.upper()} SET")
    print("=" * 80)
    
    # Read lifecycle file
    lifecycle_file = f"responsiveness/tweet_notes_lifecycle_{keys}_set.csv"
    print(f"\n[STEP 1] Loading lifecycle file: {lifecycle_file}")
    df = pd.read_csv(lifecycle_file, low_memory=False)
    print(f"  ✓ Loaded {len(df):,} rows")
    
    misinfo_col = "misinfo_type_final"
    
    # Filter to three types 
    target_types = ['miscaptioned', 'edited', 'ai_generated']
    df = df[df[misinfo_col].isin(target_types)].copy()
    
    # Clean status columns
    for c in ["first_nonNMR_status", "latest_nonNMR_status", "current_status", "locked_status"]:
        if c in df.columns:
            df[c] = df[c].replace({"": np.nan, "None": np.nan, "none": np.nan, "NONE": np.nan})
    
    print("\n[STEP 2] Computing metrics...")
    
    def compute_metrics_group(df_subset, group_name="Overall"):
        """Compute metrics for a subset of data."""
        n_total = len(df_subset)
        
        # a) Helpful/Not Helpful percentages - compute separately for each status column
        helpful_status = "CURRENTLY_RATED_HELPFUL"
        not_helpful_status = "CURRENTLY_RATED_NOT_HELPFUL"
        
        # Current status
        if "current_status" in df_subset.columns:
            # For pct_left_nmr: count only notes that have left NMR (not "NEEDS_MORE_RATINGS")
            nmr_status = "NEEDS_MORE_RATINGS"
            current_left_nmr = df_subset["current_status"].notna() & (df_subset["current_status"] != nmr_status)
            current_has_status = df_subset["current_status"].notna()
            current_helpful = (df_subset["current_status"] == helpful_status)
            current_not_helpful = (df_subset["current_status"] == not_helpful_status)
            
            n_current_left_nmr = current_left_nmr.sum()
            n_current_with_status = current_has_status.sum()
            n_current_helpful = current_helpful.sum()
            n_current_not_helpful = current_not_helpful.sum()
            
            pct_left_nmr_current = (n_current_left_nmr / n_total * 100) if n_total > 0 else 0
            pct_helpful_current = (n_current_helpful / n_current_with_status * 100) if n_current_with_status > 0 else 0
            pct_not_helpful_current = (n_current_not_helpful / n_current_with_status * 100) if n_current_with_status > 0 else 0
        else:
            n_current_with_status = n_current_helpful = n_current_not_helpful = 0
            pct_left_nmr_current = pct_helpful_current = pct_not_helpful_current = 0
        
        # Locked status
        # Note: locked_status only exists for notes that have been locked (left NMR)
        # So if locked_status is not null, the note has left NMR
        if "locked_status" in df_subset.columns:
            locked_has_status = df_subset["locked_status"].notna()
            locked_helpful = (df_subset["locked_status"] == helpful_status)
            locked_not_helpful = (df_subset["locked_status"] == not_helpful_status)
            
            n_locked_with_status = locked_has_status.sum()
            n_locked_helpful = locked_helpful.sum()
            n_locked_not_helpful = locked_not_helpful.sum()
            
            # Locked status only exists for notes that left NMR, so this is correct
            pct_left_nmr_locked = (n_locked_with_status / n_total * 100) if n_total > 0 else 0
            pct_helpful_locked = (n_locked_helpful / n_locked_with_status * 100) if n_locked_with_status > 0 else 0
            pct_not_helpful_locked = (n_locked_not_helpful / n_locked_with_status * 100) if n_locked_with_status > 0 else 0
        else:
            n_locked_with_status = n_locked_helpful = n_locked_not_helpful = 0
            pct_left_nmr_locked = pct_helpful_locked = pct_not_helpful_locked = 0
        
        # Latest non-NMR status
        if "latest_nonNMR_status" in df_subset.columns:
            latest_has_status = df_subset["latest_nonNMR_status"].notna()
            latest_helpful = (df_subset["latest_nonNMR_status"] == helpful_status)
            latest_not_helpful = (df_subset["latest_nonNMR_status"] == not_helpful_status)
            
            n_latest_with_status = latest_has_status.sum()
            n_latest_helpful = latest_helpful.sum()
            n_latest_not_helpful = latest_not_helpful.sum()
            
            pct_left_nmr_latest = (n_latest_with_status / n_total * 100) if n_total > 0 else 0
            pct_helpful_latest = (n_latest_helpful / n_latest_with_status * 100) if n_latest_with_status > 0 else 0
            pct_not_helpful_latest = (n_latest_not_helpful / n_latest_with_status * 100) if n_latest_with_status > 0 else 0
        else:
            n_latest_with_status = n_latest_helpful = n_latest_not_helpful = 0
            pct_left_nmr_latest = pct_helpful_latest = pct_not_helpful_latest = 0
        
        # b) Median time to first non-NMR status
        if "time_to_first_nonNMR_h" in df_subset.columns:
            valid_times = df_subset["time_to_first_nonNMR_h"].dropna()
            median_time_to_first_nonNMR = valid_times.median() if len(valid_times) > 0 else np.nan
            n_with_first_nonNMR = len(valid_times)
        else:
            median_time_to_first_nonNMR = np.nan
            n_with_first_nonNMR = 0
        
        # d) Notes per tweet
        # Calculation: n_total / n_tweets (same as responsiveness.py)
        # Uses tweet_id_norm.nunique() to count unique tweets, then divides total notes by unique tweets
        # This gives average notes per tweet for the group
        tweet_col = None
        for col in ["tweet_id_norm", "tweet_id", "tweetId"]:
            if col in df_subset.columns:
                tweet_col = col
                break
        
        if tweet_col:
            n_tweets = df_subset[tweet_col].nunique()
            notes_per_tweet = (n_total / n_tweets) if n_tweets > 0 else 0
        else:
            n_tweets = 0
            notes_per_tweet = 0
        
        # c) Median reaction time (per tweet - earliest note only)
        # Reaction time is a tweet-level metric: how fast did the system react to this tweet?
        # So we first get the earliest note per tweet, then compute median across tweets
        if "time_from_tweet_to_note_creation_h" in df_subset.columns and tweet_col:
            # Group by tweet and get minimum (earliest) reaction time per tweet
            per_tweet_reaction = (
                df_subset.groupby(tweet_col)["time_from_tweet_to_note_creation_h"]
                .min()  # earliest note per tweet
                .dropna()
            )
            median_reaction_time = per_tweet_reaction.median() if len(per_tweet_reaction) > 0 else np.nan
            n_with_reaction_time = len(per_tweet_reaction)
        elif "time_from_tweet_to_note_creation_h" in df_subset.columns:
            # Fallback: use all notes if no tweet column found
            valid_reaction = df_subset["time_from_tweet_to_note_creation_h"].dropna()
            median_reaction_time = valid_reaction.median() if len(valid_reaction) > 0 else np.nan
            n_with_reaction_time = len(valid_reaction)
        else:
            median_reaction_time = np.nan
            n_with_reaction_time = 0
        
        return {
            "group": group_name,
            "n_total": n_total,
            "n_tweets": n_tweets,
            "notes_per_tweet": round(notes_per_tweet, 2),
            # Primary metrics (using latest_nonNMR_status)
            "pct_left_nmr": round(pct_left_nmr_latest, 2),
            "pct_helpful": round(pct_helpful_latest, 2),
            "pct_not_helpful": round(pct_not_helpful_latest, 2),
            # Fallback metrics (current status)
            "pct_left_nmr_current": round(pct_left_nmr_current, 2),
            "pct_helpful_current": round(pct_helpful_current, 2),
            "pct_not_helpful_current": round(pct_not_helpful_current, 2),
            # Fallback metrics (locked status)
            "pct_left_nmr_locked": round(pct_left_nmr_locked, 2),
            "pct_helpful_locked": round(pct_helpful_locked, 2),
            "pct_not_helpful_locked": round(pct_not_helpful_locked, 2),
            # Latest non-NMR status metrics (for reference)
            "pct_left_nmr_latest": round(pct_left_nmr_latest, 2),
            "pct_helpful_latest": round(pct_helpful_latest, 2),
            "pct_not_helpful_latest": round(pct_not_helpful_latest, 2),
            # Time metrics
            "median_time_to_first_nonNMR_h": round(median_time_to_first_nonNMR, 2) if pd.notna(median_time_to_first_nonNMR) else np.nan,
            "n_with_first_nonNMR": n_with_first_nonNMR,
            "median_reaction_time_h": round(median_reaction_time, 2) if pd.notna(median_reaction_time) else np.nan,
            "n_with_reaction_time": n_with_reaction_time,
        }
    
    # Compute overall metrics
    overall_metrics = compute_metrics_group(df, "Overall")
    
    results = {"overall": overall_metrics}
    
    # Print overall results
    print(f"\n📊 OVERALL METRICS (using latest_nonNMR_status):")
    print(f"  Total notes: {overall_metrics['n_total']:,}")
    print(f"\n  a) Percentage that left NMR: {overall_metrics['pct_left_nmr']:.2f}%")
    print(f"  b) Percentage Helpful: {overall_metrics['pct_helpful']:.2f}%")
    print(f"  c) Percentage Not Helpful: {overall_metrics['pct_not_helpful']:.2f}%")
    print(f"\n  d) Notes per tweet: {overall_metrics['notes_per_tweet']:.2f} (from {overall_metrics['n_tweets']:,} tweets)")
    if pd.notna(overall_metrics['median_time_to_first_nonNMR_h']):
        print(f"  e) Median time to first non-NMR status: {overall_metrics['median_time_to_first_nonNMR_h']:.2f} hours")
    else:
        print(f"  e) Median time to first non-NMR status: N/A")
    if pd.notna(overall_metrics['median_reaction_time_h']):
        print(f"  f) Median reaction time (tweet → note creation): {overall_metrics['median_reaction_time_h']:.2f} hours")
    else:
        print(f"  f) Median reaction time: N/A")
    
    # Compute by misinfo_type if requested
    print(f"\n📊 METRICS BY MISINFO TYPE:")
    grouped_metrics = []
    for misinfo_type in df[misinfo_col].dropna().unique():
        df_type = df[df[misinfo_col] == misinfo_type].copy()
        type_metrics = compute_metrics_group(df_type, misinfo_type)
        grouped_metrics.append(type_metrics)
        results[misinfo_type] = type_metrics
        
    # Create summary DataFrame for display
    summary_df = pd.DataFrame(grouped_metrics)
    summary_df = summary_df.sort_values("n_total", ascending=False)
        
    print("\n  Summary Table:")
    print("  NOTE: All metrics use 'latest_nonNMR_status' as the primary status column")
    print("        (pct_left_nmr = % with non-null latest_nonNMR_status,")
    print("         pct_helpful/not_helpful = % among those with latest_nonNMR_status)")
    print(summary_df[["group", "n_total", "notes_per_tweet", 
                      "pct_left_nmr", "pct_helpful", "pct_not_helpful",
                      "median_time_to_first_nonNMR_h", "median_reaction_time_h"]].to_string(index=False))
        
    print("\n  Detailed breakdown by type:")
    for metrics in grouped_metrics:
        print(f"\n  ── {metrics['group'].upper()} ──")
        print(f"    Total notes: {metrics['n_total']:,} | Notes per tweet: {metrics['notes_per_tweet']:.2f}")
        print(f"    Left NMR: {metrics['pct_left_nmr']:.2f}% | Helpful: {metrics['pct_helpful']:.2f}% | Not Helpful: {metrics['pct_not_helpful']:.2f}%")
        if pd.notna(metrics['median_time_to_first_nonNMR_h']):
            print(f"    Median time to first non-NMR: {metrics['median_time_to_first_nonNMR_h']:.2f} hours")
        if pd.notna(metrics['median_reaction_time_h']):
            print(f"    Median reaction time: {metrics['median_reaction_time_h']:.2f} hours")
    
    print("\n" + "=" * 80)
    print("METRICS COMPUTATION COMPLETE")
    print("=" * 80)
    
    return results



def plot_misinfo_metrics_timeseries(keys, start_month=None, min_notes_per_month=0, save=False):
    """
    Plot misinfo comparison metrics over time (by month) and by misinfo type.
    
    Plots:
    1) pct_left_nmr - Percentage that left NMR
    2) pct_helpful - Percentage helpful
    3) pct_not_helpful - Percentage not helpful
    4) median_time_to_first_nonNMR_h - Median time to first non-NMR
    5) median_reaction_time_h - Median reaction time
    """
    
    print("=" * 80)
    print(f"MISINFO METRICS TIMESERIES - {keys.upper()} SET")
    print("=" * 80)
    
    # Read lifecycle file
    lifecycle_file = f"responsiveness/tweet_notes_lifecycle_{keys}_set.csv"
    print(f"\n[STEP 1] Loading lifecycle file: {lifecycle_file}")
    df = pd.read_csv(
        lifecycle_file, 
        low_memory=False,
        parse_dates=["note_creation_ts_final", "first_nonNMR_ts", "tweet_creation_ts"]
    )
    print(f"  ✓ Loaded {len(df):,} rows")
    
    # Determine misinfo_type column
    if "misinfo_type_final" in df.columns:
        misinfo_col = "misinfo_type_final"
    elif "misinfo_type" in df.columns:
        misinfo_col = "misinfo_type"
    else:
        raise ValueError("No misinfo_type column found in dataset")
    
    # Filter to three types
    target_types = ['miscaptioned', 'edited', 'ai_generated']
    before = len(df)
    df = df[df[misinfo_col].isin(target_types)].copy()
    print(f"  ✓ Filtered to 3 misinfo types: {len(df):,} rows ({before - len(df):,} dropped)")
    
    # Clean status columns
    for c in ["first_nonNMR_status", "latest_nonNMR_status", "current_status", "locked_status"]:
        if c in df.columns:
            df[c] = df[c].replace({"": np.nan, "None": np.nan, "none": np.nan, "NONE": np.nan})
    
    # Create note_month column
    print("\n[STEP 2] Creating monthly aggregates...")
    df["note_month"] = pd.to_datetime(df["note_creation_ts_final"], errors="coerce").dt.tz_localize(None)
    df["note_month"] = df["note_month"].dt.to_period("M").dt.to_timestamp()

    # Filter to last month
    last_month = df["note_month"].max()
    df = df[df["note_month"] < last_month].copy()

    
    # Filter by start_month if provided
    if start_month:
        start_ts = pd.to_datetime(f"{start_month}-01")
        before = len(df)
        df = df[df["note_month"] >= start_ts].copy()
        print(f"  ✓ Filtered to notes from {start_month}+: {len(df):,} rows ({before - len(df):,} dropped)")
    
    # Compute metrics per (misinfo_type, month)
    monthly_metrics = []
    
    for misinfo_type in target_types:
        df_type = df[df[misinfo_col] == misinfo_type].copy()
        
        for month, df_month in df_type.groupby("note_month"):
            n_total = len(df_month)
            
            # Skip if below minimum threshold
            if min_notes_per_month > 0 and n_total < min_notes_per_month:
                continue
            
            # 1) pct_left_nmr (using latest_nonNMR_status as primary)
            if "latest_nonNMR_status" in df_month.columns:
                n_left_nmr = df_month["latest_nonNMR_status"].notna().sum()
            elif "first_nonNMR_ts" in df_month.columns:
                n_left_nmr = df_month["first_nonNMR_ts"].notna().sum()
            elif "left_NMR_at_least_once" in df_month.columns:
                n_left_nmr = df_month["left_NMR_at_least_once"].fillna(False).astype(bool).sum()
            else:
                n_left_nmr = 0
            pct_left_nmr = (n_left_nmr / n_total * 100) if n_total > 0 else 0
            
            # 2) pct_helpful and 3) pct_not_helpful
            # Use latest_nonNMR_status as primary, fallback to current_status, then locked_status
            status_col = None
            if "latest_nonNMR_status" in df_month.columns:
                status_col = "latest_nonNMR_status"
            elif "current_status" in df_month.columns:
                status_col = "current_status"
            elif "locked_status" in df_month.columns:
                status_col = "locked_status"
            
            if status_col:
                helpful_mask = df_month[status_col] == "CURRENTLY_RATED_HELPFUL"
                not_helpful_mask = df_month[status_col] == "CURRENTLY_RATED_NOT_HELPFUL"
                has_status = df_month[status_col].notna()
                
                n_helpful = helpful_mask.sum()
                n_not_helpful = not_helpful_mask.sum()
                n_with_status = has_status.sum()
                
                pct_helpful = (n_helpful / n_with_status * 100) if n_with_status > 0 else 0
                pct_not_helpful = (n_not_helpful / n_with_status * 100) if n_with_status > 0 else 0
            else:
                n_helpful = n_not_helpful = n_with_status = 0
                pct_helpful = pct_not_helpful = 0
            
            # 4) median_time_to_first_nonNMR_h
            if "time_to_first_nonNMR_h" in df_month.columns:
                valid_times = df_month["time_to_first_nonNMR_h"].dropna()
                median_time_to_first_nonNMR = valid_times.median() if len(valid_times) > 0 else np.nan
            else:
                median_time_to_first_nonNMR = np.nan
            
            # 5) median_reaction_time_h (per tweet - earliest note only)
            # Reaction time is a tweet-level metric: how fast did the system react to this tweet?
            # So we first get the earliest note per tweet, then compute median across tweets
            tweet_col_month = None
            for col in ["tweet_id_norm", "tweet_id", "tweetId"]:
                if col in df_month.columns:
                    tweet_col_month = col
                    break
            
            if "time_from_tweet_to_note_creation_h" in df_month.columns and tweet_col_month:
                # Group by tweet and get minimum (earliest) reaction time per tweet
                per_tweet_reaction = (
                    df_month.groupby(tweet_col_month)["time_from_tweet_to_note_creation_h"]
                    .min()  # earliest note per tweet
                    .dropna()
                )
                median_reaction_time = per_tweet_reaction.median() if len(per_tweet_reaction) > 0 else np.nan
            elif "time_from_tweet_to_note_creation_h" in df_month.columns:
                # Fallback: use all notes if no tweet column found
                valid_reaction = df_month["time_from_tweet_to_note_creation_h"].dropna()
                median_reaction_time = valid_reaction.median() if len(valid_reaction) > 0 else np.nan
            else:
                median_reaction_time = np.nan
            
            monthly_metrics.append({
                "misinfo_type": misinfo_type,
                "note_month": month,
                "n_notes": n_total,
                "pct_left_nmr": round(pct_left_nmr, 2),
                "pct_helpful": round(pct_helpful, 2),
                "pct_not_helpful": round(pct_not_helpful, 2),
                "median_time_to_first_nonNMR_h": round(median_time_to_first_nonNMR, 2) if pd.notna(median_time_to_first_nonNMR) else np.nan,
                "median_reaction_time_h": round(median_reaction_time, 2) if pd.notna(median_reaction_time) else np.nan,
            })
    
    monthly_df = pd.DataFrame(monthly_metrics)
    monthly_df = monthly_df.sort_values(["misinfo_type", "note_month"]).reset_index(drop=True)
    
    print(f"  ✓ Computed metrics for {len(monthly_df):,} (misinfo_type, month) combinations")
    
    # Create plots
    print("\n[STEP 3] Creating plots...")
    
    metrics_to_plot = [
        ("pct_left_nmr", "Percentage that Left NMR", "%"),
        ("pct_helpful", "Percentage Helpful", "%"),
        ("pct_not_helpful", "Percentage Not Helpful", "%"),
        ("median_time_to_first_nonNMR_h", "Median Time to First Non-NMR", "hours"),
        ("median_reaction_time_h", "Median Reaction Time", "hours"),
    ]
    
    for metric_col, metric_title, ylabel_unit in metrics_to_plot:
        plt.figure(figsize=(12, 6))
        
        for misinfo_type in target_types:
            df_type = monthly_df[monthly_df["misinfo_type"] == misinfo_type].copy()
            if len(df_type) > 0:
                plt.plot(
                    df_type["note_month"], 
                    df_type[metric_col], 
                    marker="o", 
                    label=misinfo_type,
                    linewidth=2,
                    markersize=4
                )
        
        plt.title(f"{metric_title} Over Time by Misinfo Type — {keys.capitalize()} Set", fontsize=14)
        plt.xlabel("Note Month", fontsize=12)
        plt.ylabel(f"{metric_title} ({ylabel_unit})", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(title="Misinfo Type", loc="best", frameon=True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save:
            # Create plots directory if it doesn't exist
            import os
            os.makedirs("plots", exist_ok=True)
            filename = f"plots/{metric_col}_timeseries_by_misinfo_{keys}.png"
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"  ✓ Saved: {filename}")
        else:
            plt.show()
    
    # Save CSV if requested
    if save:
        csv_filename = f"misinfo_metrics_timeseries_{keys}.csv"
        monthly_df.to_csv(csv_filename, index=False)
        print(f"  ✓ Saved CSV: {csv_filename}")
    
    print("\n" + "=" * 80)
    print("TIMESERIES PLOTTING COMPLETE")
    print("=" * 80)
    
    return monthly_df


"""
    Compute per misinfo type:
        - the probability of a tweet having at least one note to reach consensus (to leave NMR)
        - the probability of the first note to reach consensus
        - the median number of notes until one reaches consensus
        - the mean number of notes until one reaches consensus
"""

def consensus_metrics_summary(keys, save=False, out_dir="results"):
    """
    Compute concise tweet-level consensus metrics per misinfo type.

    Metrics returned per misinfo type:
      - n_tweets_with_notes
      - n_tweets_with_consensus
      - P_tweet_has_consensus
      - P_first_note_leaves_NMR
      - notes_until_first_consensus_median
      - notes_until_first_consensus_mean

    Definitions:
      - A tweet reaches consensus if ANY of its notes left NMR at least once
      - First note = earliest by note_creation_ts_final
      - notes_until_first_consensus counts notes up to (and including) the first
        note that left NMR (only for tweets that reached consensus)
    """

    # ------------------------------------------------------------
    # Load
    # ------------------------------------------------------------
    lifecycle_file = f"responsiveness/tweet_notes_lifecycle_{keys}_set.csv"
    df = pd.read_csv(lifecycle_file, low_memory=False)

    required = [
        "tweet_id_norm",
        "noteId",
        "misinfo_type_final",
        "left_NMR_at_least_once",
        "note_creation_ts_final",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Filter misinfo types
    target_types = ["miscaptioned", "edited", "ai_generated"]
    df = df[df["misinfo_type_final"].isin(target_types)].copy()

    # ------------------------------------------------------------
    # Clean / coerce
    # ------------------------------------------------------------
    df["left_NMR_at_least_once"] = (
        df["left_NMR_at_least_once"]
        .replace({"True": 1, "False": 0, True: 1, False: 0})
    )
    df["left_NMR_at_least_once"] = pd.to_numeric(
        df["left_NMR_at_least_once"], errors="coerce"
    )
    df = df.dropna(subset=["left_NMR_at_least_once"])
    df["left_NMR_at_least_once"] = (df["left_NMR_at_least_once"] > 0).astype(int)

    if np.issubdtype(df["note_creation_ts_final"].dtype, np.number):
        df["note_creation_ts_final"] = pd.to_datetime(
            df["note_creation_ts_final"], unit="ms", utc=True, errors="coerce"
        )
    else:
        df["note_creation_ts_final"] = pd.to_datetime(
            df["note_creation_ts_final"], utc=True, errors="coerce"
        )
    df = df.dropna(subset=["note_creation_ts_final"])

    # ------------------------------------------------------------
    # Sort for first-note logic
    # ------------------------------------------------------------
    df = df.sort_values(
        ["tweet_id_norm", "note_creation_ts_final", "noteId"]
    )

    # ------------------------------------------------------------
    # Tweet-level aggregation
    # ------------------------------------------------------------
    def _tweet_metrics(g):
        g = g.reset_index(drop=True)

        tweet_has_consensus = int(g["left_NMR_at_least_once"].max() == 1)
        first_note_leaves = int(g.loc[0, "left_NMR_at_least_once"] == 1)

        if tweet_has_consensus:
            first_success_idx = int(
                np.argmax(g["left_NMR_at_least_once"].values == 1)
            )
            notes_until = first_success_idx + 1
        else:
            notes_until = np.nan

        return pd.Series({
            "tweet_has_consensus": tweet_has_consensus,
            "first_note_leaves_NMR": first_note_leaves,
            "notes_until_first_consensus": notes_until,
        })

    tweet_level = (
        df.groupby(["tweet_id_norm", "misinfo_type_final"], as_index=False)
          .apply(_tweet_metrics)
          .reset_index(drop=True)
    )

    # ------------------------------------------------------------
    # Summary per misinfo type
    # ------------------------------------------------------------
    rows = []
    for t in target_types:
        sub = tweet_level[tweet_level["misinfo_type_final"] == t]
        n_tweets = len(sub)

        n_consensus = int(sub["tweet_has_consensus"].sum())
        p_consensus = n_consensus / n_tweets if n_tweets else np.nan
        p_first = sub["first_note_leaves_NMR"].mean() if n_tweets else np.nan

        sub_succ = sub[sub["tweet_has_consensus"] == 1]
        x = sub_succ["notes_until_first_consensus"].astype(float)

        rows.append({
            "misinfo_type": t,
            "n_tweets_with_notes": int(n_tweets),
            "n_tweets_with_consensus": int(n_consensus),
            "P_tweet_has_consensus": float(p_consensus),
            "P_first_note_leaves_NMR": float(p_first),
            "notes_until_first_consensus_median": float(np.nanmedian(x)) if len(x) else np.nan,
            "notes_until_first_consensus_mean": float(np.nanmean(x)) if len(x) else np.nan,
        })

    summary = pd.DataFrame(rows).sort_values(
        "P_tweet_has_consensus", ascending=False
    )

    # ------------------------------------------------------------
    # Save (optional)
    # ------------------------------------------------------------
    if save:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"consensus_metrics_{keys}.csv")
        summary.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

    return summary



if __name__ == "__main__":
    
    keys = "image" # for both image and video

    notes_all_pth = "path/to/notes.tsv"
    history_pth = "path/to/noteStatusHistory.tsv"
        
    """
        FIRST RUN THE merge_fixed_dataset_with_history() FUNCTION TO CREATE THE lifecycle_file --> use this file for further analysis

        merge_fixed_dataset_with_history(): 
            Merge fixed dataset with notes_all.tsv and noteStatusHistory to build lifecycle file:
                1. Load dataset → Extract unique tweet IDs (tweet_id_norm)
                2. Load notes_all.tsv → Gather all notes where tweetId matches dataset tweets → This gives us ALL notes for those tweets 
                3. Load noteStatusHistory.tsv → Merge with notes_all notes on noteId
                → Adds status history columns 
                4. Merge back with fixed dataset → Adds tweet metadata (misinfo_type_final, event, topic, etc.)
                → Notes not in fixed dataset will have nulls for these columns
            RESULT:
            - Base: All notes from notes_all.tsv that match dataset tweets
            - Enriched with: Status history from noteStatusHistory.tsv
            - Enriched with: Tweet metadata from fixed dataset (when available)    
    """

    lifecycle_df = merge_fixed_dataset_with_history(keys, notes_all_pth, history_pth)



    """
        Now read the lifecycle_file and perform basic analysis
    """
    lifecycle_file_analysis(keys=keys)


    """
        Compute statistics on the WHOLE dataset (no filtering)
        This allows comparison with filtered subsets
    """
    whole_stats = compute_whole_dataset_statistics(notes_all_pth, history_pth)


    """
        Compute metrics overall and by misinfo type
        Reaction time:
            speed of note creation (community responsiveness)
        Time to first non-NMR:
            speed of note evaluation (rating speed)
    """
    results_df = compute_misinfo_comparison_metrics(keys=keys)



    """
        Plot misinfo metrics timeseries
    """
    monthly_df = plot_misinfo_metrics_timeseries(keys=keys, start_month="2023-05", min_notes_per_month=10, save=False)





    """
        Compute per misinfo type:
            - the probability of a tweet having at least one note to reach consensus (to leave NMR)
            - the probability of the first note to reach consensus
            - the median number of notes until one reaches consensus
            - the mean number of notes until one reaches consensus
    """


    print(consensus_metrics_summary(keys).round(4))











