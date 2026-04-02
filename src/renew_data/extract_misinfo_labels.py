import re, os
from pathlib import Path
import pandas as pd

from misinformation.misinformation_keywords import process_file
from misinformation.find_disagreements import find_disagreements

from misinformation.misinformation_final_label import build_final_misinfo_labels





def merge_final_batches(folder, output_name="merged_final.csv"):
    """
    Merge *_final.csv files in numerical order, print stats, and save merged file.
    """

    # collect final csvs
    files = [
        f for f in os.listdir(folder)
        if f.endswith("_final.csv")
    ]

    if not files:
        raise ValueError("No *_final.csv files found")

    # sort by numeric ranges in filename (e.g. _0_1700_)
    def extract_nums(fname):
        nums = re.findall(r"\d+", fname)
        return tuple(int(n) for n in nums) if nums else (float("inf"),)

    files = sorted(files, key=extract_nums)

    print("Merging files in order:")
    for f in files:
        print("  ", f)

    dfs = []
    for f in files:
        path = os.path.join(folder, f)
        dfs.append(pd.read_csv(path))

    df = pd.concat(dfs, ignore_index=True)

    # --- stats ---
    n_total = len(df)

    n_empty_llm = (
        df["llm_response"].isna()
        | (df["llm_response"].astype(str).str.strip() == "")
    ).sum()

    print("\n=== MERGE STATS ===")
    print(f"Total rows           : {n_total:,}")
    print(f"Empty llm_response   : {n_empty_llm:,} ({n_empty_llm / n_total:.2%})")

    print("\n=== misinfo_type_llm distribution ===")
    vc = (
        df["misinfo_type_llm"]
        .fillna("MISSING")
        .value_counts(dropna=False)
        .to_frame("count")
    )
    vc["pct"] = (vc["count"] / n_total * 100).round(2)
    print(vc)

    # save
    out_path = os.path.join(folder, output_name)
    df.to_csv(out_path, index=False)
    print(f"\nSaved merged file → {out_path}")

    return df


def extract_empty_llm_rows(path, out_path, id_col="noteId", response_col="llm_response"):
    
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False, low_memory=False)

    if id_col not in df.columns:
        raise KeyError(f"Missing id_col='{id_col}'")
    if response_col not in df.columns:
        raise KeyError(f"Missing response_col='{response_col}'")

    # ensure id is string + strip
    d = df.copy()
    d[id_col] = d[id_col].astype(str).str.strip()

    # sanity: id must be unique for safe merge-back
    n_dups = int(d[id_col].duplicated().sum())
    if n_dups:
        raise ValueError(f"'{id_col}' not unique: {n_dups} duplicates. Use a different key.")

    empty_mask = d[response_col].isna() | (d[response_col].astype(str).str.strip() == "")
    rerun_df = d.loc[empty_mask].copy()

    rerun_df.to_csv(out_path, index=False)
    print(f"Saved rerun df with {len(rerun_df)} rows -> {out_path}")
    return rerun_df

def merge_rerun_back(
    original_path,
    rerun_path,
    out_rerun_path,
    id_col="noteId",
    orig_response_col="llm_response",
    rerun_response_col="llm_response",
    orig_label_col="misinfo_type_llm",
    rerun_label_col="misinfo_type_llm",
    orig_conf_col="confidence",
    rerun_conf_col="confidence",
    orig_rat_col="rationale",
    rerun_rat_col="rationale",
):
    original_df = pd.read_csv(original_path, dtype=str, keep_default_na=False, na_filter=False, low_memory=False)
    rerun_df = pd.read_csv(rerun_path, dtype=str, keep_default_na=False, na_filter=False, low_memory=False)

    df = original_df.copy()
    rr = rerun_df.copy()

    original_cols = df.columns.tolist()

    for d in (df, rr):
        if id_col not in d.columns:
            raise KeyError(f"Missing '{id_col}'")
        d[id_col] = d[id_col].astype(str).str.strip()

    # uniqueness checks
    if df[id_col].duplicated().any():
        raise ValueError(f"'{id_col}' is not unique in original_df")
    if rr[id_col].duplicated().any():
        raise ValueError(f"'{id_col}' is not unique in rerun_df")

    # merge rerun columns onto original
    cols_needed = [id_col, rerun_response_col, rerun_label_col, rerun_conf_col, rerun_rat_col]
    for c in cols_needed:
        if c not in rr.columns:
            raise KeyError(f"Rerun df missing column '{c}'")

    rr_small = rr[cols_needed].set_index(id_col)
    df = df.set_index(id_col)

    # only overwrite where original response is empty
    empty_mask = df[orig_response_col].isna() | (df[orig_response_col].astype(str).str.strip() == "")

    # update from rerun (only for ids present in rr)
    idx = df.index.intersection(rr_small.index)
    idx_to_update = idx[empty_mask.loc[idx]]

    df.loc[idx_to_update, orig_response_col] = rr_small.loc[idx_to_update, rerun_response_col]
    df.loc[idx_to_update, orig_label_col] = rr_small.loc[idx_to_update, rerun_label_col]
    df.loc[idx_to_update, orig_conf_col] = rr_small.loc[idx_to_update, rerun_conf_col]
    df.loc[idx_to_update, orig_rat_col] = rr_small.loc[idx_to_update, rerun_rat_col]

    df = df.reset_index()
    # restore original column order
    df = df[original_cols]

    print(f"Updated rows: {len(idx_to_update)}")
    df.to_csv(out_rerun_path, index=False)

    empty_after = (df["llm_response"].isna() | (df["llm_response"].astype(str).str.strip() == "")).sum()
    print("Empty llm_response after merge:", empty_after)
    print(df["misinfo_type_llm"].fillna("MISSING").value_counts(normalize=True).mul(100).round(2))


    return df



def merge_disagreement_columns(
    full_csv,
    disagreement_csv,
    out_csv=None,
    id_col="noteId",
):
    """
    Merge disagreement rerun-related columns from disagreement CSV into the full CSV.
    Only rows present in disagreement_csv get values; others remain empty.
    """

    rerun_cols = [
        "misinfo_type_llm_rerun",
        "confidence_rerun",
        "rationale_rerun",
        "llm_rerun_response",
    ]

    df_full = pd.read_csv(full_csv, dtype=str, keep_default_na=False)
    df_dis  = pd.read_csv(disagreement_csv, dtype=str, keep_default_na=False)

    original_cols = df_full.columns.tolist()

    # sanity checks
    if id_col not in df_full.columns or id_col not in df_dis.columns:
        raise ValueError(f"Missing id column '{id_col}'")

    missing = [c for c in rerun_cols if c not in df_dis.columns]
    if missing:
        raise ValueError(f"Disagreement CSV missing columns: {missing}")

    # normalize ids
    df_full[id_col] = df_full[id_col].astype(str).str.strip()
    df_dis[id_col]  = df_dis[id_col].astype(str).str.strip()

    if df_full[id_col].duplicated().any():
        raise ValueError("ID is not unique in full dataset")
    if df_dis[id_col].duplicated().any():
        raise ValueError("ID is not unique in disagreement dataset")

    # ensure columns exist in full df (empty by default)
    for c in rerun_cols:
        if c not in df_full.columns:
            df_full[c] = ""

    # keep only needed columns from disagreement df
    df_dis_small = df_dis[[id_col] + rerun_cols].set_index(id_col)
    df_full = df_full.set_index(id_col)

    # update values (only matching rows get filled)
    df_full.update(df_dis_small)

    df_full = df_full.reset_index()

    # restore original column order 
    new_cols = [c for c in df_full.columns if c not in original_cols]
    df_full = df_full[original_cols + new_cols]

    # stats
    n_total = len(df_full)
    n_filled = (
        df_full["misinfo_type_llm_rerun"].astype(str).str.strip() != ""
    ).sum()

    print("\n=== RERUN MERGE STATS ===")
    print(f"Total rows                     : {n_total:,}")
    print(f"Rows with rerun values filled  : {n_filled:,}")
    print(f"Share rerun-updated            : {n_filled / n_total * 100:.2f}%")

    if out_csv:
        df_full.to_csv(out_csv, index=False)
        print(f"\nSaved merged file → {out_csv}")

    return df_full






keys = "image"
input_path= f"renew_data/tweet_features_{keys}_normalized_after_retry.csv"



"""
    First, add misinformation labels based on keywords
    Save on column misinfo_type_keys
"""
output_path= f"renew_data/misinfo/tweet_features_{keys}_normalized_after_retry_with_misinfo_keys.csv"

# process_file(input_path=input_path, output_path=output_path)

"""
    Then take the file and run gemma 3 to get the misinformation labels
    Required functions in misinformation/misinformation_gemma.py
    + Use helper to merge the subfiles (checkpoint chunks in one file per keyword)
    + rerun for the empty rows if needed
"""
############################################################
################### Merge the subfiles #####################
path = f"renew_data/misinfo/parts/{keys}/"
output_name = f"tweet_features_{keys}_misinfo_llm.csv"
# merge_final_batches(path, output_name=output_name)
############################################################
############################################################
################# Extract empty LLM rows ###################
input_path = f"renew_data/misinfo/tweet_features_{keys}_misinfo_llm1.csv"
empty_llm_path = f"renew_data/misinfo/tweet_features_{keys}_misinfo_llm_empty.csv"
# extract_empty_llm_rows(input_path, empty_llm_path)
############################################################
#################### Merge rerun back ######################
rerun_path = f"renew_data/misinfo/empty_rerun_tweet_features_{keys}_misinfo_llm.csv"
out_merged_path = f"renew_data/misinfo/tweet_features_{keys}_misinfo_llm_after_rerun.csv"
# merge_rerun_back(input_path, rerun_path, out_merged_path)
############################################################


"""
    Then find entries where misinfo_type_keys != misinfo_type_llm, excluding 'other' from keys
    Save in a disagreement CSV file
"""
# THIS CSV MUST CONTAIN BOTH misinfo_type_keys AND misinfo_type_llm
input_path=f"renew_data/misinfo/tweet_features_{keys}_misinfo_llm.csv"
disagreeement_path = f"renew_data/misinfo/tweet_features_{keys}_keys_llm_disagreements.csv"
# find_disagreements(Path(input_path), Path(disagreeement_path))

"""
    Then Re-run the gemma 3 inference on the disagreement CSV file
    Merge results to the original CSV and print stats
"""
disagreement_rerun_path = f"renew_data/misinfo/disagreement_tweet_features_{keys}_keys_llm.csv"
output_path = input_path.replace(".csv", "_with_disagreement_rerun.csv")
# merge_disagreement_columns(full_csv=input_path, disagreement_csv=disagreement_rerun_path, out_csv=output_path)


"""
    Then decide the final misinformation label 
"""
# build_final_misinfo_labels(input_path=output_path, output_path=output_path.replace(".csv", "_final.csv"))


"""
    The produced file is ready to be added in the main dataset
    Proceed to fetch_new_notes_pipeline.py
"""








