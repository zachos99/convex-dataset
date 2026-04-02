import os
import pandas as pd

from fetch_new_notes_utils import process_tsv, searchMediaKeywords, combine_tags, find_new_notes, inspect_timestamps



def process_tsv_pipeline(path_tsv, keyword_type="image"):
    """
        Read tsv --> process pipeline --> save final csv 
        1) Process TSV to DataFrame
        2) Filter by media keywords (image or video) and misleading classification
        3) Combine misleading tags
        4) Keep specific columns (rename 'summary' to 'noteText')
        5) Remove duplicate notes
        6) Add uniqueID column
        7) Save final CSV
    """

    # Read one or multiple TSV files
    print("Processing TSV file(s)...")
    if isinstance(path_tsv, (list, tuple)):
        if len(path_tsv) == 0:
            raise ValueError("path_tsv list is empty. Provide at least one TSV path.")

        frames = []
        for tsv_path in path_tsv:
            df_part = process_tsv(tsv_path)
            if df_part is None:
                raise ValueError(f"Failed to process TSV file: {tsv_path}")
            frames.append(df_part)

        df = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    else:
        df = process_tsv(path_tsv)
        if df is None:
            raise ValueError(f"Failed to process TSV file: {path_tsv}")

    # Filter by media keywords (image or video)
    print("Filtering by media keywords...")
    df = searchMediaKeywords(df, media_type=keyword_type) 

    print("Combining misleading tags...")
    df = combine_tags(df)

    # tweetUrl is created in searchMediaKeywords(). Keep this guard so reordering/refactors don't break.
    if "tweetUrl" not in df.columns and "tweetId" in df.columns:
        df["tweetUrl"] = "https://x.com/i/web/status/" + df["tweetId"].astype(str)

    # Keep specific columns
    columns_to_keep = ["noteId","createdAtMillis", "tweetId", "tweetUrl", "classification","trustworthySources", "isMediaNote","misleadingTags", "summary"] 
    df = df[columns_to_keep]

    # Rename 'summary' to 'noteText'
    df = df.rename(columns={"summary": "noteText"})

    # Drop duplicates on "noteText"
    print("Dropping duplicates...")
    df = df.drop_duplicates(subset="noteText", keep="first")

    # Add uniqueID column
    print("Adding uniqueID column...")
    df.insert(0, "uniqueID", range(1, len(df) + 1))

    # Save final CSV
    print("Saving final CSV...")
    if isinstance(path_tsv, (list, tuple)):
        base_dir = os.path.dirname(path_tsv[0]) if path_tsv else "."
    else:
        base_dir = os.path.dirname(path_tsv) or "."
    output_path = os.path.join(base_dir, f"notes-data-{keyword_type}-set.csv")

    return df, output_path


def run_notes_pipeline_both_modalities(
    path_tsv,
    incremental=False,
    path_current_image=None,
    path_current_video=None,
):
    """
    Run the notes pipeline for both modalities (image + video).

    Args:
        path_tsv (str): Path to raw Community Notes TSV.
        incremental (bool):
            - False: first-time/full run. Returns the 2 filtered CSVs.
            - True: renewal run. Compares against checkpoints and returns the 2 new csv files.
        path_current_image (str|None): required when incremental=True.
        path_current_video (str|None): required when incremental=True.

    Returns:
        dict: {"image": <output_path>, "video": <output_path>}
    """
    if incremental and (not path_current_image or not path_current_video):
        raise ValueError(
            "When incremental=True, you must provide both path_current_image and path_current_video."
        )

    modalities = ["image", "video"]
    checkpoint_paths = {
        "image": path_current_image,
        "video": path_current_video,
    }
    outputs = {}

    for modality in modalities:
        print(f"\n=== Running modality: {modality} ===")

        df_filtered, path_filtered = process_tsv_pipeline(path_tsv, keyword_type=modality)
        path_incremental = path_filtered.replace(".csv", "_new.csv")

        if not incremental:
            df_filtered.to_csv(path_filtered, index=False)
            outputs[modality] = path_filtered
            inspect_timestamps(path=path_filtered)
            continue

        path_current = checkpoint_paths[modality]
        if not os.path.exists(path_current):
            raise FileNotFoundError(
                f"Checkpoint file not found for modality '{modality}': {path_current}. "
                "Check the provided checkpoint path."
            )

        df_current = pd.read_csv(path_current, dtype={"noteId": str}, low_memory=False)
        path_only_new_notes = find_new_notes(
            df_old=df_current,
            df_new=df_filtered,
            out_path=path_incremental,
            save=True,
        )
        outputs[modality] = path_only_new_notes
        if path_only_new_notes:
            inspect_timestamps(path=path_only_new_notes)

    return outputs




"""
    To run this step independently, run:
"""

if __name__ == "__main__":

    
    # You can pass one or multiple TSV paths
    # eg "notes-00000.tsv" or ["notes-00000.tsv", "notes-00001.tsv"]
    TSV_PATH = ["notes-00001.tsv"]

    # If you are running the pipeline for the first time, set INCREMENTAL to False
    # If you are running the pipeline for data renewal, set INCREMENTAL to True
    INCREMENTAL = True  

    # Path to the current dataset csv files (for data renewal case)
    PATH_CURRENT_IMAGE = "dataset-image-set.csv"
    PATH_CURRENT_VIDEO = "dataset-video-set.csv"

    outputs = run_notes_pipeline_both_modalities(
        path_tsv=TSV_PATH,
        incremental=INCREMENTAL,
        path_current_image=PATH_CURRENT_IMAGE,
        path_current_video=PATH_CURRENT_VIDEO,
    )

    print("\nPipeline outputs:")
    for modality, path_out in outputs.items():
        print(f" - {modality}: {path_out}")



