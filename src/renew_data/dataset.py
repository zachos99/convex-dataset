"""
End-to-end dataset renewal pipeline.

Run from renew_data/:  python dataset.py

Steps:
  1) Filter new Community Notes from TSV (image + video modalities)
  2) Fetch tweet data + download media for each note
  3a) Merge tweet CSV shards (only if START_ID/END_ID were used)
  3b) Merge notes with tweet data
  4) Misinformation labeling (keywords → Gemma → rerun → final label fusion)
  4b) Collect per-modality misinfo final CSVs into renew_data/
  5) Align schema + append new entries to existing dataset

Configure each section below before running.
See renew_data/integrate-new-data.md for the full pipeline description.
"""
import os
import sys
from pathlib import Path

from fetch_new_notes_pipeline import run_notes_pipeline_both_modalities
from tweet_extraction import run_tweet_extraction_sync
from integrate_tweets_notes_data import (
    merge_csvs_numerical,
    validate_media_paths,
    merge_notes_with_tweet_data,
)

# Allow importing misinfo_pipeline (same folder) and src/misinformation/* (its deps)
_here = Path(__file__).resolve().parent
_src  = _here.parent / "src"
for _p in (_here, _src):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from misinfo_pipeline import run_pipeline as run_misinfo_pipeline, collect_misinfo_outputs
from merge_new_with_existing import run_integration_pipeline


"""
    CONFIG 
"""

"""
    Fetch New Notes Pipeline
"""
# Community Notes TSV(s)
# You can pass one or multiple TSV paths
# eg "notes-00000.tsv" or ["notes-00000.tsv", "notes-00001.tsv"]
TSV_PATHS = ["notes-00001.tsv"]  

# Incremental mode
# If you are running the pipeline for the first time, set INCREMENTAL to False
# If you are running the pipeline for data renewal, set INCREMENTAL to True
INCREMENTAL = True

# Path to the current dataset csv files (for data renewal case)
PATH_CURRENT_IMAGE = "dataset-image-set.csv"
PATH_CURRENT_VIDEO = "dataset-video-set.csv"


"""
    Tweet Extraction 
        - This pipeline is used to extract tweets and download media
        - You have to install and set up Twikit library first
        - Use either guest mode or login mode
            - For login mode, you need to use credentials, then save a cookies file --> setup on tweet_extraction.py
"""

# Mode can be "guest" or "login" --> Set it up on tweet_extraction.py
MODE = "guest"  
COOKIES_PATH = None  # Path to the cookies file for login mode (optional)

# Directory to save the downloaded tweet media files
# Two subdirectories will be created: "image" and "video" for each subset
MEDIA_DIR = "tweet_media"

# OPTIONAL: Set the start and end ids to run for a specific range of notes
START_ID = None #0
END_ID = None #100



def run_pipeline():

    """
    1) Notes Pipeline
        - Get all the new Community Notes from the new TSV checkpoints
        - Filter the notes for the image and video modalities and create the two sets
    """
    notes_outputs = run_notes_pipeline_both_modalities(
        path_tsv=TSV_PATHS,
        incremental=INCREMENTAL,
        path_current_image=PATH_CURRENT_IMAGE,
        path_current_video=PATH_CURRENT_VIDEO,
    )
    notes_image_csv = notes_outputs.get("image")
    notes_video_csv = notes_outputs.get("video")

    if not notes_image_csv or not notes_video_csv:
        raise RuntimeError(
            "Step 1 did not produce both image and video note CSV paths. "
            "Got: %r" % (notes_outputs,)
        )

    """
    2) Tweet Extraction Pipeline
        - This pipeline is used to extract tweets and download media
        - This will produce per modality:
            - One combined CSV (per modality)
            - One shard CSV per batch
                --> so a long run produces multiple shards plus the single combined file.
    """
    tweet_outputs = run_tweet_extraction_sync(
        input_path_image=notes_image_csv,
        input_path_video=notes_video_csv,
        image_dir=MEDIA_DIR,
        mode=MODE,
        start_id=START_ID,
        end_id=END_ID,
        cookies_path=COOKIES_PATH,
    )

    """
        3a) Merge tweet data subsets into a single CSV (per modality)
            - If you run the pipeline for a specific range of notes, using start/end ids, you need first to merge the subsets into a single CSV (e.g. the files "notes-data-{image/video}-set-new_{start_id}-{end_id}_tweet_data.csv")
            - For this case we provide the helper function merge_csvs_numerical()
            -If you run the pipeline for the first time, you don't need to merge the subsets, because the tweet extraction pipeline already produced the required CSVs
            - You can now delete the shard csvs as they were for debugging purposes
    """

    if START_ID is not None and END_ID is not None:
        fixed_paths = merge_csvs_numerical(folder=".")
        fixed_image_path = fixed_paths.get("image")
        fixed_video_path = fixed_paths.get("video")
        if not fixed_image_path or not fixed_video_path:
            raise RuntimeError(
                "merge_csvs_numerical() did not find expected run CSVs for both modalities."
            )
    else:
        # No start/end run merging needed; tweet extraction already produced the required CSVs
        fixed_image_path = tweet_outputs.get("image")
        fixed_video_path = tweet_outputs.get("video")
        if not fixed_image_path or not fixed_video_path:
            raise RuntimeError(
                "Step 2 did not return expected tweet CSV output paths for both modalities."
            )

    # Optionally validate that all media paths are valid
    # validate_media_paths(fixed_image_path, "tweet_media/image_set")
    # validate_media_paths(fixed_video_path, "tweet_media/video_set")

    """
        3b) Merge Tweet CSVs with Notes data
            - Merge the tweet data CSV with the notes data CSV to create the combined notes+tweet data CSV
            - This produces one combined CSV per modality like "notes-tweets-data-{image/video}-set-new.csv"
    """
    notes_tweets_outputs = merge_notes_with_tweet_data(
        fixed_image_path=fixed_image_path,
        fixed_video_path=fixed_video_path,
        notes_image_path=notes_image_csv,
        notes_video_path=notes_video_csv,
    )


    """
    - Up until this step, we have notes+tweets+media data for 
        each new note we fetched 
        - We keep only those files for now on ("notes-tweets-data-
        {image/video}-set-new.csv")
        - Now we can proceed with the misinformation type extraction 
        pipeline
    """
    """
    4) Misinformation labeling pipeline
        - Runs keyword labels, Gemma first pass, Gemma rerun on disagreements,
          and final label fusion for both modalities.
        - LLM settings (MODEL, TEMPERATURE, SAVE_EVERY) are configured in misinfo_pipeline.py
        - Outputs land in the "misinfo/" subfolder
        - Optionally pass start/end to batch Gemma inference, e.g. start=0, end=100
          (applied consistently to every labeling step)
    """
    misinfo_outputs = run_misinfo_pipeline(
        image_csv=notes_tweets_outputs.get("image"),
        video_csv=notes_tweets_outputs.get("video"),
        media_dir=MEDIA_DIR,
        # start = 0,
        # end = 100,
    )

    """
    4b) Collect final misinfo CSVs into renew_data/
        - If batches were used: merges all *_misinfo_final_S_E.csv files numerically.
        - If full-set run: copies the single *_misinfo_final.csv from misinfo/.
        - dest_dir="." resolves to renew_data/ in both standalone and dataset.py runs.
    """
    collected = collect_misinfo_outputs(
        image_csv=notes_tweets_outputs.get("image"),
        video_csv=notes_tweets_outputs.get("video"),
        dest_dir=".",
    )

    """
    5) Integrate new entries into the existing dataset
        - Aligns new data schema (iteration_id, column order; adds empty cols if missing)
        - Runs a safety check before appending
        - Appends and saves as *_NEW.csv next to the existing dataset files
    """
    integrated = run_integration_pipeline(
        image_new_path=collected.get("image"),
        video_new_path=collected.get("video"),
        image_existing_path=PATH_CURRENT_IMAGE,
        video_existing_path=PATH_CURRENT_VIDEO,
    )

    return {
        "notes": notes_outputs,
        "tweets": tweet_outputs,
        "notes_tweets": notes_tweets_outputs,
        "misinfo": misinfo_outputs,
        "misinfo_final": collected,
        "integrated": integrated,
    }


if __name__ == "__main__":
    # Make relative paths behave like other scripts in this folder.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    outputs = run_pipeline()
    print("\n=== Pipeline outputs ===")
    print("Notes CSVs        :", outputs["notes"])
    print("Tweet CSVs        :", outputs["tweets"])
    print("Notes+Tweet CSVs  :", outputs["notes_tweets"])
    print("Misinfo CSVs      :", outputs["misinfo"])
    print("Misinfo final     :", outputs["misinfo_final"])
    print("Integrated        :", outputs["integrated"])
