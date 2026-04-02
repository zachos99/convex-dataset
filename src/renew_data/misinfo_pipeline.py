"""
Misinformation labeling pipeline orchestrator.

Input: output of `renew_data/integrate_new_data_pipeline.merge_notes_with_tweet_data()`:
  - notes-tweets-data-image-set-new.csv
  - notes-tweets-data-video-set-new.csv

Run order (see `src/misinformation/misinfo_labels.md`):
  1) misinformation_keywords.py      -> misinfo_type_keys
  2) misinformation_gemma.py         -> misinfo_type_llm (+ confidence, rationale, llm_response)
  3) misinformation_gemma.py rerun   -> internally filters disagreements, reruns Gemma on them,
                                        merges misinfo_type_llm_rerun back into the full slice,
                                        saves the combined result as ..._with_misinfo_gemma_final_label.csv
  4) misinformation_final_label.py   -> misinfo_type_final (+ source flag)
"""

import os
import re
import shutil
import sys
from pathlib import Path
import pandas as pd

# Allow importing from src/ when running from renew_data/.
repo_root = Path(__file__).resolve().parent.parent
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from misinformation.misinformation_keywords import process_file
from misinformation.misinformation_gemma import extract_misinfo_batch
import misinformation.misinformation_gemma as mg
from misinformation.misinformation_final_label import build_final_misinfo_labels

# -----------------------
# CONFIG (edit these)
# -----------------------

# Inputs (produced by renew_data/integrate_new_data_pipeline.py)
INPUT_IMAGE = "notes-tweets-data-image-set-new.csv"
INPUT_VIDEO = "notes-tweets-data-video-set-new.csv"

# Output directory for all misinfo intermediate files
OUT_DIR = Path("misinfo")

# Media base dirs 
BASE_MEDIA_DIR_IMAGE = Path("tweet_media/image_set")
BASE_MEDIA_DIR_VIDEO = Path("tweet_media/video_set")

# LLM inference settings
MODEL = "gemma-3-27b-it"
TEMPERATURE = 0.1
MAX_TOKENS = 512
SAVE_EVERY = 200


def run_modality(keys, input_csv, base_media_dir, start=None, end=None):
    """Run full misinfo pipeline for one modality ('image' or 'video')."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    input_csv = Path(input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV for {keys}: {input_csv}")

    # Shared batch tag for output filenames — applied consistently to every step.
    # process_file slices the data once and saves the batch file; all downstream
    # steps read that file fully (no re-slicing) and use btag for naming only.
    btag = f"{start}_{end}" if (start is not None and end is not None) else None
    batch_suffix = f"_{btag}" if btag else ""

    # 1) Keyword labels — process_file does the actual slicing here.
    out_keys = OUT_DIR / f"{input_csv.stem}_with_misinfo_keys{batch_suffix}.csv"
    out_keys_path, _, _ = process_file(input_path=str(input_csv), output_path=str(out_keys), start=start, end=end)

    # 2) Gemma first pass — reads the already-sliced keys file fully (no re-slicing).
    #    batch_tag carries the naming suffix without triggering another iloc slice.
    #    Output: ..._with_misinfo_gemma[_start_end].csv
    mg.BASE_MEDIA_DIR = Path(base_media_dir)
    out_gemma = extract_misinfo_batch(
        csv_path=out_keys_path,
        model=MODEL,
        temp=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        save_every=SAVE_EVERY,
        rerun=False,
        batch_tag=btag,
    )

    # 3) Gemma rerun — same: reads the gemma file fully, batch_tag for naming only.
    #    Output: ..._with_misinfo_gemma[_start_end]_rerun.csv
    out_rerun = extract_misinfo_batch(
        csv_path=out_gemma,
        model=MODEL,
        temp=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        save_every=SAVE_EVERY,
        rerun=True,
        batch_tag=btag,
    )

    # 4) Final label fusion.
    #    Output: ..._misinfo_final[_start_end].csv
    out_final = str(OUT_DIR / f"{input_csv.stem}_misinfo_final{batch_suffix}.csv")
    build_final_misinfo_labels(input_path=out_rerun, output_path=out_final)

    return {
        "keys_csv": str(out_keys_path),
        "gemma_csv": str(out_gemma),
        "rerun_csv": str(out_rerun),
        "final_csv": out_final,
    }


def run_pipeline(image_csv=None, video_csv=None, media_dir=None, start=None, end=None):
    img = image_csv or INPUT_IMAGE
    vid = video_csv or INPUT_VIDEO
    if media_dir is not None:
        img_media = Path(media_dir) / "image_set"
        vid_media = Path(media_dir) / "video_set"
    else:
        img_media = BASE_MEDIA_DIR_IMAGE
        vid_media = BASE_MEDIA_DIR_VIDEO
    out = {}
    out["image"] = run_modality("image", img, img_media, start=start, end=end)
    out["video"] = run_modality("video", vid, vid_media, start=start, end=end)
    return out


def collect_misinfo_outputs(image_csv=None, video_csv=None, dest_dir="."):
    """
    Call this once all batches are complete to produce the final per-modality CSV
    in dest_dir (default: current directory = renew_data/).

    - Batch run (start/end were used): finds all *_misinfo_final_S_E.csv files in
      OUT_DIR, merges them in numerical order, and saves *_misinfo_final.csv to dest_dir.
    - Full-set run (no start/end): simply copies the existing *_misinfo_final.csv
      from OUT_DIR to dest_dir.
    """
    img_stem = Path(image_csv or INPUT_IMAGE).stem
    vid_stem = Path(video_csv or INPUT_VIDEO).stem
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    result = {}
    for modality, stem in (("image", img_stem), ("video", vid_stem)):
        dest_path = dest_dir / f"{stem}_misinfo_final.csv"

        # Look for batch files: stem_misinfo_final_S_E.csv
        pattern = re.compile(rf"^{re.escape(stem)}_misinfo_final_(\d+)_(\d+)\.csv$")
        batch_files = []
        for f in OUT_DIR.iterdir():
            m = pattern.match(f.name)
            if m:
                batch_files.append((int(m.group(1)), f))

        if batch_files:
            batch_files.sort(key=lambda x: x[0])   # sort by start id
            merged = pd.concat(
                [pd.read_csv(f) for _, f in batch_files], ignore_index=True
            )
            merged.to_csv(dest_path, index=False)
            print(f"[{modality}] Merged {len(batch_files)} batch(es) → {dest_path}")
        else:
            src = OUT_DIR / f"{stem}_misinfo_final.csv"
            if src.exists():
                shutil.copy2(src, dest_path)
                print(f"[{modality}] Copied {src} → {dest_path}")
            else:
                print(f"[{modality}] Warning: no misinfo final file found in {OUT_DIR}")
                dest_path = None

        result[modality] = str(dest_path) if dest_path else None

    return result


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Configure your run here
    START = 50
    END   = 70

    # outputs = run_pipeline(start=START, end=END)
    # print("\n=== Misinfo pipeline outputs ===")
    # for modality in ("image", "video"):
    #     o = outputs[modality]
    #     print(f"\n[{modality}]")
    #     print(f"  keywords : {o['keys_csv']}")
    #     print(f"  gemma    : {o['gemma_csv']}")
    #     print(f"  rerun    : {o['rerun_csv']}")
    #     print(f"  final    : {o['final_csv']}")

    # Run this once all batches are done to merge and move the final CSVs to renew_data/
    collect_misinfo_outputs()

