# Dataset Renewal Pipeline

End-to-end flow: Community Notes TSV → filtered notes → tweet/media features → misinformation labels → append to existing dataset.

> **Recommended approach**: run `dataset.py` step by step, reading the comments and configuration before each step. The file is structured so you can comment out later steps and run only what you need.

---

## Running the pipeline

```bash
cd renew_data/
python dataset.py
```

All paths in `dataset.py` are relative to `renew_data/`. The script calls `os.chdir` at startup so this is consistent whether you run it directly or import it.

---

## Step 1 — Filter New Community Notes

**Script:** `fetch_new_notes_pipeline.py`  
**Config in `dataset.py`:** `TSV_PATHS`, `INCREMENTAL`, `PATH_CURRENT_IMAGE`, `PATH_CURRENT_VIDEO`

Downloads and filters the Community Notes TSV for image and video modalities. Two modes:

- `INCREMENTAL = False` — first run, processes the full TSV. Outputs: `notes-data-image-set.csv`, `notes-data-video-set.csv`
- `INCREMENTAL = True` — renewal run, keeps only notes not already in the existing dataset. Outputs: `notes-data-image-set_new.csv`, `notes-data-video-set_new.csv`

`TSV_PATHS` can be a single path or a list of paths (multiple TSVs are concatenated before filtering).

---

## Step 2 — Fetch Tweet Data and Download Media

**Script:** `tweet_extraction.py`  
**Config in `dataset.py`:** `MODE`, `COOKIES_PATH`, `MEDIA_DIR`, `START_ID`, `END_ID`

Fetches tweet metadata and downloads media (images/videos) for each note.

- `MODE`: `"guest"` (no login) or `"login"` (requires cookies file)
- `MEDIA_DIR`: base folder for media; subfolders `image_set/` and `video_set/` are created automatically
- `START_ID` / `END_ID`: optional row slice for batched runs (e.g. `START_ID=0, END_ID=200`). Leave as `None` to process all notes.

---

## Step 3 — Merge and Integrate Data

**Script:** `integrate_tweets_notes_data.py`

### 3a — Merge tweet CSV shards _(only needed if START_ID/END_ID were used)_

If you ran tweet extraction in batches, this merges all `notes-data-{image/video}-set-new_{start}-{end}_tweet_data.csv` shards numerically into a single file per modality.

### 3b — Merge notes with tweet data

Joins notes CSVs with tweet CSVs. Filters:

- **image** rows: keeps only rows where `media` is non-empty
- **video** rows: keeps only rows where `full_text` is non-empty

Outputs: `notes-tweets-data-image-set-new.csv`, `notes-tweets-data-video-set-new.csv`

---

## Step 4 — Misinformation Labeling

**Script:** `misinfo_pipeline.py` (orchestrates `src/misinformation/`)  
**Config in `misinfo_pipeline.py`:** `MODEL`, `TEMPERATURE`, `MAX_TOKENS`, `SAVE_EVERY`

Runs four sub-steps for both modalities:

1. **Keyword labels** (`misinformation_keywords.py`) → `misinfo_type_keys`
2. **Gemma first pass** (`misinformation_gemma.py`) → `misinfo_type_llm`
3. **Gemma rerun on disagreements** — internally filters rows where keyword and LLM labels disagree, reruns Gemma on those only, merges results back into the full slice → `misinfo_type_llm_rerun`
4. **Final label fusion** (`misinformation_final_label.py`) → `misinfo_type_final`

All outputs land in `misinfo/`. Gemma steps save checkpoint chunks every `SAVE_EVERY` rows for long runs.

You can optionally pass `start`/`end` to `run_misinfo_pipeline()` in `dataset.py` to batch the Gemma inference (e.g. `start=0, end=200`). The same slice is applied consistently to all four sub-steps.

`misinfo_pipeline.py` can also be run standalone — configure `START`/`END` in its `__main__` block.

---

## Step 4b — Collect Final Misinfo CSVs

**Function:** `collect_misinfo_outputs()` in `misinfo_pipeline.py`

Moves the final labeled files out of `misinfo/` into `renew_data/`:

- **Batched run**: merges all `*_misinfo_final_S_E.csv` files numerically → `*_misinfo_final.csv`
- **Full-set run**: copies the single `*_misinfo_final.csv`

Call `collect_misinfo_outputs()` once all batches are complete (it is called automatically when running through `dataset.py`).

---

## Step 5 — Integrate New Entries into the Existing Dataset

**Script:** `merge_new_with_existing.py`  
**Config in `dataset.py`:** `PATH_CURRENT_IMAGE`, `PATH_CURRENT_VIDEO`

Three sub-steps, run automatically in order:

1. **Align schema** — assigns new `iteration_id` values continuing from the existing dataset's max, drops intermediate columns (`confidence_rerun`), adds empty columns for any fields present in the existing dataset but missing in the new data (with a warning)
2. **Safety check** — verifies column order matches and checks for `noteId` overlaps before appending
3. **Append** — concatenates and saves as `*_NEW.csv` next to the existing dataset files

---

## Scripts reference

| Script                           | Role                                 |
| -------------------------------- | ------------------------------------ |
| `dataset.py`                     | End-to-end orchestrator              |
| `fetch_new_notes_pipeline.py`    | TSV filtering, incremental dedup     |
| `tweet_extraction.py`            | Tweet fetch + media download         |
| `integrate_tweets_notes_data.py` | Merge shards, merge notes+tweets     |
| `misinfo_pipeline.py`            | Misinformation labeling orchestrator |
| `merge_new_with_existing.py`     | Schema alignment + dataset append    |
