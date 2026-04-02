import os
import time
import json
from pathlib import Path
from typing import Set
import re

import dotenv
from google import genai
from google.genai import types
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)

"""
    For this setup you can upload the images to the Google Cloud Storage bucket and then use the URL to call the API.
    Fix the both local and google cloud paths accordingly 
"""
dotenv.load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# ----------------------------
# CONFIG
# ----------------------------
MODEL = "gemma-3-27b-it"  # hosted Gemma on Gemini API 

CLASS = "ai" # "ai" or "miscaptioned"
LOCAL_FOLDER = f"gemma_test/tweet_images_{CLASS}"
BASE_URL = f"https://storage.googleapis.com/{CLASS}_images/"

RUN_START = 0          # inclusive index in the sorted image list
RUN_END = None         # exclusive; set to None for "to the end"
OUT_JSONL = f"gemma_test/gemma3_{CLASS}_{RUN_START}_{RUN_END}.jsonl"

MAX_NEW_IMAGES = None

MAX_OUTPUT_TOKENS = 32
TEMPERATURE = 0.1

SLEEP_BETWEEN_CALLS_S = 0.5
MAX_RETRIES = 4
RETRY_SLEEP_BASE = 1.5

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

SYSTEM_PROMPT = (
"""
You are an expert in detecting AI-generated images. You need to classify the image as AI or REAL.
Definitions:
- AI: the main depicted content was generated or substantially modified by a generative AI model.
- REAL: the main depicted content is authentic (e.g. photographed) and was not generated or substantially modified by a generative AI model.
Instructions:
- If the image is a screenshot or digital capture, classify based on whether the depicted content was generated or substantially modified by generative AI.
- Ignore minor edits (cropping, compression, color correction).
- Return exactly one token: AI or REAL.
- Do not include any explanation or punctuation.
"""
)

USER_TEXT = "Classify the image. Output exactly one token: AI or REAL."

# ----------------------------
# RESUME SUPPORT
# ----------------------------
def load_completed(jsonl_path):
    done: Set[str] = set()
    p = Path(jsonl_path)
    if not p.exists():
        return done

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "COMPLETED":
                rp = row.get("relative_path")
                if rp:
                    done.add(rp)
            elif row.get("status") == "EMPTY_FILE":
                rp = row.get("relative_path")
                if rp:
                    done.add(rp)  # treat empty files as done so we never revisit
    return done

# ----------------------------
# IMAGE LISTING
# ----------------------------
def list_images(folder):
    root = Path(folder)
    images = []
    empty = 0

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        if p.stat().st_size == 0:
            empty += 1
            continue
        images.append(p)

    if empty > 0:
        print(f"Skipped {empty} empty (0-byte) files (not sent to API).")

    return sorted(images)

def build_url(image_path):
    rel = image_path.relative_to(Path(LOCAL_FOLDER)).as_posix()
    return BASE_URL.rstrip("/") + "/" + rel

def guess_mime(path):
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    return "application/octet-stream"

# ----------------------------
# CALL
# ----------------------------
def normalize_label(text):
    t = (text or "").strip().upper()
    if t in ("AI", "REAL"):
        return t
    t2 = t.strip('"').strip("'").strip()
    if t2 in ("AI", "REAL"):
        return t2
    return None

def extract_usage_counts(resp):
    """
    Gemini API: resp.usage_metadata has prompt_token_count, candidates_token_count, total_token_count.
    """
    usage = getattr(resp, "usage_metadata", None)
    if not usage:
        return 0, 0, 0

    prompt = int(getattr(usage, "prompt_token_count", 0) or 0)
    out = int(getattr(usage, "candidates_token_count", 0) or 0)
    total = int(getattr(usage, "total_token_count", 0) or 0)
    return prompt, out, total

def call_gemma_one(client, image_url, mime_type):
    last_err = None

    # Gemma on Gemini API: no system role; combine system+user into one text part.
    full_prompt = f"{SYSTEM_PROMPT}\n\n{USER_TEXT}".strip()

    contents = [
        types.Part.from_text(text=full_prompt),
        types.Part.from_uri(file_uri=image_url, mime_type=mime_type),
    ]

    config = types.GenerateContentConfig(
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=config,
            )

            out_text = (getattr(resp, "text", None) or "").strip()
            label = normalize_label(out_text)

            in_tok, out_tok, total_tok = extract_usage_counts(resp)

            return {
                "raw_output": out_text,
                "label": label,
                "prompt_tokens": in_tok,
                "completion_tokens": out_tok,
                "total_tokens": total_tok,
            }

        except Exception as e:
            last_err = e
            time.sleep(RETRY_SLEEP_BASE * (2 ** (attempt - 1)))

    raise RuntimeError(f"Failed after retries: {last_err}")



# ----------------------------
# MAIN
# ----------------------------
def gemma_inference():
    if not GEMINI_API_KEY:
        raise RuntimeError("Missing GEMINI_API_KEY environment variable.")
    client = genai.Client(api_key=GEMINI_API_KEY)

    images = list_images(LOCAL_FOLDER)
    # Deterministic shard selection
    if RUN_END is None:
        shard_images = images[RUN_START:]
    else:
        shard_images = images[RUN_START:RUN_END]
    print(f"Shard indices: [{RUN_START}:{'end' if RUN_END is None else RUN_END}] -> {len(shard_images)} images")

    completed = load_completed(OUT_JSONL)

    print(f"Total images: {len(shard_images)}")
    print(f"Already completed: {len(completed)}")
    print(f"Remaining: {len(shard_images) - len(completed)}\n")

    out_f = open(OUT_JSONL, "a", encoding="utf-8")

    cumulative_in = 0
    cumulative_out = 0
    cumulative_total = 0

    try:
        new_processed = 0

        # Log empty files once (optional, but helps reproducibility)
        root = Path(LOCAL_FOLDER)
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.stat().st_size == 0:
                rel = str(p.relative_to(root))
                if rel not in completed:
                    out_f.write(json.dumps({"relative_path": rel, "status": "EMPTY_FILE"}) + "\n")
                    out_f.flush()
                    completed.add(rel)

        for idx, path in enumerate(shard_images, start=RUN_START):
            rel = str(path.relative_to(Path(LOCAL_FOLDER)))

            if rel in completed:
                continue

            if MAX_NEW_IMAGES is not None and new_processed >= MAX_NEW_IMAGES:
                print(f"\nReached MAX_NEW_IMAGES={MAX_NEW_IMAGES}. Stopping.")
                break

            url = build_url(path)
            mime = guess_mime(path)

            try:
                r = call_gemma_one(client, url, mime)

                cumulative_in += r["prompt_tokens"]
                cumulative_out += r["completion_tokens"]
                cumulative_total += r["total_tokens"]

                row = {
                    "relative_path": rel,
                    "image_url": url,
                    "label": r["label"],
                    "raw_output": r["raw_output"],
                    "prompt_tokens": r["prompt_tokens"],
                    "completion_tokens": r["completion_tokens"],
                    "total_tokens": r["total_tokens"],
                    "status": "COMPLETED",
                }

                out_f.write(json.dumps(row) + "\n")
                out_f.flush()
                new_processed += 1

                print(
                    f"[{idx}] {rel} -> {r['label']} "
                    f"| in={r['prompt_tokens']} out={r['completion_tokens']} total={r['total_tokens']} "
                    f"| cum_in={cumulative_in} cum_out={cumulative_out} cum_total={cumulative_total}"
                )

            except Exception as e:
                out_f.write(json.dumps({"relative_path": rel, "status": "ERROR", "error": str(e)}) + "\n")
                out_f.flush()
                print(f"[{idx}] {rel} -> ERROR: {e}")

            time.sleep(SLEEP_BETWEEN_CALLS_S)

    finally:
        out_f.close()

    print("\n=== RUN TOTAL TOKENS (this run only) ===")
    print(f"in={cumulative_in} out={cumulative_out} total={cumulative_total}")


# ----------------------------
# JSONL -> ENRICHED CSV
# ----------------------------
def gemma_jsonl_to_enriched_csv(jsonl_path, dataset_csv, out_csv):
    """
    Convert a Gemma/Gemini JSONL results file into an enriched CSV by joining on image filename.
    - Uses `relative_path` as the image path
    - Extracts tweetId from the filename (photo_<tweetId>_...)
    - Joins to the main notes dataset via image filename (`image_key`)
    """
    TWEETID_RE = re.compile(r"photo_(\\d+)_")

    # Load JSONL rows
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not rows:
        raise RuntimeError(f"No valid JSON rows found in: {jsonl_path}")

    df = pd.DataFrame(rows)

    if "relative_path" not in df.columns:
        raise RuntimeError("JSONL rows must contain 'relative_path' field.")

    # Derive image + key and tweetId
    df["image"] = df["relative_path"].astype(str)
    df["image_key"] = df["image"].map(os.path.basename)

    def extract_tweet_id_from_key(k):
        m = TWEETID_RE.search(str(k))
        return m.group(1) if m else None

    df["tweetId"] = df["image_key"].map(extract_tweet_id_from_key)

    df_inf = df[["tweetId", "label", "image", "image_key", "status"]].copy()

    # Enrich from dataset CSV (same columns as in the OpenAI/Grok helpers)
    df_ds = pd.read_csv(dataset_csv, low_memory=False, dtype={"tweet_id"})

    tmp = df_ds[
        [
            "media",
            "tweet_id",
            "created_at_datetime",
            "tweetUrl",
            "noteText",
            "misinfo_type_final",
            "topic",
        ]
    ].copy()

    tmp["media"] = tmp["media"].fillna("").astype(str)
    tmp["media_item"] = tmp["media"].str.split(",")
    tmp = tmp.explode("media_item")
    tmp["media_item"] = tmp["media_item"].astype(str).str.strip()

    tmp["image_key"] = tmp["media_item"].map(os.path.basename)
    tmp = tmp[tmp["image_key"] != ""].drop_duplicates("image_key", keep="first")

    merged = df_inf.merge(
        tmp[
            [
                "image_key",
                "tweet_id",
                "created_at_datetime",
                "tweetUrl",
                "noteText",
                "misinfo_type_final",
                "topic",
            ]
        ],
        on="image_key",
        how="left",
        validate="many_to_one",
    )

    # Prefer dataset tweet_id if present; otherwise fall back to parsed tweetId
    merged["tweetId"] = merged["tweet_id"].where(
        merged["tweet_id"].notna() & (merged["tweet_id"].astype(str).str.strip() != ""),
        None,
    )
    merged["tweetId"] = merged["tweetId"].fillna(df_inf["tweetId"])

    out = merged[
        [
            "tweetId",
            "label",
            "misinfo_type_final",
            "topic",
            "created_at_datetime",
            "noteText",
            "image",
            "tweetUrl",
            "status",
        ]
    ].copy()

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out.to_csv(out_csv, index=False)

    print(f"Wrote {out_csv}")
    print(f"Matched {out['misinfo_type_final'].notna().sum()}/{len(out)} images via image_key")

# ----------------------------
# METRICS / EVALUATION
# ----------------------------
def _compute_label_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    tpr = recall_score(y_true, y_pred, pos_label=1)  # same as recall for AI class
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    # ROC-AUC only if both classes present and predictions not constant
    try:
        if len(np.unique(y_true)) == 2 and len(np.unique(y_pred)) > 1:
            roc_auc = roc_auc_score(y_true, y_pred)
        else:
            roc_auc = float("nan")
    except Exception:
        roc_auc = float("nan")

    return {
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "TPR (Recall)": float(tpr),
        "FPR": float(fpr),
        "Precision": float(precision),
        "F1": float(f1),
        "Accuracy": float(acc),
        "Balanced Accuracy": float(bal_acc),
        "ROC-AUC": float(roc_auc),
    }


def evaluate_gemma_ai_miscaptioned(path_ai, path_miscaptioned, label_col="label"):
    """
    Evaluate Gemma-3 (AI vs miscaptioned) using discrete labels only.

    - Ground truth_ai = AI images (y_true=1), path_miscaptioned = REAL/miscaptioned (y_true=0)
    - Prediction: label == 'AI' -> 1, label == 'REAL' -> 0
    """
    df_ai = pd.read_csv(path_ai)
    df_real = pd.read_csv(path_miscaptioned)

    df_ai["y_true"] = 1
    df_real["y_true"] = 0
    df = pd.concat([df_ai, df_real], ignore_index=True)

    n_total = len(df)

    # Normalize label
    raw_labels = df[label_col]
    is_null = raw_labels.isna()
    norm = (
        raw_labels.astype(str)
        .str.strip()
        .str.upper()
        .where(~is_null, other=None)
    )

    df["_label_norm"] = norm

    # Diagnostic: label distribution
    print("\n[Gemma-3] Label diagnostics:")
    counts = df["_label_norm"].value_counts(dropna=False)
    for val, cnt in counts.items():
        if val is None or (isinstance(val, float) and np.isnan(val)):
            name = "<NULL/NaN>"
        elif str(val).strip() == "":
            name = "<EMPTY>"
        else:
            name = str(val)
        print(f"  {name!r}: {cnt}")

    # Keep only rows with label in {'AI','REAL'}
    valid_mask = df["_label_norm"].isin({"AI", "REAL"})
    df_valid = df.loc[valid_mask].copy()
    n_used = len(df_valid)
    n_dropped = n_total - n_used

    y_true = df_valid["y_true"].to_numpy(dtype=int)
    y_pred = (df_valid["_label_norm"] == "AI").astype(int).to_numpy()

    metrics = _compute_label_metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print("\n==============================")
    print("Gemma-3 evaluation (AI vs miscaptioned)")
    print("==============================")
    print(f"Total rows (before drop): {n_total}")
    print(f"Dropped rows (label not in {{'AI','REAL'}}): {n_dropped}")
    print(f"Used rows: {n_used} (AI={int((y_true==1).sum())}, Real={int((y_true==0).sum())})\n")

    print("Confusion matrix (rows=true [Real, AI], cols=pred [Real, AI]):")
    print(cm)
    print()

    for k in [
        "TPR (Recall)",
        "FPR",
        "Precision",
        "F1",
        "Accuracy",
        "Balanced Accuracy",
        "ROC-AUC",
    ]:
        val = metrics.get(k, float("nan"))
        if isinstance(val, float):
            print(f"{k:18}: {val:.4f}")
        else:
            print(f"{k:18}: {val}")

    return metrics


def evaluate_gemma_ai_miscaptioned_overtime(
    path_ai,
    path_miscaptioned,
    date_col = "created_at_datetime",
    label_col = "label",
    start_year: int = 2023,
    end_year: int = 2026,
    plot: bool = False,
):
    """
    Evaluate Gemma-3 (AI vs miscaptioned) over time in 6-month spans (H1/H2).

    Buckets:
    - YYYY-H1: Jan–Jun
    - YYYY-H2: Jul–Dec

    Bucketing is always in separate H1/H2 spans.
    """
    df_ai = pd.read_csv(path_ai)
    df_real = pd.read_csv(path_miscaptioned)

    df_ai["y_true"] = 1
    df_real["y_true"] = 0
    df = pd.concat([df_ai, df_real], ignore_index=True)

    dates = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    df["_date"] = dates
    n_total_dates = len(df)
    n_parsed_dates = int(dates.notna().sum())
    print("\n[Date parsing diagnostic]")
    print(f"Parsed dates: {n_parsed_dates}/{n_total_dates} ({(n_parsed_dates / n_total_dates) if n_total_dates else float('nan'):.3%})")

    df["_year"] = dates.dt.year
    df["_half"] = np.where(dates.dt.month <= 6, 1, 2)
    df["_bucket"] = df["_year"].astype("Int64").astype(str) + "-H" + df["_half"].astype(int).astype(str)
    df["_bucket_start"] = pd.to_datetime(
        df["_year"].astype("Int64").astype(str)
        + "-"
        + np.where(df["_half"] == 1, "01", "07")
        + "-01",
        errors="coerce",
        utc=True,
    )

    # Normalize labels
    raw_labels = df[label_col]
    is_null = raw_labels.isna()
    norm = (
        raw_labels.astype(str)
        .str.strip()
        .str.upper()
        .where(~is_null, other=None)
    )
    df["_label_norm"] = norm

    # Filter: valid date, year range, valid labels
    df = df[df["_bucket_start"].notna()].copy()
    df = df[(df["_year"] >= start_year) & (df["_year"] <= end_year)].copy()
    df = df[df["_label_norm"].isin({"AI", "REAL"})].copy()

    # Expected buckets (even if empty)
    bucket_starts = []
    bucket_labels = []
    for y in range(int(start_year), int(end_year) + 1):
        for h, m in [(1, 1), (2, 7)]:
            lab = f"{y}-H{h}"
            bucket_starts.append(pd.Timestamp(year=y, month=m, day=1, tz="UTC"))
            bucket_labels.append(lab)

    bucket_index = pd.DataFrame({"bucket": bucket_labels, "bucket_start": bucket_starts})
    bucket_index = bucket_index.sort_values("bucket_start").reset_index(drop=True)

    print("\n==============================")
    title = f"Gemma-3 Over Time (Half-year, {start_year}–{end_year})"
    print(title)
    print("==============================\n")

    rows = []
    for b, b_start in zip(bucket_labels, bucket_starts):
        df_b = df[df["_bucket"] == b]
        if len(df_b) == 0:
            rows.append(
                {
                    "bucket": b,
                    "bucket_start": b_start,
                    "n_total": 0,
                    "n_ai": 0,
                    "n_real": 0,
                    "recall": float("nan"),
                    "fpr": float("nan"),
                    "balanced_accuracy": float("nan"),
                }
            )
            print(f"{b}  (no data)\n")
            continue

        y_true = df_b["y_true"].to_numpy(dtype=int)
        y_pred = (df_b["_label_norm"] == "AI").astype(int).to_numpy()
        metrics = _compute_label_metrics(y_true, y_pred)

        n_ai = int((y_true == 1).sum())
        n_real = int((y_true == 0).sum())

        rows.append(
            {
                "bucket": b,
                "bucket_start": b_start,
                "n_total": int(len(df_b)),
                "n_ai": n_ai,
                "n_real": n_real,
                "recall": float(metrics["TPR (Recall)"]),
                "fpr": float(metrics["FPR"]),
                "balanced_accuracy": float(metrics["Balanced Accuracy"]),
            }
        )

        print(f"{b}  (AI={n_ai}, Real={n_real}, Total={len(df_b)})")
        print(f"  Recall: {metrics['TPR (Recall)']:.3f}")
        print(f"  FPR:    {metrics['FPR']:.3f}")
        print(f"  BalAcc: {metrics['Balanced Accuracy']:.3f}\n")

    out = pd.DataFrame(rows).merge(bucket_index, on=["bucket", "bucket_start"], how="right")
    out = out.sort_values("bucket_start").reset_index(drop=True)

    if plot:
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            raise RuntimeError(
                "Plot requested but matplotlib is not available. Install it (e.g. `pip install matplotlib`) "
                "or rerun with plot=False."
            ) from e

        out_plot = out[out["n_total"] > 0].copy()
        if len(out_plot) == 0:
            print("\n[Plot] No half-year buckets with data to plot.")
            out_plot = out.copy()

        x = out_plot["bucket_start"]
        if getattr(x.dt, "tz", None) is not None:
            x = x.dt.tz_convert(None)

        xlim_start = pd.Timestamp(x.min()) if len(out_plot) else pd.Timestamp(year=start_year, month=1, day=1)
        xlim_end = (pd.Timestamp(x.max()) + pd.offsets.MonthEnd(6)) if len(out_plot) else pd.Timestamp(year=end_year, month=12, day=31)

        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 7), sharex=True)

        axes[0].plot(x, out_plot["recall"], linewidth=2)
        axes[0].set_ylabel("Recall")
        axes[0].set_ylim(0, 1)
        axes[0].grid(True, alpha=0.25)

        axes[1].plot(x, out_plot["fpr"], linewidth=2)
        axes[1].set_ylabel("FPR")
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, alpha=0.25)

        axes[2].plot(x, out_plot["balanced_accuracy"], linewidth=2)
        axes[2].set_ylabel("BalAcc")
        axes[2].set_ylim(0, 1)
        axes[2].grid(True, alpha=0.25)

        for ax in axes:
            ax.set_xlim(xlim_start, xlim_end)

        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(out_plot["bucket"].tolist(), rotation=45, ha="right")

        fig.suptitle(f"Gemma-3 half-year metrics ({start_year}–{end_year})")
        plt.tight_layout()
        plt.show()

    return out






"""
    MAIN RUN
"""

if __name__ == "__main__":

    """
        Run Gemma inference
    """
    gemma_inference()


    """
        Convert JSONL to enriched CSV
    """
    class_name = "ai" # "ai" or "miscaptioned"
    jsonl_path = f"gemma_test/gemma3_{class_name}.jsonl"
    dataset_csv = "path/to/dataset/images/set"
    out_csv = f"gemma_test/gemma3_{class_name}.csv"


    gemma_jsonl_to_enriched_csv(jsonl_path, dataset_csv, out_csv)


    """
        Evaluate Gemma-3 (AI vs miscaptioned)
    """

    path_ai = f"gemma_test/gemma3_ai.csv"
    path_miscaptioned = f"gemma_test/gemma3_miscaptioned.csv"


    evaluate_gemma_ai_miscaptioned(path_ai, path_miscaptioned)



    evaluate_gemma_ai_miscaptioned_overtime(
        path_ai,
        path_miscaptioned,
        plot=False,
    )

