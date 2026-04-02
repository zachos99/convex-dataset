import os
import time
import json
from pathlib import Path
from typing import Set
import re

import dotenv
from openai import OpenAI, BadRequestError
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

CHATGPT_API_KEY = os.getenv("CHATGPT_API_KEY")


# ----------------------------
# CONFIG
# ----------------------------
MODEL = "gpt-5-mini"

CLASS = "ai"

LOCAL_FOLDER = f"gpt_test/tweet_images_{CLASS}"
BASE_URL = f"https://storage.googleapis.com/{CLASS}_images/"

OUT_JSONL = f"gpt_test/openai_{MODEL}_{CLASS}_results.jsonl"

MAX_NEW_IMAGES = None  # None = run everything, else process only this many new images


MAX_OUTPUT_TOKENS = 300
SLEEP_BETWEEN_CALLS_S = 0.15
MAX_RETRIES = 4
RETRY_SLEEP_BASE = 1.5

# Pricing (adjust if model pricing changes)
INPUT_PRICE_PER_M = 0.25    # For GPT-5-mini
OUTPUT_PRICE_PER_M = 2.00  # For GPT-5-mini

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
def load_completed_and_cost(jsonl_path):
    done: Set[str] = set()
    cumulative_cost = 0.0

    p = Path(jsonl_path)
    if not p.exists():
        return done, cumulative_cost

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except:
                continue

            if row.get("status") != "COMPLETED":
                continue

            rel = row.get("relative_path")
            if rel:
                done.add(rel)

            prompt = int(row.get("prompt_tokens", 0) or 0)
            cached = int(row.get("cached_tokens", 0) or 0)
            billed = int(row.get("billed_prompt_tokens", max(prompt - cached, 0)))
            out = int(row.get("completion_tokens", 0) or 0)

            cost = (
                (billed / 1_000_000) * INPUT_PRICE_PER_M +
                (out / 1_000_000) * OUTPUT_PRICE_PER_M
            )

            cumulative_cost += cost

    return done, cumulative_cost


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
        print(f"Skipped {empty} empty (0-byte) files.")

    return sorted(images)


def build_url(image_path):
    rel = image_path.relative_to(Path(LOCAL_FOLDER)).as_posix()
    return BASE_URL.rstrip("/") + "/" + rel


# ----------------------------
# CALL
# ----------------------------
def extract_text(resp):
    if getattr(resp, "output_text", None):
        return resp.output_text.strip()

    parts = []
    for item in getattr(resp, "output", []) or []:
        content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
        if not content:
            continue
        for c in content:
            if c.get("type") in ("output_text", "text"):
                parts.append(c.get("text", ""))

    return "\n".join(parts).strip()


def get_cached_tokens(usage):
    try:
        details = getattr(usage, "input_tokens_details", None)
        if details and hasattr(details, "cached_tokens"):
            return int(details.cached_tokens or 0)
    except:
        pass
    return 0


def call_openai_one(client, image_url):
    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.responses.create(
                model=MODEL,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                reasoning={"effort": "minimal"}, 
                text={"verbosity": "low"},
                input=[
                    {"role": "developer", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": USER_TEXT},
                            {"type": "input_image", "image_url": image_url},
                        ],
                    },
                ],
            )

            # print(resp.model_dump_json(indent=2)[:4000])

            label = extract_text(resp).strip().upper()
            usage = getattr(resp, "usage", None)

            prompt_tokens = getattr(usage, "input_tokens", 0) if usage else 0
            completion_tokens = getattr(usage, "output_tokens", 0) if usage else 0
            cached = get_cached_tokens(usage)
            billed_prompt = max(prompt_tokens - cached, 0)

            return label, prompt_tokens, cached, billed_prompt, completion_tokens

        except BadRequestError as e:
            raise
        except Exception as e:
            last_err = e
            time.sleep(RETRY_SLEEP_BASE * (2 ** (attempt - 1)))

    raise RuntimeError(f"Failed after retries: {last_err}")


# ----------------------------
# MAIN
# ----------------------------
def inference_openai():
    if not CHATGPT_API_KEY:
        raise RuntimeError("Missing CHATGPT_API_KEY environment variable.")
    client = OpenAI(api_key=CHATGPT_API_KEY)

    images = list_images(LOCAL_FOLDER)
    completed, cumulative_cost = load_completed_and_cost(OUT_JSONL)

    print(f"Total images: {len(images)}")
    print(f"Already completed: {len(completed)}")
    print(f"Cumulative previous cost: ${cumulative_cost:.6f}\n")

    out_f = open(OUT_JSONL, "a", encoding="utf-8")

    try:
        new_processed = 0

        for idx, path in enumerate(images, start=1):
            rel = str(path.relative_to(Path(LOCAL_FOLDER)))

            if rel in completed:
                continue

            if MAX_NEW_IMAGES is not None and new_processed >= MAX_NEW_IMAGES:
                print(f"\nReached MAX_NEW_IMAGES={MAX_NEW_IMAGES}. Stopping.")
                break

            url = build_url(path)

            try:
                label, prompt, cached, billed, out_tok = call_openai_one(client, url)

                cost = (
                    (billed / 1_000_000) * INPUT_PRICE_PER_M +
                    (out_tok / 1_000_000) * OUTPUT_PRICE_PER_M
                )

                cumulative_cost += cost

                row = {
                    "relative_path": rel,
                    "label": label,
                    "prompt_tokens": prompt,
                    "cached_tokens": cached,
                    "billed_prompt_tokens": billed,
                    "completion_tokens": out_tok,
                    "status": "COMPLETED",
                }

                out_f.write(json.dumps(row) + "\n")
                out_f.flush()

                new_processed += 1

                print(
                    f"[{idx}] {rel} -> {label} "
                    f"| in={prompt} "
                    f"| cached={cached} "
                    f"| billed={billed} "
                    f"| out={out_tok} "
                    f"| cost=${cost:.8f} "
                    f"| cumulative=${cumulative_cost:.6f}"
                )

            except Exception as e:
                row = {
                    "relative_path": rel,
                    "status": "ERROR",
                    "error": str(e),
                }
                out_f.write(json.dumps(row) + "\n")
                out_f.flush()

                print(f"[{idx}] {rel} -> ERROR")

            time.sleep(SLEEP_BETWEEN_CALLS_S)

    finally:
        out_f.close()

    print("\n=== FINAL TOTAL COST ===")
    print(f"${cumulative_cost:.6f}")


def openai_jsonl_to_enriched_csv(jsonl_path, dataset_csv, out_csv):


    TWEETID_RE = re.compile(r"photo_(\\d+)_")

    # -------------------------
    # A) Load JSONL (one json per line)
    # -------------------------
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # ignore partial/corrupt line (e.g., crash mid-write)
                continue

    if not rows:
        raise RuntimeError(f"No valid JSON rows found in: {jsonl_path}")

    df = pd.DataFrame(rows)

    # Expect at least: relative_path, label, status
    if "relative_path" not in df.columns:
        raise RuntimeError("JSONL rows must contain 'relative_path' field.")

    # -------------------------
    # B) Extract tweetId + image key
    # -------------------------
    df["image"] = df["relative_path"].astype(str)
    df["image_key"] = df["image"].map(os.path.basename)

    def extract_tweet_id_from_key(k):
        m = TWEETID_RE.search(str(k))
        return m.group(1) if m else None

    df["tweetId"] = df["image_key"].map(extract_tweet_id_from_key)

    # Keep the minimal inference fields
    df_inf = df[["tweetId", "label", "image", "image_key", "status"]].copy()

    # -------------------------
    # C) Enrich from dataset CSV (same columns as in rine_utils.jsonl_to_enriched_csv)
    # -------------------------
    # Read tweet_id as str to avoid float precision loss for large Twitter IDs
    df_ds = pd.read_csv(dataset_csv, low_memory=False, dtype={"tweet_id": str})

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

    # Prefer dataset tweet_id if present; otherwise fallback to parsed tweetId
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


def evaluate_openai_ai_miscaptioned(path_ai, path_miscaptioned, label_col = "label"):
    """
    Evaluate OpenAI GPT-5-mini (AI vs miscaptioned) using discrete labels only.

    - Ground truth: path_ai = AI images (y_true=1), path_miscaptioned = REAL/miscaptioned (y_true=0)
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
    print("\n[OpenAI] Label diagnostics:")
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
    print("OpenAI GPT-5-mini evaluation (AI vs miscaptioned)")
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



def evaluate_openai_ai_miscaptioned_overtime(
    path_ai,
    path_miscaptioned,
    date_col = "created_at_datetime",
    label_col = "label",
    start_year= 2023,
    end_year= 2026,
    plot = False,
):
    """
    Evaluate OpenAI GPT-5-mini (AI vs miscaptioned) over time in 6-month spans (H1/H2).

    Buckets:
    - YYYY-H1: Jan–Jun
    - YYYY-H2: Jul–Dec

    If `plot=True`, plots Recall / FPR / Balanced Accuracy in 3 subplots over these half-year buckets.
    """
    df_ai = pd.read_csv(path_ai)
    df_real = pd.read_csv(path_miscaptioned)

    df_ai["y_true"] = 1
    df_real["y_true"] = 0
    df = pd.concat([df_ai, df_real], ignore_index=True)

    # Use UTC-aware timestamps to avoid tz-aware vs tz-naive comparison errors
    dates = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    df["_date"] = dates
    n_total_dates = len(df)
    n_parsed_dates = int(dates.notna().sum())
    print("\n[Date parsing diagnostic]")
    print(f"Parsed dates: {n_parsed_dates}/{n_total_dates} ({(n_parsed_dates / n_total_dates) if n_total_dates else float('nan'):.3%})")

    # Half-year bucket key + timestamp anchor (for ordering/plotting)
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

    # Build full list of expected half-year buckets (even if empty)
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
    title = f"OpenAI GPT-5-mini Over Time (Half-year, {start_year}–{end_year})"
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
        # Convert to tz-naive for plotting, if needed
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

        axes[-1].set_xticks(out_plot["bucket_start"])
        axes[-1].set_xticklabels(out_plot["bucket"].tolist(), rotation=45, ha="right")

        fig.suptitle(f"OpenAI GPT-5-mini half-year metrics ({start_year}–{end_year})")
        plt.tight_layout()
        plt.show()

    return out




if __name__ == "__main__":

    """
        Inference OpenAI GPT-5-mini 
    """
    inference_openai()

    """
        Convert JSONL to enriched CSV
    """
    class_name = "miscaptioned"
    jsonl_path = f"gpt_test/openai_{MODEL}_{class_name}_results.jsonl"
    dataset_csv = "path/to/dataset/images/set"
    out_csv = f"gpt_test/openai_{MODEL}_{class_name}_results.csv"

    openai_jsonl_to_enriched_csv(
        jsonl_path,
        dataset_csv,
        out_csv,
    )



    """
        Evaluate OpenAI GPT-5-mini (AI vs miscaptioned)
    """
    path_ai = f"gpt_test/openai_{MODEL}_ai_results.csv"
    path_miscaptioned = f"gpt_test/openai_{MODEL}_miscaptioned_results.csv"

    evaluate_openai_ai_miscaptioned(path_ai, path_miscaptioned)

    evaluate_openai_ai_miscaptioned_overtime(
        path_ai,
        path_miscaptioned,
        plot=False,
    )















