import time, json, requests, re
import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime


from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score
)

# imgbb upload endpoint (used by `upload_one_to_imgbb`)
IMGBB_ENDPOINT = "https://api.imgbb.com/1/upload"


def compute_metrics(y_true, y_pred, y_scores):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    tpr = recall_score(y_true, y_pred, pos_label=1)  # same as recall for AI class
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    # For AUCs: require both classes present
    roc_auc = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) == 2 else np.nan
    pr_auc = average_precision_score(y_true, y_scores) if len(np.unique(y_true)) == 2 else np.nan

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
        "PR-AUC": float(pr_auc),
    }

# -----------------------------
# IO helpers
# -----------------------------
def _read_df(path):
    if not isinstance(path, str) or not path:
        raise ValueError("Path must be a non-empty string.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv"]:
        return pd.read_csv(path)
    if ext in [".tsv"]:
        return pd.read_csv(path, sep="\t")
    if ext in [".parquet"]:
        return pd.read_parquet(path)
    if ext in [".feather"]:
        return pd.read_feather(path)

    # Try CSV fallback
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise ValueError(
            f"Unsupported file extension '{ext}'. Supported: .csv, .tsv, .parquet, .feather"
        ) from e

def _prepare_binary_df(df_ai, df_real, score_col):
    if score_col not in df_ai.columns:
        raise KeyError(f"Missing column '{score_col}' in AI df. Columns: {list(df_ai.columns)[:20]}...")
    if score_col not in df_real.columns:
        raise KeyError(f"Missing column '{score_col}' in real df. Columns: {list(df_real.columns)[:20]}...")

    ai = df_ai[[score_col]].copy()
    ai["y_true"] = 1

    real = df_real[[score_col]].copy()
    real["y_true"] = 0

    df = pd.concat([ai, real], ignore_index=True)
    df = df.dropna(subset=[score_col, "y_true"]).reset_index(drop=True)

    return df

def _pretty_print_results(title, summary_df, confusion_mats):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

    # Format for readability
    display_cols = [
        "TPR (Recall)", "FPR", "Precision", "F1", "Accuracy", "Balanced Accuracy", "ROC-AUC", "PR-AUC",
        "TP", "FP", "TN", "FN",
    ]
    df_show = summary_df.copy()
    for c in df_show.columns:
        if c in ["TP", "FP", "TN", "FN"]:
            df_show[c] = df_show[c].astype(int)
        else:
            df_show[c] = df_show[c].astype(float)

    with pd.option_context("display.max_columns", None, "display.width", 140):
        print(df_show[display_cols])

    print("\nConfusion matrices (rows=true [Real, AI], cols=pred [Real, AI])")
    for k, cm in confusion_mats.items():
        print(f"\n{k}")
        print(cm)




EPS = 1e-6


def upload_one_to_imgbb(path, api_key, expiration=3600, timeout=60):
    api_key = str(api_key).strip()
    if not api_key:
        raise ValueError("IMGBB_API_KEY is empty/blank.")

    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    if os.path.isdir(path):
        raise ValueError(f"Path is a directory, not a file: {path}")
    if os.path.getsize(path) == 0:
        print(f"File is empty (0 bytes): {path}")
        return None

    with open(path, "rb") as f:
        resp = requests.post(
            IMGBB_ENDPOINT,
            data={"key": api_key, "expiration": int(expiration)},
            files={"image": f},
            timeout=timeout,
        )

    if not resp.ok:
        try:
            detail = resp.json()
        except Exception:
            detail = {"raw_text": resp.text[:500]}
        raise RuntimeError(f"imgbb upload failed HTTP {resp.status_code} for {path}. Detail: {detail}")

    return resp.json()["data"]["url"]


BASE_URL = "https://apis.mever.gr/deepfake/v4"

HEADERS = {"accept": "application/json"}

def submit_image_job_url(service, url, search_similar=False, max_retries=3):
    """
    Create an image job by URL.
    Sends query params + empty multipart part (like the site's curl).
    Returns the job dict (with 'id').
    Retries on request timeout (read/connect).
    """
    endpoint = f"{BASE_URL}/images/jobs"
    params = {
        "url": url,
        "services": service,
        "search_similar": str(search_similar).lower()
    }
    files = {"file": (None, "")}  
    for attempt in range(max_retries):
        try:
            r = requests.post(endpoint, params=params, headers=HEADERS, files=files, timeout=180)
            r.raise_for_status()
            return r.json()  # e.g. {"id": "...job id..."}
        except requests.exceptions.Timeout:
            if attempt == max_retries - 1:
                raise
            time.sleep(5)

def get_image_report(job_id):
    """
    Fetch the image report using the PATH style:
    GET /images/reports/{id}
    """
    endpoint = f"{BASE_URL}/images/reports/{job_id}"
    r = requests.get(endpoint, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()

def wait_for_completion(job_id, interval=1.5):
    """
    Poll until status == COMPLETED, then return the full report JSON.
    """
    while True:
        rep = get_image_report(job_id)
        status = (rep.get("status") or "").upper()
        print("Status:", status)
        if status == "COMPLETED":
            return rep
        time.sleep(interval)


def load_done_paths(jsonl_path):
    done = set()
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
                lp = row.get("local_path")
                status = (row.get("status") or "").upper()
                if lp and status == "COMPLETED":
                    done.add(lp)
            except json.JSONDecodeError:
                # ignore partial/corrupt last line
                continue
    return done


def run_rine_on_folder(
    folder,
    service,
    imgbb_api_key=None,
    out_jsonl=None,
    extensions=(".jpg", ".jpeg", ".png", ".webp"),
    expiration=3600,
    upload_wait=1,
    rine_wait_interval=1.5,
):
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    if not folder.is_dir():
        raise ValueError(f"Not a folder: {folder}")

    if imgbb_api_key is None:
        raise ValueError("imgbb_api_key is required (pass it from rine_pipeline.py or similar).")
    key = str(imgbb_api_key).strip()
    if not key:
        raise ValueError("imgbb_api_key is empty/blank.")

    paths = sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in extensions])
    if not paths:
        raise ValueError(f"No images found in {folder} with extensions {extensions}")

    done_paths = load_done_paths(out_jsonl) if out_jsonl else set()
    if done_paths:
        print(f"Resuming: skipping {len(done_paths)} already COMPLETED images from {out_jsonl}")

    results = []
    errors = []
    out_f = open(out_jsonl, "a", encoding="utf-8") if out_jsonl else None

    try:
        i_done = 0
        i_total = len(paths)

        for p in paths:
            if str(p) in done_paths:
                i_done += 1
                continue

            url = upload_one_to_imgbb(p, key, expiration=expiration)
            if url is None:
                print(f"SKIP (empty file): {p}")
                row = {"local_path": str(p), "label": None, "prediction": None, "status": "EMPTY_FILE"}
                if out_f:
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out_f.flush()
                continue
            if upload_wait:
                time.sleep(upload_wait)

            try:
                job = submit_image_job_url(service, url, search_similar=False)
                job_id = job["id"]
                final = wait_for_completion(job_id, interval=rine_wait_interval)

                rep = final.get(f"{service}_report", {}) or {}

                row = {
                    "local_path": str(p),
                    "label": rep.get("label"),
                    "prediction": rep.get("prediction"),
                    "status": final.get("status"),
                }

                results.append(row)

                if out_f:
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out_f.flush()

                print(f"[{i_done + len(results)}/{i_total}] {p.name} -> {row['prediction']}")
            except Exception as e:
                row = {"local_path": str(p), "label": None, "prediction": None, "status": "ERROR", "error": str(e)}
                results.append(row)
                errors.append((str(p), str(e)))
                if out_f:
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out_f.flush()
                print(f"[{i_done + len(results)}/{i_total}] {p.name} -> ERROR (skipped)")

        if errors:
            print(f"\n--- Errors ({len(errors)}) ---")
            for path, msg in errors:
                print(f"  {path}: {msg}")

    finally:
        if out_f:
            out_f.close()

    return results




def jsonl_to_enriched_csv(jsonl_path, dataset_csv, out_csv):

    TWEETID_RE = re.compile(r"photo_(\d+)_")
    
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

    # Expect at least: local_path, label, prediction, status
    if "local_path" not in df.columns:
        raise RuntimeError("JSONL rows must contain 'local_path' field.")

    # -------------------------
    # B) Extract tweetId + image key
    # -------------------------
    df["image"] = df["local_path"].astype(str)
    df["image_key"] = df["image"].map(os.path.basename)

    def extract_tweet_id_from_key(k):
        m = TWEETID_RE.search(str(k))
        return m.group(1) if m else None

    df["tweetId"] = df["image_key"].map(extract_tweet_id_from_key)

    # Keep the minimal inference fields
    # (status is useful; keep it unless you truly don't want it)
    df_inf = df[["tweetId", "label", "prediction", "image", "image_key", "status"]].copy()

    # -------------------------
    # C) Enrich from dataset CSV (mimic fix_spai)
    # -------------------------
    # Read tweet_id as str to avoid float precision loss for large Twitter IDs
    df_ds = pd.read_csv(dataset_csv, low_memory=False, dtype={"tweet_id": str})

    tmp = df_ds[[
        "media",
        "tweet_id",
        "created_at_datetime",
        "tweetUrl",
        "noteText",
        "misinfo_type_final",
        "topic",
    ]].copy()

    tmp["media"] = tmp["media"].fillna("").astype(str)
    tmp["media_item"] = tmp["media"].str.split(",")
    tmp = tmp.explode("media_item")
    tmp["media_item"] = tmp["media_item"].astype(str).str.strip()

    tmp["image_key"] = tmp["media_item"].map(os.path.basename)
    tmp = tmp[tmp["image_key"] != ""].drop_duplicates("image_key", keep="first")

    merged = df_inf.merge(
        tmp[[
            "image_key",
            "tweet_id",
            "created_at_datetime",
            "tweetUrl",
            "noteText",
            "misinfo_type_final",
            "topic",
        ]],
        on="image_key",
        how="left",
        validate="many_to_one",
    )

    # Prefer dataset tweet_id if present; otherwise fallback to parsed tweetId (no Int64 to avoid precision loss)
    merged["tweetId"] = merged["tweet_id"].where(
        merged["tweet_id"].notna() & (merged["tweet_id"].astype(str).str.strip() != ""),
        None,
    )
    merged["tweetId"] = merged["tweetId"].fillna(df_inf["tweetId"])

    out = merged[[
        "tweetId",
        "label",
        "prediction",
        "misinfo_type_final",
        "topic",
        "created_at_datetime",
        "noteText",
        "image",
        "tweetUrl",
    ]].copy()

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out.to_csv(out_csv, index=False)

    print(f"Wrote {out_csv}")
    print(f"Matched {out['misinfo_type_final'].notna().sum()}/{len(out)} images via image_key")







# Label sets for RINE "detected as AI" (used in AI-only detectability evaluation)
RINE_LABEL_LEVELS = {
    "MODERATE": {"MODERATE_EVIDENCE", "STRONG_EVIDENCE", "VERY_STRONG_EVIDENCE"},
    "STRONG": {"STRONG_EVIDENCE", "VERY_STRONG_EVIDENCE"},
}



def _parse_tweet_date(ser):
    """Parse tweet date column; supports ISO-like and Twitter 'Sun Nov 29 10:02:09 +0000 2020' formats."""
    out = pd.to_datetime(ser, errors="coerce")
    still_nat = out.isna() & ser.notna() & (ser.astype(str).str.strip() != "")

    def try_twitter(s):
        s = str(s).strip()
        if not s:
            return pd.NaT
        try:
            return datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y")
        except Exception:
            return pd.NaT

    filled = ser.loc[still_nat].map(try_twitter)
    out = out.copy()
    out.loc[still_nat] = filled.values
    return pd.to_datetime(out, errors="coerce")


"""
    Evaluate on AI vs Miscaptioned
"""
def evaluate_rine_ai_miscaptioned(
    path_ai,
    path_miscaptioned,
    score_col="prediction",
    label_col="label",
    score_threshold=0.5,
    label_levels=("MODERATE", "STRONG"),
):


    # --- Read ---
    df_ai = _read_df(path_ai)
    df_real = _read_df(path_miscaptioned)

    # --- Prepare + merge (creates y_true, keeps only score_col) ---
    df = _prepare_binary_df(df_ai, df_real, score_col=score_col)

    # --- Add label col back in (we need it for label-based thresholds) ---
    # _prepare_binary_df currently keeps only [score_col] + y_true; we must re-merge labels carefully.
    # We do this by re-creating merged df with both columns to keep behavior predictable.
    def _select_cols(d, y):
        out = d[[score_col, label_col]].copy()
        out["y_true"] = y
        return out

    df2 = pd.concat(
        [_select_cols(df_ai, 1), _select_cols(df_real, 0)],
        ignore_index=True
    )

    n_total = len(df2)

    # --- Coerce score to numeric and drop invalid rows CONSISTENTLY ---
    df2[score_col] = pd.to_numeric(df2[score_col], errors="coerce")
    df2[label_col] = df2[label_col].astype(str).str.strip()

    # Keep only rows with valid numeric score AND non-empty label
    df_valid = df2.dropna(subset=[score_col]).copy()
    df_valid = df_valid[df_valid[label_col].notna() & (df_valid[label_col] != "")].copy()

    n_dropped = n_total - len(df_valid)

    y_true = df_valid["y_true"].to_numpy(dtype=int)
    scores = df_valid[score_col].to_numpy(dtype=float)
    labels = df_valid[label_col].to_numpy(dtype=str)

    rows = []
    confusion_mats = {}

    # -----------------------
    # 1) Score-threshold eval
    # -----------------------
    y_pred_score = (scores >= float(score_threshold)).astype(int)
    metrics_score = compute_metrics(y_true, y_pred_score, scores)
    name_score = f"RINE score ≥ {float(score_threshold):.2f}"
    rows.append((name_score, metrics_score))
    confusion_mats[name_score] = confusion_matrix(y_true, y_pred_score, labels=[0, 1])

    # -----------------------
    # 2) Label-threshold evals
    # -----------------------
    for level in label_levels:
        lvl = str(level).upper().strip()
        if "RINE_LABEL_LEVELS" not in globals():
            raise NameError(
                "RINE_LABEL_LEVELS is not defined. Define it globally, e.g.\n"
                "RINE_LABEL_LEVELS = {\n"
                "  'MODERATE': {'MODERATE_EVIDENCE','STRONG_EVIDENCE','VERY_STRONG_EVIDENCE'},\n"
                "  'STRONG': {'STRONG_EVIDENCE','VERY_STRONG_EVIDENCE'}\n"
                "}"
            )
        if lvl not in RINE_LABEL_LEVELS:
            raise KeyError(f"Unknown label level '{lvl}'. Available: {list(RINE_LABEL_LEVELS.keys())}")

        positive = RINE_LABEL_LEVELS[lvl]
        y_pred_label = np.isin(labels, list(positive)).astype(int)

        # For AUC metrics, we still use continuous `scores` (same as score-based)
        metrics_label = compute_metrics(y_true, y_pred_label, scores)

        name_label = f"RINE label ≥ {lvl}"
        rows.append((name_label, metrics_label))
        confusion_mats[name_label] = confusion_matrix(y_true, y_pred_label, labels=[0, 1])

    summary = pd.DataFrame({name: m for name, m in rows}).T

    _pretty_print_results(
        title=(
            f"RINE evaluation (AI vs miscaptioned)\n"
            f"Total rows (before drop): {n_total} | Dropped invalid (score/label): {n_dropped}\n"
            f"Used rows: {len(df_valid)} (AI={y_true.sum()}, Real={(y_true==0).sum()})"
        ),
        summary_df=summary,
        confusion_mats=confusion_mats
    )

    return summary



def evaluate_rine_ai_miscaptioned_overtime(
    path_ai,
    path_miscaptioned,
    date_col: str = "created_at_datetime",
    score_col: str = "prediction",
    label_col: str = "label",
    label_level: str = "MODERATE",
    start_year: int = 2023,
    end_year: int = 2026,
    plot: bool = False,
):

    df_ai = _read_df(path_ai)
    df_real = _read_df(path_miscaptioned)

    df_ai["y_true"] = 1
    df_real["y_true"] = 0
    df = pd.concat([df_ai, df_real], ignore_index=True)

    dates = _parse_tweet_date(df[date_col])
    dates = pd.to_datetime(dates, errors="coerce", utc=True)
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

    # Clean score + label
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df[label_col] = df[label_col].astype(str).str.strip()

    # Drop invalid rows consistently
    df = df.dropna(subset=["_bucket_start", score_col, "y_true"]).copy()
    df = df[df[label_col].notna() & (df[label_col] != "")].copy()
    df = df[(df["_year"] >= start_year) & (df["_year"] <= end_year)].copy()

    lvl = str(label_level).upper().strip()
    if lvl not in RINE_LABEL_LEVELS:
        raise ValueError(f"Unknown label level: {label_level}. Available: {list(RINE_LABEL_LEVELS.keys())}")
    positive_labels = RINE_LABEL_LEVELS[lvl]

    # Expected buckets
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
    title = f"RINE Over Time (Half-year, label ≥ {lvl}, {start_year}–{end_year})"
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
        scores = df_b[score_col].to_numpy(dtype=float)
        labels = df_b[label_col].to_numpy(dtype=str)
        y_pred = np.isin(labels, list(positive_labels)).astype(int)

        metrics = compute_metrics(y_true, y_pred, scores)

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
        x = out.loc[out["n_total"] > 0, "bucket_start"]
        yrec = out.loc[out["n_total"] > 0, "recall"]
        yfpr = out.loc[out["n_total"] > 0, "fpr"]
        yba = out.loc[out["n_total"] > 0, "balanced_accuracy"]
        labels_plot = out.loc[out["n_total"] > 0, "bucket"].tolist()

        if len(x) == 0:
            print("\n[Plot] No half-year buckets with data to plot.")
        else:
            x = pd.to_datetime(x, utc=True).dt.tz_convert(None)
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 7), sharex=True)

            axes[0].plot(x, yrec, linewidth=2)
            axes[0].set_ylabel("Recall")
            axes[0].set_ylim(0, 1)
            axes[0].grid(True, alpha=0.25)

            axes[1].plot(x, yfpr, linewidth=2)
            axes[1].set_ylabel("FPR")
            axes[1].set_ylim(0, 1)
            axes[1].grid(True, alpha=0.25)

            axes[2].plot(x, yba, linewidth=2)
            axes[2].set_ylabel("BalAcc")
            axes[2].set_ylim(0, 1)
            axes[2].grid(True, alpha=0.25)

            axes[-1].set_xticks(x)
            axes[-1].set_xticklabels(labels_plot, rotation=45, ha="right")

            fig.suptitle(f"RINE half-year metrics (label ≥ {lvl}, {start_year}–{end_year})")
            plt.tight_layout()
            plt.show()

    return out










        