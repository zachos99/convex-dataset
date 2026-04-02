import os
from datetime import datetime

import numpy as np
import pandas as pd

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


# Core metric computation
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


# IO helpers
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


def enrich_spai(spai_csv_path, dataset_csv, out_enriched):
    # A) Load SPAI CSV from explicit path 
    if not os.path.exists(spai_csv_path):
        raise RuntimeError(f"SPAI CSV not found: {spai_csv_path}")

    df_spai = pd.read_csv(spai_csv_path)

    # B) Fix + sort SPAI CSV
    s = df_spai["spai"].astype(float).clip(0.0, 1.0)
    s = np.where(s < EPS, 0.0, s)
    # Round to n decimals
    # decimals = 2
    # df_spai["spai"] = np.round(s, decimals)

    df_spai = df_spai.sort_values(by="image", kind="stable").reset_index(drop=True)

    # overwrite SPAI CSV
    df_spai.to_csv(spai_csv_path, index=False)
    print(f"Fixed + sorted SPAI CSV: {spai_csv_path}")

    # C) Enrich from dataset
    df_ds = pd.read_csv(dataset_csv, low_memory=False)

    df_spai["image_key"] = df_spai["image"].astype(str).str.strip().map(os.path.basename)

    tmp = df_ds[[
        "media",
        "tweet_id",
        "tweetUrl",
        "noteText",
        "misinfo_type_final",
        "topic",
        "created_at_datetime",
    ]].copy()

    # Ensure tweet_id is written without scientific notation in the output CSV
    if "tweet_id" in tmp.columns:
        def _normalize_tweet_id(x):
            if pd.isna(x):
                return ""
            # Handle numeric types (including floats that may have come from CSV inference)
            if isinstance(x, (int, np.integer)):
                return str(x)
            if isinstance(x, (float, np.floating)):
                return str(int(x))
            # Fallback: treat as string
            return str(x)

        tmp["tweet_id"] = tmp["tweet_id"].map(_normalize_tweet_id)

    tmp["media"] = tmp["media"].fillna("").astype(str)
    tmp["media_item"] = tmp["media"].str.split(",")
    tmp = tmp.explode("media_item")
    tmp["media_item"] = tmp["media_item"].astype(str).str.strip()
    tmp["image_key"] = tmp["media_item"].map(os.path.basename)

    tmp = tmp[tmp["image_key"] != ""].drop_duplicates("image_key", keep="first")

    merged = df_spai.merge(
        tmp[[
            "image_key",
            "tweet_id",
            "tweetUrl",
            "noteText",
            "misinfo_type_final",
            "topic",
            "created_at_datetime",
        ]],
        on="image_key",
        how="left",
        validate="many_to_one",
    )

    out = merged[[
        "tweet_id",
        "spai",
        "misinfo_type_final",
        "topic",
        "created_at_datetime",
        "noteText",
        "image",
        "tweetUrl",
    ]].copy()
    out = out.rename(columns={"created_at_datetime": "tweet_date"})

    os.makedirs(os.path.dirname(out_enriched), exist_ok=True)
    out.to_csv(out_enriched, index=False)

    print(f"Wrote {out_enriched}")
    print(f"Matched {out['tweet_id'].notna().sum()}/{len(out)} images")

def _parse_tweet_date(ser):
    """Parse tweet_date column; supports ISO-like and Twitter 'Sun Nov 29 10:02:09 +0000 2020' formats."""
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





# -----------------------------
# Model-specific evaluators
# -----------------------------
def evaluate_spai_ai_miscaptioned(path_ai, path_miscaptioned, threshold=0.5):
    """
    SPAI:
      - scores in column: 'spai'
      - threshold default: 0.5
    """
    df_ai = _read_df(path_ai)
    df_real = _read_df(path_miscaptioned)
    df = _prepare_binary_df(df_ai, df_real, score_col="spai")

    y_true = df["y_true"].to_numpy(dtype=int)
    scores = df["spai"].to_numpy(dtype=float)
    y_pred = (scores >= threshold).astype(int)

    metrics = compute_metrics(y_true, y_pred, scores)
    summary = pd.DataFrame([metrics], index=[f"SPAI @ {threshold:.2f}"])

    confusion_mats = {
        f"SPAI @ {threshold:.2f}": confusion_matrix(y_true, y_pred, labels=[0, 1])
    }
    _pretty_print_results(
        title=f"SPAI evaluation (AI vs miscaptioned) | n={len(df)} (AI={y_true.sum()}, Real={(y_true==0).sum()})",
        summary_df=summary,
        confusion_mats=confusion_mats
    )
    return summary


def evaluate_spai_ai_miscaptioned_overtime(
    path_ai,
    path_miscaptioned,
    date_col: str = "tweet_date",
    score_col: str = "spai",
    threshold: float = 0.5,
    start_year: int = 2023,
    end_year: int = 2026,
    plot: bool = False,
):
    """
    SPAI evaluation (AI vs miscaptioned) over time in 6-month spans (H1/H2).

    Rule: score >= threshold -> predict AI.
    """
    df_ai = _read_df(path_ai)
    df_real = _read_df(path_miscaptioned)

    df_ai["y_true"] = 1
    df_real["y_true"] = 0
    df = pd.concat([df_ai, df_real], ignore_index=True)

    # Parse dates (normalize to UTC-aware to avoid tz-aware vs tz-naive comparisons)
    dates = _parse_tweet_date(df[date_col])
    dates = pd.to_datetime(dates, errors="coerce", utc=True)
    df["_date"] = dates

    n_total_dates = len(df)
    n_parsed_dates = int(dates.notna().sum())
    print("\n[Date parsing diagnostic]")
    print(f"Parsed dates: {n_parsed_dates}/{n_total_dates} ({(n_parsed_dates / n_total_dates) if n_total_dates else float('nan'):.3%})")

    # Half-year bucket
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

    # Clean score
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

    # Drop invalid rows consistently
    df = df.dropna(subset=["_bucket_start", score_col, "y_true"]).copy()
    df = df[(df["_year"] >= start_year) & (df["_year"] <= end_year)].copy()

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
    title = f"SPAI Over Time (Half-year, {start_year}–{end_year})"
    print(title)
    print("==============================")
    print(f"Threshold: {threshold}\n")

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
        y_pred = (scores >= float(threshold)).astype(int)
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
        labels = out.loc[out["n_total"] > 0, "bucket"].tolist()

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
            axes[-1].set_xticklabels(labels, rotation=45, ha="right")

            fig.suptitle(f"SPAI half-year metrics ({start_year}–{end_year})")
            plt.tight_layout()
            plt.show()

    return out



