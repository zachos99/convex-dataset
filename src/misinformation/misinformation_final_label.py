import pandas as pd
import os

"""
    Reads:
        misinfo_type_keys
        misinfo_type_llm
        misinfo_type_llm_rerun (if present; otherwise treated as missing)

    
    Fusion rules:
      1) agreement (keys==llm) and not 'other' -> keep it
      2) if keys != llm (both valid) and rerun is specific -> use rerun
      3) else if keys is specific -> keep keys
      4) else if llm is specific -> use llm
      5) else if rerun is specific -> use rerun (rare but safe)
      6) else -> other
    

    Writes a new CSV with two extra columns:
        misinfo_type_final
        misinfo_type_final_source_flag
"""




VALID_LABELS = {"ai_generated", "edited", "miscaptioned", "other"}


def _norm_label(x):
    """Normalize labels to lowercase stripped strings, or None if missing/invalid."""
    if isinstance(x, str):
        x = x.strip().lower()
        return x if x in VALID_LABELS else None
    return None



def compute_final_label(row):
    
    k  = _norm_label(row.get("misinfo_type_keys"))
    g1 = _norm_label(row.get("misinfo_type_llm"))
    g2 = _norm_label(row.get("misinfo_type_llm_rerun"))

    # 1) Agreement (non-other) wins
    if k is not None and g1 is not None and k == g1 and k != "other":
        return k, "agree_keys_llm"

    # 2) Disagreement -> rerun resolves if specific
    disagreement = (k is not None and g1 is not None and k != g1)
    if disagreement and g2 is not None and g2 != "other":
        return g2, "rerun_llm_decider"

    # 3) Keys specific -> keep keys
    if k in {"ai_generated", "edited", "miscaptioned"}:
        return k, "keys_only_or_keys_over_llm"

    # 4) Else use first LLM if specific
    if g1 in {"ai_generated", "edited", "miscaptioned"}:
        return g1, "llm_only"

    # 5) Else use rerun if specific (rare but safe)
    if g2 in {"ai_generated", "edited", "miscaptioned"}:
        return g2, "rerun_only"

    # 6) Fallback
    return "other", "fallback_other"



def build_final_misinfo_labels(input_path, output_path=None):
    """
    Read the CSV, compute misinfo_type_final and misinfo_type_final_source_flag
    for every row based on the specified rules, and save to a new CSV.
    """
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_with_final_labels{ext}"

    df = pd.read_csv(input_path)

    # Make sure required columns exist
    required_cols = ["misinfo_type_keys", "misinfo_type_llm"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")

    # misinfo_type_llm_rerun is optional; handled gracefully in compute_final_label

    results = df.apply(
        lambda row: compute_final_label(row),
        axis=1,
        result_type="expand"
    )
    results.columns = ["misinfo_type_final", "misinfo_type_final_source_flag"]

    df = pd.concat([df, results], axis=1)
    df.to_csv(output_path, index=False)
    print(f"Saved final labels to: {output_path}")

     # ---- stats: final label ----
    print("\n=== misinfo_type_final distribution ===")
    vc = df["misinfo_type_final"].replace({"": "MISSING"}).fillna("MISSING") \
        .value_counts(dropna=False).to_frame("count")
    vc["pct"] = (vc["count"] / len(df) * 100).round(2)
    print(vc)

    # ---- stats: source flag ----
    print("\n=== misinfo_type_final_source_flag distribution ===")
    vc2 = df["misinfo_type_final_source_flag"].replace({"": "MISSING"}).fillna("MISSING") \
        .value_counts(dropna=False).to_frame("count")
    vc2["pct"] = (vc2["count"] / len(df) * 100).round(2)
    print(vc2)

    return df











if __name__ == "__main__":

    PATH="path/to/dataset/after_misinfo_run"

    FINAL_PATH="path/to/dataset/after_misinfo_run_final.csv"


    build_final_misinfo_labels(
        input_path=PATH, 
        output_path=FINAL_PATH
    )






