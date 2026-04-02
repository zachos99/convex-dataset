from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib.ticker import PercentFormatter

from rine_utils import (
    evaluate_rine_ai_miscaptioned,
    evaluate_rine_ai_miscaptioned_overtime,
)

from spai_utils import (
    evaluate_spai_ai_miscaptioned,
    evaluate_spai_ai_miscaptioned_overtime,
)
from bfree_utils import (
    evaluate_bfree_ai_miscaptioned,
    evaluate_bfree_ai_miscaptioned_overtime,
)
from grok_xai_inference import (
    evaluate_grok_xai_ai_miscaptioned,
    evaluate_grok_xai_ai_miscaptioned_overtime,
)
from openai_inference import (
    evaluate_openai_ai_miscaptioned,
    evaluate_openai_ai_miscaptioned_overtime,
)
from gemma_inference import (
    evaluate_gemma_ai_miscaptioned,
    evaluate_gemma_ai_miscaptioned_overtime,
)


def _df_to_metric_dict(df):
    """
    Convert a metrics DataFrame (as returned by SPAI / RINE / BFREE evaluators)
    into a nested dict: {row_name -> {metric_name -> value}}.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for idx, row in df.iterrows():
        out[str(idx)] = row.to_dict()
    return out


def compare_all_models(
    spai_ai_csv,
    spai_misc_csv,
    rine_ai_csv,
    rine_misc_csv,
    bfree_ai_csv,
    bfree_misc_csv,
    grok_ai_csv,
    grok_misc_csv,
    openai_ai_csv,
    openai_misc_csv,
    gemma_ai_csv,
    gemma_misc_csv,
):
    """
    Run all five evaluators on AI vs miscaptioned and return a unified metrics dict.

    Parameters are paths to the enriched CSVs for each model:
      - SPAI / RINE / BFREE: CSVs with score columns as expected by their utils
      - Grok / OpenAI: enriched CSVs produced from their JSONLs

    Returns:
      {
        "spai": { "<row_name>": {metric -> value}, ... },
        "rine": { "<row_name>": {metric -> value}, ... },
        "bfree": { "<row_name>": {metric -> value}, ... },
        "gemma": { "<row_name>": {metric -> value}, ... },
        "grok_xai": {metric -> value},          # single dict
        "openai": {metric -> value},            # single dict
      }
    """

    results: Dict[str, Any] = {}

    # SPAI
    spai_df = evaluate_spai_ai_miscaptioned(
        path_ai=spai_ai_csv,
        path_miscaptioned=spai_misc_csv,
    )
    results["spai"] = _df_to_metric_dict(spai_df)

    # RINE
    rine_df = evaluate_rine_ai_miscaptioned(
        path_ai=rine_ai_csv,
        path_miscaptioned=rine_misc_csv,
    )
    results["rine"] = _df_to_metric_dict(rine_df)

    # BFREE
    bfree_df = evaluate_bfree_ai_miscaptioned(
        path_ai=bfree_ai_csv,
        path_miscaptioned=bfree_misc_csv,
    )
    results["bfree"] = _df_to_metric_dict(bfree_df)

    # Grok XAI
    grok_metrics = evaluate_grok_xai_ai_miscaptioned(
        path_ai=grok_ai_csv,
        path_miscaptioned=grok_misc_csv,
    )
    results["grok_xai"] = grok_metrics

    # OpenAI GPT-5-mini
    openai_metrics = evaluate_openai_ai_miscaptioned(
        path_ai=openai_ai_csv,
        path_miscaptioned=openai_misc_csv,
    )
    results["openai"] = openai_metrics

    # Gemma-3
    gemma_metrics = evaluate_gemma_ai_miscaptioned(
        path_ai=gemma_ai_csv,
        path_miscaptioned=gemma_misc_csv,
    )
    results["gemma"] = gemma_metrics

    # Pretty-print a quick comparison table for common metrics
    common_metrics = [
        "TPR (Recall)",
        "FPR",
        "Precision",
        "F1",
        "Accuracy",
        "Balanced Accuracy",
        "ROC-AUC",
    ]

    rows = []
    index = []

    # For SPAI / RINE / BFREE, take the first row as the primary operating point
    for name in ["spai", "rine", "bfree"]:
        model_dict = results[name]
        if not model_dict:
            continue
        first_key = sorted(model_dict.keys())[0]
        row_metrics = model_dict[first_key]
        rows.append({m: row_metrics.get(m) for m in common_metrics})
        index.append(name)

    # Grok / OpenAI / Gemma are already single dicts
    for name in ["grok_xai", "openai", "gemma"]:
        row_metrics = results[name]
        rows.append({m: row_metrics.get(m) for m in common_metrics})
        index.append(name)

    summary_df = pd.DataFrame(rows, index=index)

    # Make the table a bit nicer to read
    # - Order models in a fixed, logical order
    # - Round floats
    # - Show both model-wise and metric-wise views
    preferred_order = ["spai", "rine", "bfree","gemma", "grok_xai", "openai"]
    ordered = [m for m in preferred_order if m in summary_df.index]
    summary_df = summary_df.loc[ordered]

    # Round numeric columns for compact display
    summary_df = summary_df.astype(float).round(4)

    print("\n=== Summary comparison (primary operating point) ===")
    print("Models as rows, metrics as columns:\n")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(summary_df)

    return results






def compare_all_models_overtime(
    spai_ai_csv,
    spai_misc_csv,
    rine_ai_csv,
    rine_misc_csv,
    bfree_ai_csv,
    bfree_misc_csv,
    grok_ai_csv,
    grok_misc_csv,
    openai_ai_csv,
    openai_misc_csv,
    gemma_ai_csv,
    gemma_misc_csv,
    start_year = 2023,
    end_year = 2026,
    skip_2026 = False,
    save_plot = False,
    plot_path = "benchmark_evaluation/models_overtime.png",
):
    """
    Compare all models over time in 6-month spans (H1/H2) on Recall (TPR), FPR, Balanced Accuracy.

    Uses the *_overtime helpers for each model and plots them side-by-side.
    - If skip_2026=True: drop ALL 2026 buckets from tables/plots (even if present in inputs).
    """
    # Run half-year evaluators (no individual plots)
    df_openai = evaluate_openai_ai_miscaptioned_overtime(
        path_ai=openai_ai_csv,
        path_miscaptioned=openai_misc_csv,
        start_year=start_year,
        end_year=end_year,
        plot=False,
    )
    df_grok = evaluate_grok_xai_ai_miscaptioned_overtime(
        path_ai=grok_ai_csv,
        path_miscaptioned=grok_misc_csv,
        start_year=start_year,
        end_year=end_year,
        plot=False,
    )
    df_gemma = evaluate_gemma_ai_miscaptioned_overtime(
        path_ai=gemma_ai_csv,
        path_miscaptioned=gemma_misc_csv,
        start_year=start_year,
        end_year=end_year,
        plot=False,
    )
    df_rine = evaluate_rine_ai_miscaptioned_overtime(
        path_ai=rine_ai_csv,
        path_miscaptioned=rine_misc_csv,
        start_year=start_year,
        end_year=end_year,
        plot=False,
    )
    df_bfree = evaluate_bfree_ai_miscaptioned_overtime(
        path_ai=bfree_ai_csv,
        path_miscaptioned=bfree_misc_csv,
        start_year=start_year,
        end_year=end_year,
        plot=False,
    )
    df_spai = evaluate_spai_ai_miscaptioned_overtime(
        path_ai=spai_ai_csv,
        path_miscaptioned=spai_misc_csv,
        start_year=start_year,
        end_year=end_year,
        plot=False,
    )

    # Use OpenAI half-year buckets as canonical ordering
    df_openai = df_openai.sort_values("bucket_start").reset_index(drop=True)
    buckets = df_openai["bucket"].tolist()
    if skip_2026:
        buckets = [b for b in buckets if not str(b).startswith("2026-")]

    # All half-year evaluators already expose 'bucket' and 'bucket_start'
    per_model = {
        "SPAI": df_spai.set_index("bucket"),
        "RINE": df_rine.set_index("bucket"),
        "BFREE": df_bfree.set_index("bucket"),
        "Gemma-3": df_gemma.set_index("bucket"),
        "Grok": df_grok.set_index("bucket"),
        "GPT-5-Mini": df_openai.set_index("bucket"),
    }

    metrics_to_show = ["recall", "fpr", "balanced_accuracy"]
    metric_pretty = {
        "recall": "TPR",
        "fpr": "FPR",
        "balanced_accuracy": "Accuracy",
    }

    # Print counts per half-year bucket (sanity check)
    print("\n[Counts] Number of entries per half-year bucket:")
    count_rows = []
    for model_name, df_m in per_model.items():
        row = {"model": model_name}
        for b in buckets:
            if b in df_m.index:
                row[b] = int(df_m.loc[b, "n_total"])
            else:
                row[b] = 0
        count_rows.append(row)
    df_counts = pd.DataFrame(count_rows).set_index("model")
    df_counts = df_counts[buckets]
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(df_counts)

    print("\n=== Over-time comparison (per half-year/period) ===")
    for metric in metrics_to_show:
        rows = []
        for model_name, df_m in per_model.items():
            row = {"model": model_name}
            for b in buckets:
                if b in df_m.index:
                    row[b] = float(df_m.loc[b, metric])
                else:
                    row[b] = float("nan")
            rows.append(row)
        df_metric = pd.DataFrame(rows).set_index("model")
        df_metric = df_metric[buckets].round(4)
        print(f"\n--- {metric_pretty[metric]} ---")
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(df_metric)

    # Dynamic y-limits per metric across all models/buckets
    metric_ranges = {}
    for metric in metrics_to_show:
        vals = []
        for _, df_m in per_model.items():
            for b in buckets:
                if b not in df_m.index:
                    continue
                v = df_m.loc[b, metric]
                if pd.isna(v):
                    continue
                vals.append(float(v))
        if vals:
            vmin, vmax = min(vals), max(vals)
            span = vmax - vmin
            if span <= 0:
                span = 0.1
            pad = 0.1 * span
            lower = max(0.0, vmin - pad)
            upper = min(1.0, vmax + pad)
            metric_ranges[metric] = (lower, upper)
        else:
            metric_ranges[metric] = (0.0, 1.0)

    # Plot

    sns.set_theme(style="ticks")
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2.5,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "TeX Gyre Termes", "Nimbus Roman", "DejaVu Serif"],
    })


    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharex=True, sharey=False)

    
    colors = {
        "SPAI": "#4C78A8",      # blue
        "RINE": "#F58518",      # orange
        "BFREE": "#54A24B",     # green
        "Gemma-3": "#72B7B2",   # teal
        "Grok": "#B279A2",      # purple
        "GPT-5-Mini": "#E45756", # red
    }

    markers = {
        "SPAI": "o",
        "RINE": "s",
        "BFREE": "^",
        "Gemma-3": "D",
        "Grok": "v",
        "GPT-5-Mini": "P",
    }

    bucket_keys = buckets
    x = np.arange(len(bucket_keys))

    year_to_positions = {}
    for i, b in enumerate(bucket_keys):
        year = str(b).split("-")[0]
        year_to_positions.setdefault(year, []).append(i)

    xtick_positions = [sum(pos) / len(pos) for pos in year_to_positions.values()]
    xtick_labels = list(year_to_positions.keys())


    for metric, ax in zip(metrics_to_show, axes):
        for model_name, df_m in per_model.items():
            ys = []
            for b in bucket_keys:
                if b in df_m.index:
                    ys.append(float(df_m.loc[b, metric]))
                else:
                    ys.append(float("nan"))
            ax.plot(
                x,
                ys,
                marker=markers.get(model_name, "o"),
                markersize=6,
                markeredgewidth=0.5,
                linewidth=2,
                label=model_name,
                color=colors.get(model_name, None),
                zorder=3,
            )


        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels, rotation=0, ha="center")

        ax.set_title(metric_pretty[metric], pad=6)
        ax.set_ylim(*metric_ranges[metric])
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.18)
        ax.grid(axis="x", visible=False)

        for spine in ["left", "bottom"]:
            ax.spines[spine].set_linewidth(0.5)
            ax.spines[spine].set_color("0.6")

        ax.tick_params(axis="x", which="major", length=1.5, width=0.4, colors="0.4", pad=1)
        ax.tick_params(axis="y", which="major", length=0, colors="0.4", pad=1)

        # subtle separators between years
        for v in [1.5, 3.5]:
            ax.axvline(v, color="0.75", alpha=0.18, linewidth=0.6, zorder=1)
        

    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=True,
        framealpha=0.85,
        bbox_to_anchor=(0.5, -0.005),
        borderpad=0.25,
        columnspacing=1.0,
        handlelength=1.5,
        handletextpad=0.4,
    )
    leg.get_frame().set_alpha(0.7)
    leg.get_frame().set_linewidth(0.0)

    
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])


    if save_plot:
        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)

        # ensure pdf extension
        if not plot_path.endswith(".pdf"):
            plot_path = plot_path.replace(".png", ".pdf")

        fig.savefig(
            plot_path,
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.01,
        )
    else:
        fig.show()

    return per_model






path_config = {
    "SPAI": {
        "ai": "spai_images_enriched_ai_images.csv",
        "misc": "spai_enriched_miscaptioned_images.csv",
    },
    "RINE": {
        "ai": "itw_rine_ai_images_enriched.csv",
        "misc": "itw_rine_miscaptioned_images_enriched.csv",
    },
    "BFREE": {
        "ai": "bfree_ai_images_enriched.csv",
        "misc": "bfree_miscaptioned_images_enriched.csv",
    },
    "Grok XAI": {
        "ai": "xai_grok4-1-non-reasoning_ai_results.csv",
        "misc": "xai_grok4-1-non-reasoning_miscaptioned_results.csv",
    },
    "OpenAI": {
        "ai": "openai_gpt-5-mini_ai_results.csv",
        "misc": "openai_gpt-5-mini_miscaptioned_results.csv",
    },
    "Gemma-3": {
        "ai": "gemma3_ai.csv",
        "misc": "gemma3_miscaptioned.csv",
    },
}





"""
    Evaluate all models and plot the results both overall and overtime
"""

if __name__ == "__main__":

    results = compare_all_models(
        spai_ai_csv=path_config["SPAI"]["ai"],
        spai_misc_csv=path_config["SPAI"]["misc"],
        rine_ai_csv=path_config["RINE"]["ai"],
        rine_misc_csv=path_config["RINE"]["misc"],
        bfree_ai_csv=path_config["BFREE"]["ai"],
        bfree_misc_csv=path_config["BFREE"]["misc"],
        grok_ai_csv=path_config["Grok XAI"]["ai"],
        grok_misc_csv=path_config["Grok XAI"]["misc"],
        openai_ai_csv=path_config["OpenAI"]["ai"],
        openai_misc_csv=path_config["OpenAI"]["misc"],
        gemma_ai_csv=path_config["Gemma-3"]["ai"],
        gemma_misc_csv=path_config["Gemma-3"]["misc"],
    )




    # Half-year comparison (H1/H2, with 2025-H2+2026-H1 combined by default)
    halfyear_results = compare_all_models_overtime(
        spai_ai_csv=path_config["SPAI"]["ai"],
        spai_misc_csv=path_config["SPAI"]["misc"],
        rine_ai_csv=path_config["RINE"]["ai"],
        rine_misc_csv=path_config["RINE"]["misc"],
        bfree_ai_csv=path_config["BFREE"]["ai"],
        bfree_misc_csv=path_config["BFREE"]["misc"],
        grok_ai_csv=path_config["Grok XAI"]["ai"],
        grok_misc_csv=path_config["Grok XAI"]["misc"],
        openai_ai_csv=path_config["OpenAI"]["ai"],
        openai_misc_csv=path_config["OpenAI"]["misc"],
        gemma_ai_csv=path_config["Gemma-3"]["ai"],
        gemma_misc_csv=path_config["Gemma-3"]["misc"],
        skip_2026 = True,
        save_plot=True,
        plot_path="benchmark_evaluation/benchmark_eval_overtime.pdf",
    )

















