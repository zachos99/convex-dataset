import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  
import numpy as np
import matplotlib as mpl
from matplotlib import dates as mdates



def parse_mixed_note_date(series):
    """
    Parse a Series containing mixed date representations:
      - ISO timestamps (possibly with timezone)
      - epoch millis (as string/int)
      - human-readable dates like 'July 04, 2025'

    Returns a UTC-aware datetime Series (dtype datetime64[ns, UTC]).
    Unparseable entries -> NaT.
    """
    s = series.astype(str).str.strip()
    s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    # Heuristic: epoch milliseconds are usually 13 digits (sometimes 12–17)
    is_millis = s.str.fullmatch(r"\d{12,17}", na=False)

    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")

    # Parse millis subset
    if is_millis.any():
        out.loc[is_millis] = pd.to_datetime(
            s.loc[is_millis].astype("int64"),
            unit="ms",
            utc=True,
            errors="coerce",
        )

    # Parse everything else (ISO, 'July 04, 2025', etc.)
    if (~is_millis).any():
        out.loc[~is_millis] = pd.to_datetime(
            s.loc[~is_millis],
            utc=True,
            errors="coerce",
        )

    return out


def _metric_key_for_filename(metric):
    """
    Produce a short, stable filename token from a metric (string or list/tuple).
    Examples:
      ("retweet_count","reply_count","favorite_count") -> "retweet_reply_favorite"
      "view_count" -> "view"
    """
    if isinstance(metric, (list, tuple)):
        parts = [str(m) for m in metric]
    else:
        parts = [str(metric)]

    def _clean_one(s):
        s = s.strip().lower()
        if s.endswith("_count"):
            s = s[:-6]
        return s

    cleaned = [_clean_one(p) for p in parts if p]
    return "_".join(cleaned)

PRETTY_TYPE = {
    "miscaptioned": "Miscaptioned",
    "edited": "Edited",
    "ai_generated": "AI-generated",
}

 
def _draw_virality_lift(
    diag,
    keys,
    percentile,
    metric_key,
):
    """
    Draw the lift plot (Virality Share = P(type|viral)/P(type)) 
    """
    sns.set_theme(style="ticks")
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 6,
        "xtick.labelsize": 5,
        "ytick.labelsize": 4,
        "lines.linewidth": 1.6,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "TeX Gyre Termes", "Nimbus Roman", "DejaVu Serif"],
    })

    CB = {
        "miscaptioned": "#4C78A8",
        "edited": "#F58518",
        "ai_generated": "#54A24B",
    }
    MARKERS = {"miscaptioned": "o", "edited": "^", "ai_generated": "D"}
    type_order = ["miscaptioned", "edited", "ai_generated"]
    wide_lift = (
        diag.pivot(index="month", columns="misinfo_type_final", values="p2_over_p1")
        .sort_index()
    )
    marker_every = 4

    fig, ax = plt.subplots(figsize=(3.25, 2.4))

    for t in type_order:
        if t not in wide_lift.columns:
            continue
        s = wide_lift[t].dropna()
        if s.empty:
            continue
        ax.plot(
            s.index,
            s.values,
            color=CB.get(t, "0.2"),
            linestyle="-",
            linewidth=1.1,
            marker=MARKERS.get(t, "o"),
            markersize=4,
            markeredgewidth=0.3,
            markevery=min(marker_every, max(1, len(s) // 4)),
            label=PRETTY_TYPE.get(t, t),
            zorder=3,
        )

    # Reference line at 1.0 (no percentages on y-axis)
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.0, alpha=0.8, zorder=1)

    # Numeric y-axis: 3.0, 2.5, 2.0, ... (no PercentFormatter)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%Y"))
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.18)
    ax.grid(axis="x", visible=False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(0.5)
        ax.spines[spine].set_color("0.6")
    ax.tick_params(axis="x", which="major", length=1.5, width=0.4, colors="0.4")
    ax.tick_params(axis="y", which="major", length=0, colors="0.4")
    ax.tick_params(axis="x", labelsize=5)
    ax.tick_params(axis="y", pad=1)
    ax.tick_params(axis="x", pad=1)

    
    if keys == "image":
        anchor = (0.03, 1)
        loc = "upper left"
    elif keys == "video":
        anchor = (0.97, 1)
        loc = "upper right"

    leg = ax.legend(
        title=None,
        loc=loc,
        bbox_to_anchor=anchor,
        frameon=True,
        framealpha=0.85,
        borderpad=0.3,
        labelspacing=0.2,
        handlelength=1.6,
        handletextpad=0.5,
    )
    leg.get_frame().set_alpha(0.7)
    leg.get_frame().set_linewidth(0.0)
    for line in leg.get_lines():
        line.set_linewidth(1.0)
    for handle in leg.legendHandles:
        handle.set_markersize(3.5)

    ax.set_ylabel("Virality Share", labelpad=1)
    ax.set_xlabel("")
    xmin = wide_lift.index.min()
    xmax = wide_lift.index.max()
    ax.set_xlim(xmin - pd.Timedelta(days=10), xmax + pd.Timedelta(days=10))
    fig.subplots_adjust(left=0.12, right=0.998, bottom=0.22, top=0.98)

    outpath = f"p{percentile}_virality_share_{metric_key}_{keys}.pdf"
    fig.savefig(outpath, format="pdf", bbox_inches="tight", pad_inches=0.0, transparent=False)
    plt.close(fig)
    print(f"Saved: {outpath}")


def _plot_virality_single(
    pth,
    keys,
    metric="view_count",          # string OR list/tuple
    start_month="2023-05",
    percentile=90,
    min_n_per_point=30,
    min_viral_n_per_month=30,
):
    df = pd.read_csv(pth, low_memory=False)

    df["noteDate"] = parse_mixed_note_date(df["noteDate"]).dt.tz_convert(None)  # tz-naive


    df = df[df["misinfo_type_final"].isin(["miscaptioned", "edited", "ai_generated"])].copy()
    # df = df[df["noteDate"] >= pd.to_datetime(start_month)]

    if start_month:
        start_ts = pd.to_datetime(start_month, format="%Y-%m", errors="coerce")
        if pd.isna(start_ts):
            raise ValueError("start_month must be in 'YYYY-MM' format")

        df = df.dropna(subset=["noteDate"])
        df = df[df["noteDate"] >= start_ts].copy()

    print(
        f"[Diagnostics] rows={len(df):,} | "
        f"noteDate NaT={df['noteDate'].isna().sum()} | "
        f"range={df['noteDate'].min()} → {df['noteDate'].max()}"
    )


    original_metric = metric

    required_cols = ["noteDate", "misinfo_type_final", "tweet_id"]

    # metric can be string or list/tuple
    if isinstance(original_metric, (list, tuple)):
        missing = [c for c in original_metric if c not in df.columns]
        if missing:
            raise ValueError(f"Missing metric columns: {missing}")
        required_cols += list(original_metric)
    else:
        if original_metric not in df.columns:
            raise ValueError(f"Missing metric column: {original_metric}")
        required_cols.append(original_metric)

    df = df.dropna(subset=required_cols)


    if isinstance(original_metric, (list, tuple)):
        for c in original_metric:
            df[c] = pd.to_numeric(df[c], errors="coerce").clip(lower=0)
    else:
        df[original_metric] = pd.to_numeric(df[original_metric], errors="coerce").clip(lower=0)

    df = df.dropna(subset=required_cols)


    df = df.sort_values(["tweet_id", "noteDate"]).drop_duplicates("tweet_id", keep="first")

    # ---- build a single metric column (metric_value) ----
    if isinstance(original_metric, (list, tuple)):
        metric_col = "_combined_metric"
        df[metric_col] = df[list(original_metric)].fillna(0).sum(axis=1)
        pretty_metric = " + ".join(m.replace("_", " ") for m in original_metric).title()
        metric_key = _metric_key_for_filename(original_metric)
    else:
        metric_col = original_metric
        pretty_metric = str(original_metric).replace("_", " ").title()
        metric_key = _metric_key_for_filename(original_metric)

    # ---- normalize into score (what percentiles operate on) ----
    df["score"] = df[metric_col]


    df["month"] = df["noteDate"].dt.to_period("M").dt.to_timestamp()

    # ---------- drop incomplete last month ----------
    last_month = df["month"].max()
    df = df[df["month"] < last_month].copy()


    # ---------- volume composition (diagnostic) ----------
    volume = (
        df.groupby(["month", "misinfo_type_final"])
        .size()
        .rename("n")
        .reset_index()
    )
    volume["month_total"] = volume.groupby("month")["n"].transform("sum")
    volume["volume_pct"] = volume["n"] / volume["month_total"]

    # ---------- define thresholds (per_month) ----------
    p_m = df.groupby("month")["score"].transform(lambda x: np.percentile(x, percentile))
    df["is_viral"] = df["score"] >= p_m

    # ---------- aggregate ----------
    g = df.groupby(["month", "misinfo_type_final"])
    agg = g["is_viral"].mean().rename("rate").reset_index()
    counts = g.size().rename("n").reset_index()
    agg = agg.merge(counts, on=["month", "misinfo_type_final"])

    agg.loc[agg["n"] < min_n_per_point, "rate"] = np.nan

    # ---------- composition among viral posts ----------
    comp = None
    viral = df[df["is_viral"]].copy()

    # post-share among viral 
    # numerator: viral posts by (month, type)
    num = (
        viral.groupby(["month", "misinfo_type_final"])
        .size()
        .rename("viral_n")
        .reset_index()
    )
    # denominator: all viral posts in month
    den = viral.groupby("month").size().rename("viral_total").reset_index()
    comp = num.merge(den, on="month")
    comp["share_among_viral"] = comp["viral_n"] / comp["viral_total"]
    # stability filter (since denominator is smaller/noisier)
    comp.loc[comp["viral_total"] < min_viral_n_per_month, "share_among_viral"] = np.nan

    # view-share among viral
    vnum = (
        viral.groupby(["month", "misinfo_type_final"])[metric_col]
        .sum()
        .rename("viral_views")
        .reset_index()
    )
    vden = (
        viral.groupby("month")[metric_col]
        .sum()
        .rename("viral_views_total")
        .reset_index()
    )
    comp = comp.merge(vnum, on=["month","misinfo_type_final"]).merge(vden, on="month")
    comp["view_share_among_viral"] = comp["viral_views"] / comp["viral_views_total"]


    # Static pooled statistics
    print("\n" + "="*90)
    print(
        f"[Static virality summary] "
        f"p{percentile} | baseline=per_month | metric={pretty_metric} | set={keys}"
    )
    print("="*90)

    # (1) P(viral | type)
    p_viral_given_type = (
        df.groupby("misinfo_type_final")["is_viral"]
        .mean()
        .rename("p_viral_given_type")
        .reset_index()
    )

    # (3) P(type | viral)
    viral = df[df["is_viral"]]
    p_type_given_viral = (
        viral.groupby("misinfo_type_final")
            .size()
            .rename("n_viral")
            .reset_index()
    )
    p_type_given_viral["p_type_given_viral"] = (
        p_type_given_viral["n_viral"] / max(len(viral), 1)
    )

    # merge
    stat = p_viral_given_type.merge(
        p_type_given_viral[["misinfo_type_final", "p_type_given_viral"]],
        on="misinfo_type_final",
        how="left"
    ).fillna({"p_type_given_viral": 0})

    # pretty ordering / labels if available
    type_order = ["miscaptioned", "edited", "ai_generated"]
    stat["type_order"] = stat["misinfo_type_final"].map(
        {t: i for i, t in enumerate(type_order)}
    ).fillna(999)
    stat = stat.sort_values("type_order")

    def _pct(x):
        return f"{100*x:.2f}%" if pd.notna(x) else "NA"

    for _, r in stat.iterrows():
        label = PRETTY_TYPE.get(r["misinfo_type_final"], r["misinfo_type_final"])
        print(
            f"- {label}: "
            f"P(viral | type) = {_pct(r['p_viral_given_type'])} | "
            f"P(type | viral) = {_pct(r['p_type_given_viral'])}"
        )

    print("="*90 + "\n")



    
    diag = comp.merge(
        volume[["month", "misinfo_type_final", "volume_pct"]],
        on=["month", "misinfo_type_final"],
        how="left",
    )
    diag["p2_over_p1"] = diag["share_among_viral"] / diag["volume_pct"]

    # ---------- lift plot ----------
    _draw_virality_lift(
        diag=diag,
        keys=keys,
        percentile=percentile,
        metric_key=metric_key,
    )
    return agg, comp, diag


def plot_virality(
    image_csv_path,
    video_csv_path,
    metric="view_count",          # string OR list/tuple
    percentile=90,
    min_n_per_point=30,
    min_viral_n_per_month=30,
):
    """
    Run the virality analysis for both modalities (image + video)
    Returns a dict: {"image": (agg, comp, diag), "video": (agg, comp, diag)}.
    """
    return {
        "image": _plot_virality_single(
            pth=image_csv_path,
            keys="image",
            metric=metric,
            start_month="2023-05",
            percentile=percentile,
            min_n_per_point=min_n_per_point,
            min_viral_n_per_month=min_viral_n_per_month,
        ),
        "video": _plot_virality_single(
            pth=video_csv_path,
            keys="video",
            metric=metric,
            start_month="2023-09",
            percentile=percentile,
            min_n_per_point=min_n_per_point,
            min_viral_n_per_month=min_viral_n_per_month,
        ),
    }


if __name__ == "__main__":

    """
       Plot the Virality Share
       - For the paper, we plot the p99 virality share (top 1% of posts)  by the metric of retweet_count, reply_count, favorite_count
    """

    plot_virality(
        image_csv_path="data/dataset_image.csv",
        video_csv_path="data/dataset_video.csv",
        metric=("retweet_count", "reply_count", "favorite_count"),
        percentile=99,
        min_n_per_point=10,
        min_viral_n_per_month=5,
    )
