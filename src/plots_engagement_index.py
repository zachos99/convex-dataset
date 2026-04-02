import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  
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


PRETTY_TYPE = {
    "miscaptioned": "Miscaptioned",
    "edited": "Edited",
    "ai_generated": "AI-generated",
}


def _metric_key_for_filename(metrics) -> str:
    if isinstance(metrics, (list, tuple)):
        parts = [str(m) for m in metrics]
    else:
        parts = [str(metrics)]

    def _clean_one(s: str) -> str:
        s = s.strip().lower()
        if s.endswith("_count"):
            s = s[:-6]
        return s

    cleaned = [_clean_one(p) for p in parts if p]
    return "_".join(cleaned)


def _draw_engagement_index(
    wide,
    keys,
    out_path,
    metrics,
):
    """Draw monthly median custom z-score index in paper style."""
    sns.set_theme(style="ticks")
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 6,
        "xtick.labelsize": 4,
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

    cb_paper = {
        "miscaptioned": "#4C78A8",
        "edited": "#F58518",
        "ai_generated": "#54A24B",
    }
    markers_paper = {"miscaptioned": "o", "edited": "^", "ai_generated": "D"}
    type_order = ["miscaptioned", "edited", "ai_generated"]
    marker_every = 4

    fig, ax = plt.subplots(figsize=(3.25, 2.4))

    for t in type_order:
        if t not in wide.columns:
            continue
        s = wide[t].dropna()
        if s.empty:
            continue
        ax.plot(
            s.index,
            s.values,
            color=cb_paper.get(t, "0.2"),
            linestyle="-",
            linewidth=1.1,
            marker=markers_paper.get(t, "o"),
            markersize=4,
            markeredgewidth=0.3,
            markevery=min(marker_every, max(1, len(s) // 4)),
            label=PRETTY_TYPE.get(t, t),
            zorder=3,
        )

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

    if keys == "video":
        anchor = (0, 0.97)
        loc = "upper left"
    elif keys == "image":
        anchor = (0.97, 0.0)
        loc = "lower right"
    else:
        anchor = (0.97, 1.0)
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

    ax.set_ylabel("Engagement Index", labelpad=1)
    ax.set_xlabel("")
    xmin, xmax = wide.index.min(), wide.index.max()
    ax.set_xlim(xmin - pd.Timedelta(days=10), xmax + pd.Timedelta(days=10))
    fig.subplots_adjust(left=0.12, right=0.998, bottom=0.22, top=0.98)

    if out_path is None:
        metric_key = _metric_key_for_filename(metrics)
        out_path = f"engagement_index_{metric_key}_{keys}.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.0, transparent=False)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _plot_engagement_index_single(
    csv_path,
    keys,
    start_month_fixed,
    metrics="view_count",
    time_col="noteDate",
    misinfo_type_column="misinfo_type_final",
    min_n_per_point=15,
    out_path=None,
):

    df = pd.read_csv(csv_path, low_memory=False)

    if time_col == "noteDate":
        df[time_col] = parse_mixed_note_date(df[time_col]).dt.tz_convert(None)  # tz-naive
    else:
        df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce").dt.tz_convert(None)

    df = df[df[misinfo_type_column].isin(["miscaptioned", "edited", "ai_generated"])].copy()

    start_ts = pd.to_datetime(start_month_fixed, format="%Y-%m", errors="coerce")
    if pd.isna(start_ts):
        raise ValueError("start_month must be in 'YYYY-MM' format")
    df = df[df[time_col] >= start_ts].copy()

    if isinstance(metrics, str):
        metrics = (metrics,)
    elif not isinstance(metrics, (list, tuple)):
        raise TypeError("metrics must be a string or list/tuple of strings")

    
    required_cols = [time_col, misinfo_type_column] + list(metrics)
    missing = [m for m in required_cols if m not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=required_cols).copy()

    if "tweet_id" in df.columns:
        df = df.sort_values(["tweet_id", time_col]).drop_duplicates(subset="tweet_id", keep="first")
    else:
        df = df.sort_values([time_col])

    df["month"] = df[time_col].dt.to_period("M").dt.to_timestamp()
    last_month = df["month"].max()
    df = df[df["month"] < last_month].copy()

    def _zscore(s: pd.Series) -> pd.Series:
        mu = s.mean()
        sd = s.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - mu) / sd

    z_cols = []
    suffix = ""
    for m in metrics:
        x = pd.to_numeric(df[m], errors="coerce").fillna(0).clip(lower=0)

        log_col = f"log1p_{m}{suffix}"
        z_col = f"z_{m}{suffix}"
        df[log_col] = np.log1p(x)
        # zscore_scope fixed to monthly
        df[z_col] = df.groupby("month", group_keys=False)[log_col].apply(_zscore)

        z_cols.append(z_col)

    z_reply = f"z_reply_count{suffix}"
    z_rt = f"z_retweet_count{suffix}"
    z_like = f"z_favorite_count{suffix}"

    missing_z = [c for c in (z_reply, z_rt, z_like) if c not in df.columns]
    if missing_z:
        raise ValueError(
            "Custom engagement index requires these metrics in `metrics`: "
            "reply_count, retweet_count, favorite_count"
        )

    lam = 2.0
    """
        Engagement Index Equation:
            Reply Count + Retweet Count - Lambda * Favorite Count
    """
    df["custom_score"] = df[z_reply] + df[z_rt] - lam * df[z_like]

    g = df.groupby(["month", misinfo_type_column])
    agg = g["custom_score"].median().rename("value").reset_index()
    monthly_n = g.size().rename("n").reset_index()
    agg = agg.merge(monthly_n, on=["month", misinfo_type_column])

    if min_n_per_point and min_n_per_point > 0:
        agg.loc[agg["n"] < min_n_per_point, "value"] = np.nan

    wide = agg.pivot(index="month", columns=misinfo_type_column, values="value").sort_index()

    _draw_engagement_index(
        wide=wide,
        keys=keys,
        out_path=out_path,
        metrics=metrics,
    )

    return df, agg


def plot_engagement_index(
    image_csv_path,
    video_csv_path,
    metrics="view_count",
    time_col="noteDate",
    misinfo_type_column="misinfo_type_final",
    min_n_per_point=15,
    out_path=None,
):
    """
    Run the engagement index plot for both modalities (image + video) in one call.
    Fixed start months:
      - image: 2023-05
      - video: 2023-09
    Returns {"image": (df, agg), "video": (df, agg)}.
    """
    return {
        "image": _plot_engagement_index_single(
            csv_path=image_csv_path,
            keys="image",
            start_month_fixed="2023-05",
            metrics=metrics,
            time_col=time_col,
            misinfo_type_column=misinfo_type_column,
            min_n_per_point=min_n_per_point,
            out_path=out_path,
        ),
        "video": _plot_engagement_index_single(
            csv_path=video_csv_path,
            keys="video",
            start_month_fixed="2023-09",
            metrics=metrics,
            time_col=time_col,
            misinfo_type_column=misinfo_type_column,
            min_n_per_point=min_n_per_point,
            out_path=out_path,
        ),
    }



if __name__ == "__main__":

    """

    Plot the Engagement Index
    - For the paper, we plot the engagement index by the metric of reply_count, retweet_count, favorite_count
    """

    results = plot_engagement_index(
        image_csv_path="data/dataset_image.csv",
        video_csv_path="data/dataset_video.csv",
        metrics=("reply_count", "retweet_count", "favorite_count"),
    )



