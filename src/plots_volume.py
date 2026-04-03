import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  
from matplotlib import dates as mdates
import matplotlib as mpl



def set_paper_style():
    sns.set_theme(style="ticks")#, context="paper")
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
        "pdf.fonttype": 42, # Embed TrueType fonts
        "ps.fonttype": 42,
        "font.family": "serif", 
        "font.serif": ["Times New Roman", "Times", "TeX Gyre Termes", "Nimbus Roman", "DejaVu Serif"],
    })

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



def plot_volume_weekly(
    image_csv_path,
    video_csv_path,
    misinfo_type_column="misinfo_type_final",
    include_types=("miscaptioned", "edited", "ai_generated"),
):
    """
    Plot weekly volume for both modalities (image + video) in one call.
    """

    def _plot_for_modality(pth, modality, start_month_fixed):
        # ---------- load ----------
        df = pd.read_csv(pth, parse_dates=["created_at_datetime", "noteDate"])

        # ---------- filter to requested types ----------
        main_types = {"miscaptioned", "edited", "ai_generated"}
        include_types_filtered = tuple(t for t in include_types if t in main_types)
        if not include_types_filtered:
            raise ValueError("include_types must contain at least one of: miscaptioned, edited, ai_generated")

        df = df[df[misinfo_type_column].isin(include_types_filtered)].copy()

        # Always enable these in the combined plot flow.
        show_total = False
        models_release_line = True
        deduplicate = True

        # ---------- time prep ----------
        # Parse columns (noteDate can be mixed: ISO, millis, 'July 04, 2025', etc.)
        df["noteDate"] = parse_mixed_note_date(df["noteDate"])
        df["created_at_datetime"] = pd.to_datetime(df["created_at_datetime"], utc=True, errors="coerce")

        # For plotting / grouping, make tz-naive consistently
        df["noteDate"] = df["noteDate"].dt.tz_convert(None)
        df["created_at_datetime"] = df["created_at_datetime"].dt.tz_convert(None)

        print(f"[{modality}] Unparseable noteDate rows: {df['noteDate'].isna().sum()}")

        # Choose time column
        time_col = "noteDate"

        # start-month filter (modality-specific)
        start_dt = pd.to_datetime(start_month_fixed, format="%Y-%m", errors="coerce")
        if pd.isna(start_dt):
            raise ValueError("start_month must be in 'YYYY-MM' format")
        df = df[df[time_col] >= start_dt]

        # drop invalid
        df = df.dropna(subset=[time_col, misinfo_type_column])

        if deduplicate:
            # Prefer tweet_id if available, otherwise fall back to tweetUrl
            dedup_key = "tweet_id" if "tweet_id" in df.columns else "tweetUrl"
            df = df.sort_values(time_col).drop_duplicates(
                subset=[dedup_key, misinfo_type_column],
                keep="first",
            )

        # weekly index (weeks starting Monday)
        df["week"] = df[time_col].dt.to_period("W-MON").apply(lambda r: r.start_time)

        # counts per type per week
        weekly_counts = (
            df.groupby(["week", misinfo_type_column])
              .size()
              .reset_index(name="count")
              .sort_values(["week", misinfo_type_column])
        )

        # total per week (optional)
        total_weekly = (
            df.groupby("week").size().reset_index(name="count")
            if show_total else pd.DataFrame()
        )

        # drop last (potentially incomplete) week based on max observed date
        if not df.empty and not weekly_counts.empty:
            max_dt = df[time_col].max()
            last_week_start = max_dt.to_period("W-MON").start_time
            weekly_counts = weekly_counts[weekly_counts["week"] < last_week_start]
            if show_total and not total_weekly.empty:
                total_weekly = total_weekly[total_weekly["week"] < last_week_start]

        LABELS = {
            "miscaptioned": "Miscaptioned",
            "edited": "Edited",
            "ai_generated": "AI-generated",
        }

        # ---------- plot ----------
        set_paper_style()
        fig, ax = plt.subplots(figsize=(3.25, 2.4))

        CB = {
            "miscaptioned": "#4C78A8",   # muted blue
            "edited": "#F58518",         # deeper orange
            "ai_generated": "#54A24B",   # teal (less bright green)
        }

        # One marker shape per type
        MARKERS = {
            "miscaptioned": "o",  # circle
            "edited": "^",        # square
            "ai_generated": "D",  # diamond
        }

        # Wide format for clean matplotlib plotting
        wide = weekly_counts.pivot(index="week", columns=misinfo_type_column, values="count").sort_index()

        marker_every = 8

        for t in include_types_filtered:
            if t not in wide.columns:
                continue
            ax.plot(
                wide.index,
                wide[t].values,
                color=CB.get(t, "0.2"),
                linestyle="-",
                linewidth=1.1,
                marker=MARKERS.get(t, "o"),
                markersize=4,
                markeredgewidth=0.3,
                markevery=marker_every,
                label=LABELS.get(t, t),
                zorder=3,
            )

        # Optional total line (reference)
        if show_total and not total_weekly.empty:
            ax.plot(
                total_weekly["week"],
                total_weekly["count"],
                color="0.35",
                linewidth=1.8,
                linestyle="--",
                alpha=0.7,
                label="Aggregate volume",
                zorder=0,
            )

        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%Y")) # stack month/year
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

        # --- Grid styling (monthly emphasis) ---
        # Subtle horizontal grid for magnitude reading
        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.18)
        ax.grid(axis="x", visible=False)

        # Vertical lines at month boundaries
        month_starts = pd.date_range(
            start=wide.index.min(),
            end=wide.index.max(),
            freq="MS",
        )
        for dt in month_starts:
            ax.axvline(dt, color="0.92", linewidth=0.6, zorder=0)

        for spine in ["left", "bottom"]:
            ax.spines[spine].set_linewidth(0.5)
            ax.spines[spine].set_color("0.6")

        ax.tick_params(axis="x", which="major", length=1.5, width=0.4, colors="0.4")
        ax.tick_params(axis="y", which="major", length=0, colors="0.4")
        ax.tick_params(axis="x", labelsize=5)

        ax.tick_params(axis="y", pad=1)
        ax.tick_params(axis="x", pad=1)

        # Keep current y-limits stable while adding model lines.
        ylim = ax.get_ylim()

        # Legend 1: types (inside)
        if modality == "image":
            anchor = (0, 0.82)  # (x, y) in axes fraction
        else:
            anchor = (0, 1)  # exact top-left corner

        leg1 = ax.legend(
            title=None,
            loc="upper left",
            bbox_to_anchor=anchor,
            frameon=True,
            framealpha=0.85,
            borderpad=0.3,
            labelspacing=0.2,
            handlelength=1.6,
            handletextpad=0.5,
        )
        leg1.get_frame().set_alpha(0.7)
        leg1.get_frame().set_linewidth(0.0)

        for line in leg1.get_lines():
            line.set_linewidth(1.0)

        for handle in leg1.legendHandles:
            handle.set_markersize(3.5)

        # ---------- model release lines ----------
        if models_release_line:
            if modality == "image":
                releases = {
                    "DALL·E 3": ("2023-10-15", "tab:blue"),
                    "Midjourney v6": ("2023-12-21", "tab:red"),
                    "DALL·E 3 free-tier": ("2024-08-08", "tab:blue"),
                    "GPT-4o Images": ("2025-04-15", "tab:purple"),
                    "Gemini 2.5 Flash Image": {
                        "date": "2025-08-26",
                        "color": "0.35",
                        "legend": "Gemini 2.5 Flash Image\n(Nano Banana)",
                    },
                    "Gemini 3 Pro Image": {
                        "date": "2025-11-10",
                        "color": "0.35",
                        "legend": "Gemini 3 Pro Image\n(Nano Banana Pro)",
                    },
                }
            else:
                releases = {
                    "Veo": ("2024-05-15", "tab:blue"),
                    "Veo 2": ("2024-12-01", "tab:blue"),
                    "Veo 3": ("2025-05-20", "tab:blue"),
                    "Sora": ("2024-12-09", "tab:red"),
                    "Sora 2": ("2025-09-30", "tab:red"),
                    "Veo 3.1": ("2025-10-15", "tab:blue"),
                }

            LABEL_POSITION_OVERRIDES = {
                "DALL·E 3 free-tier": {"side": "right", "vertical": "top", "dx_days": 5, "dy_frac": 0.0},
                "Sora": {"side": "right", "vertical": "top", "dx_days": 0, "dy_frac": 0.0},
                "Veo 3": {"side": "right", "vertical": "bottom", "dx_days": 0, "dy_frac": 0.0},
                "Veo 3.1": {"side": "right", "vertical": "top", "dx_days": 0, "dy_frac": 0.0},
                "DALL·E 3": {"side": "right", "vertical": "top", "dx_days": 5, "dy_frac": 0.0},
                "Midjourney v6": {"side": "right", "vertical": "top", "dx_days": 0, "dy_frac": 0.0},
                "Gemini 3 Pro Image": {"side": "right", "vertical": "top", "dx_days": 0, "dy_frac": 0.0},
            }

            # Normalize releases into: short_label -> {date, color, legend_label}
            norm_releases = {}
            for k, v in releases.items():
                if isinstance(v, dict):
                    norm_releases[k] = {
                        "date": v["date"],
                        "color": v["color"],
                        "legend": v.get("legend", k),
                    }
                else:
                    datestr, color = v
                    norm_releases[k] = {"date": datestr, "color": color, "legend": k}

            # Small model label styling
            if modality == "image":
                MODEL_FONT = 4.5   # slightly smaller
            else:
                MODEL_FONT = 5     # video

            MODEL_COLOR = "0.25"   # dark gray
            MODEL_LINE_W = 0.9
            MODEL_ALPHA = 0.75
            MODEL_STYLE = (0, (3, 2))  # dashed

            for i, (label_short, meta) in enumerate(norm_releases.items()):
                dt = pd.to_datetime(meta["date"])

                ax.axvline(
                    dt,
                    color=MODEL_COLOR,
                    linestyle=MODEL_STYLE,
                    linewidth=MODEL_LINE_W,
                    alpha=MODEL_ALPHA,
                    zorder=2,
                )

                # stagger x-offset slightly so labels don't overlap as much
                offset_days = 5 + (i % 3) * 3

                # defaults (match your previous behavior: top + left of line)
                side = "left"
                vertical = "top"
                dx_days = 0
                dy_frac = 0.0

                if label_short in LABEL_POSITION_OVERRIDES:
                    o = LABEL_POSITION_OVERRIDES[label_short]
                    side = o.get("side", side)
                    vertical = o.get("vertical", vertical)
                    dx_days = o.get("dx_days", dx_days)
                    dy_frac = o.get("dy_frac", dy_frac)

                # vertical placement
                if vertical == "bottom":
                    y = ylim[0] + 0.02 * (ylim[1] - ylim[0])
                    va = "bottom"
                else:
                    y = ylim[1] - 0.02 * (ylim[1] - ylim[0])
                    va = "top"

                y = y + dy_frac * (ylim[1] - ylim[0])

                # horizontal placement
                if side == "right":
                    x = dt + pd.Timedelta(days=offset_days + dx_days)
                    ha = "left"
                else:
                    x = dt - pd.Timedelta(days=offset_days + dx_days)
                    ha = "right"

                ax.text(
                    x,
                    0.98 if va == "top" else 0.004,  # top/bottom inside axes
                    label_short,
                    transform=ax.get_xaxis_transform(),
                    clip_on=True,  # IMPORTANT: don't affect bbox_inches tight
                    color=MODEL_COLOR,
                    fontsize=MODEL_FONT,
                    rotation=90,
                    va=va,
                    ha=ha,
                    zorder=5,
                )

        # keep y-limits unchanged
        ax.set_ylim(ylim)

        xmin = wide.index.min()
        xmax = wide.index.max()

        ax.set_xlim(
            xmin - pd.Timedelta(days=10),
            xmax + pd.Timedelta(days=10),
        )

        # keep type legend
        ax.add_artist(leg1)

        # Get current ticks
        ticks = ax.get_xticks()

        # Keep only ticks inside limits
        ticks = [t for t in ticks if xmin <= mdates.num2date(t).replace(tzinfo=None) <= xmax]

        ax.set_xticks(ticks)

        fig.subplots_adjust(left=0.005, right=0.998, bottom=0.22, top=0.98)

        # save figure
        start_txt = start_dt.strftime("%Y-%m")
        outpath = (
            f"volume_{modality}_{start_txt}_"
            f"{'-'.join(include_types_filtered)}{'_model_line' if models_release_line else ''}.pdf"
        )
        fig.savefig(
            outpath,
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.0,
            transparent=False,
        )
        plt.close(fig)
        print(f"Saved: {outpath}")

    # Image: start at 2023-05
    _plot_for_modality(image_csv_path, "image", "2023-05")
    # Video: start at 2023-09
    _plot_for_modality(video_csv_path, "video", "2023-09")




if __name__ == "__main__":
    """
        Plot number of notes per misinfo type
    """

    plot_volume_weekly(
        image_csv_path="data/dataset_image.csv",
        video_csv_path="data/dataset_video.csv",
        misinfo_type_column="misinfo_type_final",
        include_types=("miscaptioned", "edited", "ai_generated"),
    )


