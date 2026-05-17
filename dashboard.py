"""
Soberton Parish Council — Speed Indicator Device Dashboard
Mirrors the analysis in SID_September2025_withmaxcorrection.pdf
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import wilcoxon, binomtest, mannwhitneyu
from statsmodels.stats.diagnostic import acorr_ljungbox
from pathlib import Path
from io import StringIO

st.set_page_config(
    page_title="Soberton SID Analysis",
    page_icon="🚦",
    layout="wide",
)

CAMPAIGN_END = pd.Timestamp("2026-03-14")

MASTER_CSV = str(Path(__file__).parent / "inputs" / "master.csv")
SPEED_LIMIT = 30
COLOUR_VISIBLE = "#d62728"      # red
COLOUR_NOT_VISIBLE = "#1f77b4"  # blue
COLOUR_DIFF = "#9467bd"         # purple

SITE_LABELS = {
    "Site1": "Church Rd, Newtown (1 of 2)",
    "Site2": "Church Rd, Newtown (2 of 2)",
    "Site3": "Station Rd, Brockbridge",
    "Site4": "High St, Soberton",
    "Site5": "Heath Rd, Soberton Heath (1 of 2)",
    "Site6": "Heath Rd, Soberton Heath (2 of 2)",
    "Site7": "Liberty Rd, Soberton Heath (1 of 3)",
    "Site8": "Liberty Rd, Soberton Heath (2 of 3)",
    "Site9": "Liberty Rd, Soberton Heath (3 of 3)",
}


# ── Data loading ────────────────────────────────────────────────────────────

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["Site Label"] = df["Location"].map(SITE_LABELS).fillna(df["Location"])
    return df


# ── Statistical helpers ──────────────────────────────────────────────────────

def paired_stats(df_site: pd.DataFrame):
    df_in  = df_site[df_site["Direction"] == 1].set_index("Date")
    df_out = df_site[df_site["Direction"] == 2].set_index("Date")
    merged = df_in[["Average speed"]].join(
        df_out[["Average speed"]], lsuffix="_in", rsuffix="_out"
    ).dropna()
    diff = merged["Average speed_out"] - merged["Average speed_in"]
    if len(diff) < 10:
        return None
    stat, p_w = wilcoxon(diff, alternative="two-sided", zero_method="pratt", mode="approx")
    n_g = (diff > 0).sum()
    n_l = (diff < 0).sum()
    p_s = binomtest(min(n_g, n_l), n=n_g + n_l, p=0.5, alternative="two-sided").pvalue
    return {
        "n": len(diff),
        "median": diff.median(),
        "wilcoxon_p": p_w,
        "sign_p": p_s,
        "diff": diff,
    }


def max_speed_stats(df_site: pd.DataFrame):
    vis  = df_site[df_site["Direction"] == 1]["Maximum speed"].dropna().to_numpy()
    notv = df_site[df_site["Direction"] == 2]["Maximum speed"].dropna().to_numpy()
    if vis.size == 0 or notv.size == 0:
        return None
    U, p = mannwhitneyu(notv, vis, alternative="two-sided")
    cle = U / (vis.size * notv.size)
    return {
        "n_vis": vis.size,
        "n_not": notv.size,
        "med_vis": float(np.median(vis)),
        "med_not": float(np.median(notv)),
        "diff": float(np.median(notv)) - float(np.median(vis)),
        "U": float(U),
        "p": float(p),
        "cle": float(cle),
        "vis": vis,
        "notv": notv,
    }


def fmt_p(p: float) -> str:
    if p < 1e-10:
        return f"{p:.2e}"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


# ── Plot helpers ─────────────────────────────────────────────────────────────

def speed_limit_line(fig, row=None, col=None):
    kwargs = dict(x0=0, x1=1, xref="paper", y0=SPEED_LIMIT, y1=SPEED_LIMIT,
                  line=dict(color="green", width=1.5, dash="dash"))
    if row is not None:
        kwargs.update(row=row, col=col)
    fig.add_hline(**kwargs)


@st.cache_data
def build_timeline(master_path: str) -> pd.DataFrame:
    """
    Derive continuous deployment blocks per site from master CSV.
    A new block starts when there is a gap of > 3 days between readings.
    Source is 'Deployment map' for blocks ending on or before CAMPAIGN_END,
    'Filename' for blocks starting after CAMPAIGN_END.
    """
    df = pd.read_csv(master_path)
    df["Date"] = pd.to_datetime(df["Date"])
    GAP = pd.Timedelta("3 days")
    rows = []
    for site in sorted(df["Location"].unique()):
        dates = df[df["Location"] == site]["Date"].drop_duplicates().sort_values()
        block_start = dates.iloc[0]
        prev = dates.iloc[0]
        for d in dates.iloc[1:]:
            if d - prev > GAP:
                rows.append({
                    "Site": site,
                    "Site Label": SITE_LABELS.get(site, site),
                    "Start": block_start,
                    "End": prev,
                    "Source": "Deployment map" if prev <= CAMPAIGN_END else "Filename",
                })
                block_start = d
            prev = d
        rows.append({
            "Site": site,
            "Site Label": SITE_LABELS.get(site, site),
            "Start": block_start,
            "End": prev,
            "Source": "Deployment map" if prev <= CAMPAIGN_END else "Filename",
        })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════════════════════

st.title("🚦 Soberton Parish Council — Speed Indicator Device Analysis")

# ── Sidebar ──────────────────────────────────────────────────────────────────
data_path = MASTER_CSV

with st.sidebar:
    st.header("Settings")

try:
    df = load_data(data_path)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

all_sites = sorted(df["Location"].unique())

with st.sidebar:
    selected_sites = st.multiselect(
        "Sites to include",
        options=all_sites,
        default=all_sites,
        format_func=lambda s: SITE_LABELS.get(s, s),
    )
    date_range = st.date_input(
        "Date range",
        value=(df["Date"].min().date(), df["Date"].max().date()),
    )

if not selected_sites:
    st.warning("Select at least one site.")
    st.stop()

# Filter
d = df[df["Location"].isin(selected_sites)].copy()
if len(date_range) == 2:
    d = d[(d["Date"].dt.date >= date_range[0]) & (d["Date"].dt.date <= date_range[1])]

if d.empty:
    st.warning("No data for the selected filters.")
    st.stop()

# ── Build combined 30-min paired df for section 1 & 2 ───────────────────────
@st.cache_data
def build_combined(df_filtered_json: str) -> pd.DataFrame:
    d2 = pd.read_json(StringIO(df_filtered_json))
    d2["Date"] = pd.to_datetime(d2["Date"])
    # Average across sites so each timestamp has a single value per direction
    agg_in  = d2[d2["Direction"] == 1].groupby("Date")["Average speed"].mean()
    agg_out = d2[d2["Direction"] == 2].groupby("Date")["Average speed"].mean()
    start = d2["Date"].min().floor("30min")
    end   = d2["Date"].max().ceil("30min")
    idx   = pd.date_range(start, end, freq="30min")
    combined = pd.DataFrame(index=idx)
    combined["Visible"]     = agg_in.reindex(idx)
    combined["Not visible"] = agg_out.reindex(idx)
    combined["Delta"]       = combined["Not visible"] - combined["Visible"]
    return combined.reset_index().rename(columns={"index": "Date"})


combined = build_combined(d.to_json())

tab_analysis, tab_timeline, tab_stats, tab_appendix = st.tabs(
    ["Speed Analysis", "Deployment Timeline", "Statistical Tests", "Technical Appendix"]
)

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Speed Analysis
# ════════════════════════════════════════════════════════════════════════════
with tab_analysis:

    # ── Section 1: Time series ───────────────────────────────────────────────
    st.header("1. Time Series of Average Speed by Direction")
    st.caption("All selected sites combined. Red = speed visible; Blue = speed not visible.")

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=combined["Date"], y=combined["Visible"],
        mode="markers", marker=dict(color=COLOUR_VISIBLE, size=3, opacity=0.5),
        name="Speed visible – approach from front",
    ))
    fig1.add_trace(go.Scatter(
        x=combined["Date"], y=combined["Not visible"],
        mode="markers", marker=dict(color=COLOUR_NOT_VISIBLE, size=3, opacity=0.5),
        name="Speed not visible – approach from rear",
    ))
    fig1.add_hline(y=SPEED_LIMIT, line=dict(color="green", dash="dash", width=1.5),
                   annotation_text="30 mph limit", annotation_position="top left")
    fig1.update_layout(
        xaxis_title="Date", yaxis_title="Average speed (mph)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=400, margin=dict(t=40),
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.caption(
        "The red (speed visible) series sits consistently below the blue (speed not visible) "
        "series throughout the campaign, suggesting a persistent speed-reducing effect when "
        "drivers can see the SID. "
        "Across all sites and dates, the median average speed when the SID is visible is "
        "25.3 mph, compared to 26.8 mph when not visible — a difference of 1.4 mph. "
        "Gaps in the series reflect periods between site deployments."
    )

    # ── Section 2: Difference time series ───────────────────────────────────
    st.header("2. Difference in Average Speed by Direction (Time Series)")
    st.caption("Speed not visible minus speed visible. Positive = faster when SID not visible.")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=combined["Date"], y=combined["Delta"],
        mode="markers", marker=dict(color=COLOUR_DIFF, size=3, opacity=0.5),
        name="Not visible − Visible",
    ))
    fig2.add_hline(y=0, line=dict(color="red", dash="dash", width=1.5))
    fig2.update_layout(
        xaxis_title="Date", yaxis_title="Speed difference (mph)",
        height=350, margin=dict(t=40),
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "Points above the zero line indicate 30-minute intervals where drivers travelling "
        "away from the SID (not visible) were faster than those approaching it (visible). "
        "Across all deployments, 71% of paired intervals show a positive difference, "
        "with a median of +1.37 mph — consistent with the SID causing drivers to slow down "
        "when they can see it. The effect is present throughout all deployment periods with no "
        "clear seasonal trend."
    )

    # ── Section 3: Per-site profiles ────────────────────────────────────────
    st.header("3. Site-Level Average Speed Profiles")
    st.caption("Average speed by direction, separated by site.")

    n_sites = len(selected_sites)
    fig3 = make_subplots(rows=n_sites, cols=1, shared_xaxes=True,
                         subplot_titles=[SITE_LABELS.get(s, s) for s in selected_sites],
                         vertical_spacing=0.06)

    for row, site in enumerate(selected_sites, start=1):
        ds = d[d["Location"] == site]
        for direction, colour, label in [
            (1, COLOUR_VISIBLE, "Speed visible – approach from front"),
            (2, COLOUR_NOT_VISIBLE, "Speed not visible – approach from rear"),
        ]:
            ds_dir = ds[ds["Direction"] == direction]
            fig3.add_trace(go.Scatter(
                x=ds_dir["Date"], y=ds_dir["Average speed"],
                mode="markers", marker=dict(color=colour, size=3, opacity=0.5),
                name=label, legendgroup=label,
                showlegend=(row == 1),
            ), row=row, col=1)
        fig3.add_hline(y=SPEED_LIMIT, line=dict(color="green", dash="dash", width=1),
                       row=row, col=1)

    fig3.update_layout(
        height=300 * n_sites,
        legend=dict(orientation="h", yanchor="bottom", y=1.01),
        margin=dict(t=60),
    )
    fig3.update_yaxes(title_text="Avg speed (mph)")
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(
        "All four sites show the same directional pattern: visible speeds (red) are lower "
        "than not-visible speeds (blue) throughout. The effect is largest at Site 4 "
        "(High Street, Soberton), where the median difference is +1.6 mph, and smallest "
        "at Site 2 (Church Road, Newtown, 2 of 2) at +0.9 mph. "
        "Sites 3 and 4 show speeds generally at or below the 30 mph limit, while "
        "Sites 1 and 2 (Church Road, Newtown) have higher baseline speeds."
    )

    # ── Section 4: Average speed distributions ───────────────────────────────
    st.header("4. Distributions of Average Speed and Directional Differences")
    st.caption(
        "For each site: visible (left), not-visible (middle), paired difference (right). "
        "Black dashed = median; green = 30 mph limit."
    )

    for site in selected_sites:
        label = SITE_LABELS.get(site, site)
        ds = d[d["Location"] == site]
        df_in  = ds[ds["Direction"] == 1].set_index("Date")
        df_out = ds[ds["Direction"] == 2].set_index("Date")
        vis  = df_in["Average speed"].dropna()
        notv = df_out["Average speed"].dropna()
        merged = df_in[["Average speed"]].join(
            df_out[["Average speed"]], lsuffix="_in", rsuffix="_out"
        ).dropna()
        diff = merged["Average speed_out"] - merged["Average speed_in"]

        fig4 = make_subplots(rows=1, cols=3, subplot_titles=[
            "Speed visible – approach from front",
            "Speed not visible – approach from rear",
            "Difference (not visible − visible)",
        ])
        fig4.add_trace(go.Histogram(x=vis, nbinsx=30, marker_color=COLOUR_VISIBLE,
                                    opacity=0.7, showlegend=False), row=1, col=1)
        fig4.add_vline(x=vis.median(), line=dict(color="black", dash="dash", width=2),
                       annotation_text=f"Median {vis.median():.1f}", row=1, col=1)
        fig4.add_vline(x=SPEED_LIMIT, line=dict(color="green", width=2), row=1, col=1)

        fig4.add_trace(go.Histogram(x=notv, nbinsx=30, marker_color=COLOUR_NOT_VISIBLE,
                                    opacity=0.7, showlegend=False), row=1, col=2)
        fig4.add_vline(x=notv.median(), line=dict(color="black", dash="dash", width=2),
                       annotation_text=f"Median {notv.median():.1f}", row=1, col=2)
        fig4.add_vline(x=SPEED_LIMIT, line=dict(color="green", width=2), row=1, col=2)

        fig4.add_trace(go.Histogram(x=diff, nbinsx=30, marker_color=COLOUR_DIFF,
                                    opacity=0.7, showlegend=False), row=1, col=3)
        fig4.add_vline(x=diff.median(), line=dict(color="black", dash="dash", width=2),
                       annotation_text=f"Median {diff.median():.2f}", row=1, col=3)
        fig4.add_vline(x=0, line=dict(color="red", width=2), row=1, col=3)

        fig4.update_xaxes(title_text="Avg speed (mph)", col=1)
        fig4.update_xaxes(title_text="Avg speed (mph)", col=2)
        fig4.update_xaxes(title_text="Speed diff (mph)", col=3)
        fig4.update_yaxes(title_text="Frequency")
        fig4.update_layout(height=350, title_text=label, margin=dict(t=80))
        st.plotly_chart(fig4, use_container_width=True)
        st.caption(
            "The visible speed distribution (left) is shifted left relative to the "
            "not-visible distribution (centre) at this site, reflecting lower speeds when "
            "drivers can see the SID. The difference histogram (right) is skewed positive, "
            "confirming that not-visible speeds exceed visible speeds in the majority of "
            "paired 30-minute intervals."
        )

    # ── Section 5: Maximum speed distributions ───────────────────────────────
    st.header("5. Distributions of Maximum Speed by Direction")
    st.caption(
        "Maximum speed distributions (independent samples). "
        "Left = SID visible; Right = SID not visible. Black dashed = median; green = 30 mph limit."
    )

    for site in selected_sites:
        label = SITE_LABELS.get(site, site)
        ds = d[d["Location"] == site]
        vis_max  = ds[ds["Direction"] == 1]["Maximum speed"].dropna()
        notv_max = ds[ds["Direction"] == 2]["Maximum speed"].dropna()

        fig5 = make_subplots(rows=1, cols=2, subplot_titles=[
            "Speed visible – approach from front",
            "Speed not visible – approach from rear",
        ])
        fig5.add_trace(go.Histogram(x=vis_max, nbinsx=30, marker_color="orange",
                                    opacity=0.7, showlegend=False), row=1, col=1)
        fig5.add_vline(x=vis_max.median(), line=dict(color="black", dash="dash", width=2),
                       annotation_text=f"Median {vis_max.median():.1f}", row=1, col=1)
        fig5.add_vline(x=SPEED_LIMIT, line=dict(color="green", width=2), row=1, col=1)

        fig5.add_trace(go.Histogram(x=notv_max, nbinsx=30, marker_color=COLOUR_NOT_VISIBLE,
                                    opacity=0.7, showlegend=False), row=1, col=2)
        fig5.add_vline(x=notv_max.median(), line=dict(color="black", dash="dash", width=2),
                       annotation_text=f"Median {notv_max.median():.1f}", row=1, col=2)
        fig5.add_vline(x=SPEED_LIMIT, line=dict(color="green", width=2), row=1, col=2)

        fig5.update_xaxes(title_text="Max speed (mph)")
        fig5.update_yaxes(title_text="Frequency")
        fig5.update_layout(height=350, title_text=label, margin=dict(t=80))
        st.plotly_chart(fig5, use_container_width=True)
        st.caption(
            "Maximum speeds are more dispersed than averages and contain more readings "
            "above the 30 mph limit. Across all sites, 51% of visible-direction readings "
            "have a maximum speed above 30 mph, compared to 68% for not-visible — a "
            "16 percentage point difference. The SID therefore appears to reduce peak "
            "speeding as well as average speeds, with the not-visible (right) distribution "
            "showing a heavier tail above the speed limit."
        )

    # ── Section 6: Maximum speed time series ────────────────────────────────
    st.header("6. Maximum Speed Over Time")
    st.caption(
        "Per site: highest speed recorded by any vehicle on each day. "
        "Line breaks indicate gaps between deployments. Green dashed = 30 mph limit."
    )

    fig6 = make_subplots(
        rows=n_sites, cols=1, shared_xaxes=True,
        subplot_titles=[SITE_LABELS.get(s, s) for s in selected_sites],
        vertical_spacing=0.06,
    )

    for row, site in enumerate(selected_sites, start=1):
        ds = d[d["Location"] == site].copy()
        ds["Day"] = ds["Date"].dt.normalize()
        # Daily maximum across both directions
        daily_max = ds.groupby("Day")["Maximum speed"].max().sort_index()
        # Reindex to a continuous daily grid so deployment gaps become NaN (line breaks)
        idx = pd.date_range(daily_max.index.min(), daily_max.index.max(), freq="D")
        daily_max = daily_max.reindex(idx)
        fig6.add_trace(go.Scatter(
            x=daily_max.index, y=daily_max.values,
            mode="lines",
            line=dict(color=COLOUR_DIFF, width=1.5),
            name="Daily max speed", showlegend=(row == 1),
        ), row=row, col=1)
        fig6.add_hline(
            y=SPEED_LIMIT, line=dict(color="green", dash="dash", width=1),
            row=row, col=1,
        )

    fig6.update_layout(
        height=280 * n_sites,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01),
        margin=dict(t=60),
    )
    fig6.update_yaxes(title_text="Max speed (mph)")
    st.plotly_chart(fig6, use_container_width=True)
    st.caption(
        "Each point is the single highest speed recorded at that site on that day, "
        "across both directions. Line breaks are periods when the SID was not deployed. "
        "Peaks reflect individual fast vehicles; the overall level shows how "
        "frequently high speeds occur at each location."
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Deployment Timeline (moved before Statistical Tests)
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Statistical Tests
# ════════════════════════════════════════════════════════════════════════════
with tab_stats:
    st.header("Statistical Summary")
    st.caption(
        "Results apply to the selected sites and date range. "
        "See the Technical Appendix tab for full methodology and equations."
    )

    st.subheader("Average Speed — Directional Difference (Wilcoxon & Sign Tests)")
    st.caption(
        "Paired comparison: each 30-minute interval where both directions were recorded. "
        "Positive median difference means drivers travel faster when the SID is not visible."
    )
    avg_rows = []
    for site in selected_sites:
        res = paired_stats(d[d["Location"] == site])
        if res:
            avg_rows.append({
                "Site": SITE_LABELS.get(site, site),
                "n (paired intervals)": res["n"],
                "Median difference (mph)": f"{res['median']:+.2f}",
                "Wilcoxon p": fmt_p(res["wilcoxon_p"]),
                "Sign test p": fmt_p(res["sign_p"]),
            })
    if avg_rows:
        st.dataframe(pd.DataFrame(avg_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No paired data available for selected filters.")

    st.subheader("Maximum Speed — Directional Difference (Mann–Whitney U Test)")
    st.caption(
        "Independent samples comparison of maximum speeds by direction. "
        "CLE (Common Language Effect size): probability that a randomly chosen "
        "not-visible reading exceeds a randomly chosen visible reading."
    )
    max_rows = []
    for site in selected_sites:
        res = max_speed_stats(d[d["Location"] == site])
        if res:
            max_rows.append({
                "Site": SITE_LABELS.get(site, site),
                "N (visible)": res["n_vis"],
                "N (not visible)": res["n_not"],
                "Median visible (mph)": f"{res['med_vis']:.0f}",
                "Median not visible (mph)": f"{res['med_not']:.0f}",
                "Difference (mph)": f"{res['diff']:+.0f}",
                "Mann–Whitney p": fmt_p(res["p"]),
                "CLE": f"{res['cle']:.3f}",
            })
    if max_rows:
        st.dataframe(pd.DataFrame(max_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No data available for selected filters.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Deployment Timeline
# ════════════════════════════════════════════════════════════════════════════
with tab_timeline:
    st.header("Deployment Timeline")
    st.caption(
        "Each bar shows a continuous period when a site had active readings, "
        "derived from the verified deployment map. "
        "Where two sites overlap in date, both devices were active simultaneously."
    )

    timeline_df = build_timeline(MASTER_CSV)
    timeline_df["Duration (days)"] = (timeline_df["End"] - timeline_df["Start"]).dt.days

    fig_tl = px.timeline(
        timeline_df,
        x_start="Start",
        x_end="End",
        y="Site Label",
        color="Site Label",
        hover_data={"Duration (days)": True, "Start": "|%d %b %Y", "End": "|%d %b %Y",
                    "Site Label": False},
        labels={"Site Label": "Site"},
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )
    fig_tl.update_yaxes(autorange="reversed", title="")
    fig_tl.update_xaxes(title="Date")
    fig_tl.update_layout(
        height=max(350, 70 * timeline_df["Site Label"].nunique() + 100),
        legend=dict(title="Site", orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=60, l=10),
    )
    st.plotly_chart(fig_tl, use_container_width=True)

    st.subheader("Deployment Blocks — Detail")
    display_df = timeline_df[["Site Label", "Start", "End", "Duration (days)"]].copy()
    display_df["Start"] = display_df["Start"].dt.strftime("%d %b %Y")
    display_df["End"]   = display_df["End"].dt.strftime("%d %b %Y")
    display_df = display_df.rename(columns={"Site Label": "Site"}).reset_index(drop=True)
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Technical Appendix
# ════════════════════════════════════════════════════════════════════════════
with tab_appendix:
    st.header("Technical Appendix")

    # ── Study design ─────────────────────────────────────────────────────────
    st.subheader("Study Design")
    st.markdown("""
A Speed Indicator Device (SID) records a reading every **30 minutes**, logging the
average speed and maximum speed of all vehicles detected, separated by direction of travel.

**Direction 1 — SID visible:** vehicles approaching from the front, able to see the speed
display. These drivers receive real-time feedback on their speed.

**Direction 2 — SID not visible:** vehicles approaching from the rear, unable to see the
display. These readings serve as a control group, capturing background traffic behaviour
uninfluenced by the device.

The primary question is whether the SID causes a measurable reduction in average speed,
evidenced by Direction 1 speeds being lower than Direction 2 speeds at the same site and time.
    """)

    # ── Average speed test ────────────────────────────────────────────────────
    st.subheader("Test 1 — Average Speed Difference (Wilcoxon Signed-Rank Test)")
    st.markdown("""
For each 30-minute interval where both directions were recorded, the paired difference
is computed:
    """)
    st.latex(r"d_i = \bar{v}^{\,\text{not visible}}_i - \bar{v}^{\,\text{visible}}_i")
    st.markdown("""
The null hypothesis is that the population median difference is zero
($H_0: \\widetilde{d} = 0$). The **Wilcoxon signed-rank test** ranks the absolute
differences and tests whether positive and negative ranks are balanced:
    """)
    st.latex(r"W^+ = \sum_{i\,:\,d_i > 0} R_i")
    st.markdown("""
where $R_i$ is the rank of $|d_i|$ among all non-zero differences.
The test statistic $W^+$ is compared against its null distribution. The implementation
uses the *pratt* zero-method (zero differences are included in ranking but excluded from
the test statistic) and a normal approximation (*mode="approx"*).

A continuity check for serial autocorrelation is performed using the **Ljung–Box test**
on the paired difference series. Autocorrelation inflates the effective sample size;
where significant autocorrelation is detected results should be interpreted with caution.
    """)

    st.subheader("Test 1b — Sign Test")
    st.markdown("""
As a robust companion to the Wilcoxon test, the **sign test** considers only the
direction (sign) of each paired difference, discarding magnitude. Under $H_0$ the
number of positive differences $S^+$ follows a binomial distribution:
    """)
    st.latex(r"S^+ \sim \mathrm{Binomial}\!\left(n,\, \tfrac{1}{2}\right)")
    st.markdown("""
where $n$ is the number of non-zero differences. A two-sided p-value is computed.
The sign test is less powerful than the Wilcoxon test but makes no assumption about
the symmetry of the difference distribution.
    """)

    # ── Maximum speed test ────────────────────────────────────────────────────
    st.subheader("Test 2 — Maximum Speed Difference (Mann–Whitney U Test)")
    st.markdown("""
Maximum speed readings from Direction 1 and Direction 2 are **independent** samples
(a maximum speed cannot be meaningfully paired across directions for the same interval).
The **Mann–Whitney U test** is used:
    """)
    st.latex(
        r"U_1 = n_1 n_2 + \frac{n_1(n_1+1)}{2} - R_1"
    )
    st.markdown("""
where $n_1$, $n_2$ are the sample sizes and $R_1$ is the sum of ranks for Direction 1
in the combined ranking. The test statistic $U$ is:
    """)
    st.latex(r"U = \min(U_1,\, U_2)")
    st.markdown(r"""
The **Common Language Effect size (CLE)** — also called the probability of superiority —
estimates the probability that a randomly drawn not-visible maximum speed exceeds a
randomly drawn visible maximum speed:
    """)
    st.latex(
        r"\widehat{\theta} = \frac{U_2}{n_1 n_2}"
    )
    st.markdown(r"""
$\hat{\theta} = 0.5$ indicates no effect; $\hat{\theta} > 0.5$ indicates that
not-visible speeds tend to be higher (consistent with the SID having a slowing effect
when visible).
    """)

    # ── Data provenance ───────────────────────────────────────────────────────
    st.subheader("Data Provenance")
    st.markdown("""
Raw data was exported from SpeedViewer software in semicolon-delimited CSV format.
Device deployment dates were verified against the software's campaign log.

Data covers **6 March – 7 August 2025** (Sites 1–4). Data collected after 7 August 2025
is excluded: post-campaign download files are memory dumps containing the full recording
history of the device, and without confirmed installation/removal dates per site the
readings cannot be reliably assigned to a location.

All statistical tests are two-sided. Significance threshold: *p* < 0.05.
    """)
