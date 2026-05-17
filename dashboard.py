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

st.set_page_config(
    page_title="Soberton SID Analysis",
    page_icon="🚦",
    layout="wide",
)

CAMPAIGN_END = pd.Timestamp("2025-08-07")

MASTER_CSV = "inputs/master.csv"
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

st.info(
    "**Data coverage: 6 March 2025 – 7 August 2025** (Soberton SID campaign period).  \n"
    "Data collected after this date is excluded pending confirmation of device deployment "
    "locations. This dashboard will be updated as further data is verified.",
    icon="ℹ️",
)

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
    d2 = pd.read_json(df_filtered_json)
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

tab_analysis, tab_timeline = st.tabs(["Speed Analysis", "Deployment Timeline"])

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

    # ── Section 6: Statistical tables ────────────────────────────────────────
    st.header("6. Statistical Summary")

    st.subheader("Average Speed Difference (Wilcoxon & Sign Tests)")
    avg_rows = []
    for site in selected_sites:
        res = paired_stats(d[d["Location"] == site])
        if res:
            avg_rows.append({
                "Site": SITE_LABELS.get(site, site),
                "Median diff (mph)": f"{res['median']:.2f}",
                "Wilcoxon p": fmt_p(res["wilcoxon_p"]),
                "Sign test p": fmt_p(res["sign_p"]),
                "n": res["n"],
            })
    if avg_rows:
        st.dataframe(pd.DataFrame(avg_rows), use_container_width=True, hide_index=True)

    st.subheader("Maximum Speed Difference (Mann–Whitney U Test)")
    max_rows = []
    for site in selected_sites:
        res = max_speed_stats(d[d["Location"] == site])
        if res:
            max_rows.append({
                "Site": SITE_LABELS.get(site, site),
                "N visible": res["n_vis"],
                "N not visible": res["n_not"],
                "Median visible (mph)": f"{res['med_vis']:.0f}",
                "Median not visible (mph)": f"{res['med_not']:.0f}",
                "Diff (not–vis)": f"+{res['diff']:.0f}" if res["diff"] >= 0 else f"{res['diff']:.0f}",
                "Mann–Whitney p": fmt_p(res["p"]),
                "CLE": f"{res['cle']:.3f}",
            })
    if max_rows:
        st.dataframe(pd.DataFrame(max_rows), use_container_width=True, hide_index=True)

    # ── Section 7: Data summary ───────────────────────────────────────────────
    st.header("7. Data Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total records", f"{len(d):,}")
    col2.metric("Date range", f"{d['Date'].min().date()} → {d['Date'].max().date()}")
    col3.metric("Sites", len(selected_sites))

    with st.expander("Raw data preview"):
        st.dataframe(d.head(500), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Deployment Timeline
# ════════════════════════════════════════════════════════════════════════════
with tab_timeline:
    st.header("Deployment Timeline")
    st.caption(
        "Each bar shows a continuous period when a site had active readings. "
        "All data within the campaign period is assigned via the verified deployment map. "
        "Overlapping bars across sites indicate two devices active simultaneously."
    )

    timeline_df = build_timeline(data_path)

    SITE_COLOURS = {
        "Site1": "#1f77b4", "Site2": "#ff7f0e", "Site3": "#2ca02c",
        "Site4": "#d62728", "Site5": "#9467bd", "Site6": "#8c564b",
        "Site7": "#e377c2", "Site8": "#7f7f7f", "Site9": "#bcbd22",
    }
    SOURCE_PATTERN = {"Deployment map": "", "Filename": "/"}

    fig_tl = go.Figure()

    for _, row in timeline_df.iterrows():
        site = row["Site"]
        colour = SITE_COLOURS.get(site, "#888888")
        opacity = 0.85 if row["Source"] == "Deployment map" else 0.5
        duration_days = (row["End"] - row["Start"]).days

        fig_tl.add_trace(go.Bar(
            x=[duration_days],
            y=[row["Site Label"]],
            base=[row["Start"].timestamp() * 1000],
            orientation="h",
            marker=dict(
                color=colour,
                opacity=opacity,
                line=dict(color="white", width=1),
            ),
            name=f"{site} ({row['Source']})",
            legendgroup=site,
            showlegend=False,
            hovertemplate=(
                f"<b>{row['Site Label']}</b><br>"
                f"Start: {row['Start'].date()}<br>"
                f"End: {row['End'].date()}<br>"
                f"Duration: {duration_days} days<br>"
                f"Source: {row['Source']}"
                "<extra></extra>"
            ),
        ))

    # Legend entries — one per site
    for site, colour in SITE_COLOURS.items():
        if site in timeline_df["Site"].values:
            fig_tl.add_trace(go.Bar(
                x=[0], y=[""], base=[0],
                orientation="h",
                marker=dict(color=colour, opacity=0.7),
                name=SITE_LABELS.get(site, site),
                legendgroup=site,
                showlegend=True,
            ))

    # Campaign end marker
    fig_tl.add_vline(
        x=CAMPAIGN_END.timestamp() * 1000,
        line=dict(color="black", dash="dash", width=1.5),
        annotation_text="Campaign period ends",
        annotation_position="top right",
        annotation_font_size=11,
    )

    # Convert x-axis to date display
    all_starts = timeline_df["Start"].tolist()
    all_ends   = timeline_df["End"].tolist()
    x_min = min(all_starts)
    x_max = max(all_ends)

    tick_dates = pd.date_range(
        x_min.replace(day=1),
        (x_max + pd.offsets.MonthBegin(1)).replace(day=1),
        freq="MS",
    )
    fig_tl.update_layout(
        barmode="overlay",
        height=max(400, 60 * len(timeline_df["Site Label"].unique()) + 120),
        xaxis=dict(
            tickmode="array",
            tickvals=[d.timestamp() * 1000 for d in tick_dates],
            ticktext=[d.strftime("%b %Y") for d in tick_dates],
            tickangle=45,
            title="Date",
            range=[x_min.timestamp() * 1000, x_max.timestamp() * 1000],
        ),
        yaxis=dict(title="Site", autorange="reversed"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, title="Site"),
        margin=dict(t=80, l=220),
    )

    st.plotly_chart(fig_tl, use_container_width=True)

    # ── Audit table ───────────────────────────────────────────────────────────
    st.subheader("Deployment Blocks — Detail Table")
    st.caption(
        "Each row is one continuous deployment period. "
        "Where two sites overlap in date, both devices were active simultaneously."
    )

    display_df = timeline_df[["Site Label", "Start", "End", "Source"]].copy()
    display_df["Start"] = display_df["Start"].dt.date
    display_df["End"]   = display_df["End"].dt.date
    display_df["Duration (days)"] = (
        pd.to_datetime(display_df["End"]) - pd.to_datetime(display_df["Start"])
    ).dt.days
    display_df = display_df.rename(columns={"Site Label": "Site"})
    display_df = display_df.sort_values("Start").reset_index(drop=True)

    st.dataframe(display_df, use_container_width=True, hide_index=True)
