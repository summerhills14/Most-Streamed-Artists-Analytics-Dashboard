"""
streamlit_testing.py
------------------------------------------------------------
Spotify Streamed-Artists Analytics Dashboard
A professional, dynamic Streamlit dashboard built on top of the
"Most Streamed Artists on Spotify" dataset (expects a `dataset/`
folder next to this script containing either the extracted CSV(s)
or the original `archive.zip`).

Features
  - Left sidebar "control center": appearance, one-tap filter presets,
    filters, top-N control, color palette, reset, CSV download
  - Polished KPI / stat cards with a dark Spotify-inspired theme,
    animated gradient header, hover-lift cards, glowing fun-fact badges
  - Multi-tab layout: Overview | Genre & Country | Deep Dive & Correlation
    | Battle Mode | Data Explorer
  - Advanced Plotly charts: dual-axis line+bar, treemap, sunburst, box,
    scatter/bubble, correlation heatmap, radar, gauge indicator, and an
    animated bar-chart "genre race" by debut year
  - Chart-to-chart dependency: selecting genres on the treemap
    cross-filters the box plot & country bar chart; box/lasso-selecting
    points on the scatter cross-filters the bubble chart & heatmap
  - Artist Spotlight card with a "🎲 Discover random artist" roulette
  - ⚔️ Battle Mode: head-to-head artist comparison with a radar overlay,
    per-metric scoreboard, verdict, and a confetti celebration

Run with:
    streamlit run streamlit_testing.py
------------------------------------------------------------
"""

import glob
import io
import os
import random
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ==================================================================
# PAGE CONFIG
# ==================================================================
st.set_page_config(
    page_title="Spotify Artists Analytics",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

NUMERIC_COLS = [
    "Debut Year",
    "Total Streams (in millions)",
    "Lead Streams (in millions)",
    "Feature Streams (in millions)",
    "Solo Streams (in millions)",
    "% of Solo Streams",
    "Collaborative Streams (in millions)",
    "% of Collaborative Streams",
]

PALETTES = {
    "Spotify Green": px.colors.sequential.Greens_r,
    "Sunset": px.colors.sequential.Sunsetdark,
    "Ocean": px.colors.sequential.Teal,
    "Purple Rain": px.colors.sequential.Purp,
    "Vivid (categorical)": px.colors.qualitative.Vivid,
}
QUALITATIVE_PALETTES = {
    "Spotify Green": px.colors.qualitative.Prism,
    "Sunset": px.colors.qualitative.Bold,
    "Ocean": px.colors.qualitative.Antique,
    "Purple Rain": px.colors.qualitative.Safe,
    "Vivid (categorical)": px.colors.qualitative.Vivid,
}


# ==================================================================
# DATA LOADING
# ==================================================================
@st.cache_data(show_spinner=True)
def load_dataset() -> tuple[pd.DataFrame, str]:
    """
    Look for the dataset in ./dataset (next to this script, or in the
    current working directory). Supports either extracted CSVs or the
    original archive.zip. Falls back to None so the UI can offer an
    uploader instead of crashing.
    """
    search_dirs = []
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        search_dirs.append(os.path.join(script_dir, "dataset"))
    except NameError:
        pass
    search_dirs.append(os.path.join(os.getcwd(), "dataset"))

    for d in search_dirs:
        if not os.path.isdir(d):
            continue

        # 1) prefer an already-extracted CSV, favouring the richer "V1.1" file
        csvs = sorted(glob.glob(os.path.join(d, "*.csv")))
        if csvs:
            best = next((c for c in csvs if "V1.1" in os.path.basename(c)), csvs[0])
            df = pd.read_csv(best)
            df.columns = [c.strip() for c in df.columns]
            return df, os.path.basename(best)

        # 2) otherwise look inside archive.zip
        zips = sorted(glob.glob(os.path.join(d, "*.zip")))
        for z in zips:
            with zipfile.ZipFile(z) as zf:
                names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if not names:
                    continue
                best = next((n for n in names if "V1.1" in n), names[0])
                with zf.open(best) as f:
                    df = pd.read_csv(io.BytesIO(f.read()))
                df.columns = [c.strip() for c in df.columns]
                return df, os.path.basename(best)

    return pd.DataFrame(), ""


def human_number(value_millions: float) -> str:
    """Format a value already expressed in millions into a readable string."""
    if pd.isna(value_millions):
        return "—"
    if value_millions >= 1_000_000:
        return f"{value_millions / 1_000_000:,.1f}T"
    if value_millions >= 1_000:
        return f"{value_millions / 1_000:,.1f}B"
    return f"{value_millions:,.1f}M"


# ==================================================================
# STYLING
# ==================================================================
def inject_css(dark: bool):
    if dark:
        bg, panel, text, subtext, border = "#0e1117", "#161b22", "#f5f5f7", "#9aa0a6", "#262b33"
        accent = "#1DB954"
    else:
        bg, panel, text, subtext, border = "#f7f7f9", "#ffffff", "#111318", "#5f6570", "#e6e6eb"
        accent = "#1DB954"

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

        .stApp {{ background-color: {bg}; color: {text}; }}

        section[data-testid="stSidebar"] {{
            background-color: {panel};
            border-right: 1px solid {border};
        }}

        .kpi-card {{
            background: linear-gradient(160deg, {panel} 0%, {panel} 100%);
            border: 1px solid {border};
            border-radius: 14px;
            padding: 16px 18px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.15);
            height: 100%;
            transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
        }}
        .kpi-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 8px 22px rgba(29,185,84,0.25);
            border-color: {accent}88;
        }}

        .glow-header {{
            font-weight: 800;
            background: linear-gradient(90deg, {accent}, #58e88a, {accent});
            background-size: 200% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: shine 5s linear infinite;
            display: inline-block;
        }}
        @keyframes shine {{
            to {{ background-position: 200% center; }}
        }}
        .header-underline {{
            height: 3px;
            width: 100%;
            margin: 4px 0 14px 0;
            border-radius: 3px;
            background: linear-gradient(90deg, {accent}, #58e88a, transparent, {accent});
            background-size: 300% auto;
            animation: shine 4s linear infinite;
        }}

        .fact-badge {{
            animation: pulseGlow 2.6s ease-in-out infinite;
        }}
        @keyframes pulseGlow {{
            0%, 100% {{ box-shadow: 0 0 0 rgba(29,185,84,0); }}
            50% {{ box-shadow: 0 0 10px {accent}66; }}
        }}

        div.stButton > button {{
            border-radius: 999px;
            transition: transform 0.12s ease;
        }}
        div.stButton > button:hover {{
            transform: scale(1.03);
            border-color: {accent};
            color: {accent};
        }}
        .kpi-label {{
            font-size: 0.78rem;
            font-weight: 600;
            color: {subtext};
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 6px;
        }}
        .kpi-value {{
            font-size: 1.55rem;
            font-weight: 800;
            color: {text};
            line-height: 1.1;
        }}
        .kpi-sub {{
            font-size: 0.78rem;
            color: {accent};
            font-weight: 600;
            margin-top: 4px;
        }}

        .section-title {{
            font-size: 1.05rem;
            font-weight: 700;
            color: {text};
            margin: 6px 0 2px 0;
            border-left: 4px solid {accent};
            padding-left: 10px;
        }}
        .section-sub {{
            color: {subtext};
            font-size: 0.85rem;
            margin-bottom: 10px;
            padding-left: 14px;
        }}

        .spotlight-card {{
            background: linear-gradient(135deg, {accent}22 0%, {panel} 60%);
            border: 1px solid {accent}55;
            border-radius: 16px;
            padding: 18px 22px;
        }}
        .spotlight-name {{
            font-size: 1.4rem;
            font-weight: 800;
            color: {text};
        }}
        .spotlight-meta {{
            color: {subtext};
            font-size: 0.85rem;
            margin-bottom: 8px;
        }}
        .badge {{
            display: inline-block;
            background: {accent}33;
            color: {accent};
            font-weight: 700;
            font-size: 0.72rem;
            padding: 3px 10px;
            border-radius: 999px;
            margin-right: 6px;
        }}

        div[data-testid="stMetric"] {{
            background-color: {panel};
            border: 1px solid {border};
            border-radius: 12px;
            padding: 10px 14px;
        }}

        .stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
        .stTabs [data-baseweb="tab"] {{
            background-color: {panel};
            border-radius: 10px 10px 0 0;
            padding: 8px 16px;
            border: 1px solid {border};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(col, label, value, sub=""):
    col.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==================================================================
# LOAD DATA
# ==================================================================
raw_df, source_name = load_dataset()

if raw_df.empty:
    st.warning(
        "Couldn't find the dataset in a `dataset/` folder next to this script "
        "(looked for a `.csv` or `archive.zip`). Upload the CSV manually to continue."
    )
    uploaded = st.file_uploader("Upload the artists CSV", type="csv")
    if uploaded is None:
        st.stop()
    raw_df = pd.read_csv(uploaded)
    raw_df.columns = [c.strip() for c in raw_df.columns]
    source_name = uploaded.name

has_debut_year = "Debut Year" in raw_df.columns

# ==================================================================
# SIDEBAR — CONTROL CENTER
# ==================================================================
with st.sidebar:
    st.markdown("## 🎧 Spotify Analytics")
    st.caption(f"Source: `{source_name}` · {len(raw_df):,} artists")
    st.divider()

    dark_mode = st.toggle("🌙 Dark mode", value=True)
    palette_name = st.selectbox("🎨 Chart color theme", list(PALETTES.keys()), index=0)

    sex_opts = sorted(raw_df["Sex"].dropna().unique()) if "Sex" in raw_df else []
    type_opts = sorted(raw_df["Artist Type"].dropna().unique()) if "Artist Type" in raw_df else []
    country_opts = sorted(raw_df["Country of Origin"].dropna().unique()) if "Country of Origin" in raw_df else []
    genre_opts = sorted(raw_df["Primary Genre"].dropna().unique()) if "Primary Genre" in raw_df else []
    lang_opts = sorted(raw_df["Primary Language"].dropna().unique()) if "Primary Language" in raw_df else []

    st.divider()
    st.markdown("### ⚡ Quick Presets")
    st.caption("One-tap filter combos")

    def _make_preset(genres=None, types=None):
        def _apply():
            st.session_state.sel_genre = [g for g in (genres or []) if g in genre_opts]
            st.session_state.sel_type = [t for t in (types or []) if t in type_opts]
            st.session_state.sel_sex, st.session_state.sel_country, st.session_state.sel_lang = [], [], []
        return _apply

    presets = [
        ("🌟 Reset", _make_preset()),
        ("🎤 Solo Icons", _make_preset(types=["Solo"])),
        ("🤝 Groups", _make_preset(types=["Group"])),
        ("🔥 Hip-Hop", _make_preset(genres=["Hip-Hop"])),
        ("🎸 Rock", _make_preset(genres=["Rock"])),
        ("🌏 K-Pop", _make_preset(genres=["K-Pop"])),
    ]
    pcols = st.columns(2)
    for i, (label, fn) in enumerate(presets):
        pcols[i % 2].button(label, on_click=fn, use_container_width=True, key=f"preset_{i}")

    st.divider()
    st.markdown("### 🔎 Filters")

    sel_sex = st.multiselect("Sex", sex_opts, key="sel_sex")
    sel_type = st.multiselect("Artist Type", type_opts, key="sel_type")
    sel_genre = st.multiselect("Primary Genre", genre_opts, key="sel_genre")
    sel_country = st.multiselect("Country of Origin", country_opts, key="sel_country")
    sel_lang = st.multiselect("Primary Language", lang_opts, key="sel_lang")

    if has_debut_year:
        y_min, y_max = int(raw_df["Debut Year"].min()), int(raw_df["Debut Year"].max())
        debut_range = st.slider("Debut Year", y_min, y_max, (y_min, y_max))
    else:
        debut_range = None

    s_min, s_max = float(raw_df["Total Streams (in millions)"].min()), float(raw_df["Total Streams (in millions)"].max())
    stream_range = st.slider(
        "Total Streams (millions)",
        min_value=float(np.floor(s_min)),
        max_value=float(np.ceil(s_max)),
        value=(float(np.floor(s_min)), float(np.ceil(s_max))),
    )

    st.divider()
    st.markdown("### ⚙️ Chart Controls")
    top_n = st.slider("Top-N artists / categories", 5, 30, 15)

    st.divider()
    if st.button("♻️ Reset all filters", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

inject_css(dark_mode)
plotly_template = "plotly_dark" if dark_mode else "plotly_white"
color_seq = PALETTES[palette_name]
color_seq_qual = QUALITATIVE_PALETTES[palette_name]
px.defaults.template = plotly_template
px.defaults.color_discrete_sequence = color_seq_qual
px.defaults.color_continuous_scale = color_seq

# ==================================================================
# APPLY FILTERS
# ==================================================================
df = raw_df.copy()
if sel_sex:
    df = df[df["Sex"].isin(sel_sex)]
if sel_type:
    df = df[df["Artist Type"].isin(sel_type)]
if sel_genre:
    df = df[df["Primary Genre"].isin(sel_genre)]
if sel_country:
    df = df[df["Country of Origin"].isin(sel_country)]
if sel_lang:
    df = df[df["Primary Language"].isin(sel_lang)]
if debut_range:
    df = df[(df["Debut Year"] >= debut_range[0]) & (df["Debut Year"] <= debut_range[1])]
df = df[
    (df["Total Streams (in millions)"] >= stream_range[0])
    & (df["Total Streams (in millions)"] <= stream_range[1])
]

with st.sidebar:
    st.caption(f"Showing **{len(df):,} / {len(raw_df):,}** artists after filters")
    st.download_button(
        "⬇️ Download filtered CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_artists.csv",
        mime="text/csv",
        use_container_width=True,
    )

if df.empty:
    st.warning("No artists match the current filters — loosen them in the sidebar.")
    st.stop()

# ==================================================================
# HEADER
# ==================================================================
st.markdown(
    "<h1 style='margin-bottom:0'>🎧 <span class='glow-header'>Most Streamed Artists</span> — Analytics Dashboard</h1>"
    "<div class='header-underline'></div>",
    unsafe_allow_html=True,
)
st.caption("Advanced, dynamic exploration of Spotify's top artists · dataset last updated 17 Jul 2026")

# ------------------------------------------------------------------
# KPI CARDS
# ------------------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)
kpi_card(k1, "Total Artists", f"{len(df):,}", f"of {len(raw_df):,} total")
kpi_card(k2, "Total Streams", human_number(df["Total Streams (in millions)"].sum()), "combined, all artists")
kpi_card(k3, "Avg Solo Share", f"{df['% of Solo Streams'].mean():.1f}%", "of total streams")
top_genre = df["Primary Genre"].mode().iloc[0] if not df["Primary Genre"].mode().empty else "—"
kpi_card(k4, "Top Genre", top_genre, f"{(df['Primary Genre'] == top_genre).sum()} artists")
top_country = df["Country of Origin"].mode().iloc[0] if not df["Country of Origin"].mode().empty else "—"
kpi_card(k5, "Top Country", top_country, f"{(df['Country of Origin'] == top_country).sum()} artists")

st.write("")

# ------------------------------------------------------------------
# FUN-FACT BADGES
# ------------------------------------------------------------------
newest = df.loc[df["Debut Year"].idxmax(), "Artist Name"] if has_debut_year else "—"
oldest_year = int(df["Debut Year"].min()) if has_debut_year else "—"
male_ct, female_ct = (df["Sex"] == "Male").sum(), (df["Sex"] == "Female").sum()
st.markdown(
    f"""
    <div>
        <span class="badge fact-badge">🌍 {df['Country of Origin'].nunique()} countries</span>
        <span class="badge fact-badge">🎼 {df['Primary Genre'].nunique()} genres</span>
        <span class="badge fact-badge">🕰️ Oldest debut: {oldest_year}</span>
        <span class="badge fact-badge">✨ Newest star: {newest}</span>
        <span class="badge fact-badge">⚖️ {male_ct} Male / {female_ct} Female</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

# ------------------------------------------------------------------
# ARTIST SPOTLIGHT (with a "Discover random artist" roulette)
# ------------------------------------------------------------------
if "spotlight_pick" not in st.session_state or st.session_state.spotlight_pick not in df["Artist Name"].values:
    st.session_state.spotlight_pick = sorted(df["Artist Name"].unique())[0]

sp_col1, sp_col2 = st.columns([1, 3])
with sp_col1:
    if st.button("🎲 Discover random artist", use_container_width=True):
        st.session_state.spotlight_pick = random.choice(df["Artist Name"].tolist())
        st.toast(f"🎲 Landed on {st.session_state.spotlight_pick}!")
        if st.session_state.spotlight_pick in df.nlargest(10, "Total Streams (in millions)")["Artist Name"].values:
            st.balloons()
    spotlight_artist = st.selectbox(
        "🔦 Artist Spotlight", sorted(df["Artist Name"].unique()), key="spotlight_pick"
    )
with sp_col2:
    a = df[df["Artist Name"] == spotlight_artist].iloc[0]
    st.markdown(
        f"""
        <div class="spotlight-card">
            <div class="spotlight-name">{a['Artist Name']}</div>
            <div class="spotlight-meta">
                <span class="badge">{a['Artist Type']}</span>
                <span class="badge">{a['Primary Genre']}</span>
                <span class="badge">{a['Country of Origin']}</span>
                {"<span class='badge'>Debut " + str(int(a['Debut Year'])) + "</span>" if has_debut_year else ""}
            </div>
            Total streams: <b>{human_number(a['Total Streams (in millions)'])}</b> &nbsp;|&nbsp;
            Solo: <b>{human_number(a['Solo Streams (in millions)'])}</b> ({a['% of Solo Streams']:.1f}%) &nbsp;|&nbsp;
            Collaborative: <b>{human_number(a['Collaborative Streams (in millions)'])}</b> ({a['% of Collaborative Streams']:.1f}%)
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# ==================================================================
# TABS
# ==================================================================
tab_overview, tab_genre, tab_deep, tab_battle, tab_data = st.tabs(
    ["📊 Overview", "🌍 Genre & Country", "🔬 Deep Dive & Correlation", "⚔️ Battle Mode", "📋 Data Explorer"]
)

# ------------------------------------------------------------------
# TAB 1 — OVERVIEW
# ------------------------------------------------------------------
with tab_overview:
    st.markdown('<div class="section-title">Top Artists by Total Streams</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-sub">Top {top_n} artists in the current filter selection</div>',
        unsafe_allow_html=True,
    )

    top_artists = df.nlargest(top_n, "Total Streams (in millions)").sort_values("Total Streams (in millions)")
    fig_top = px.bar(
        top_artists,
        x="Total Streams (in millions)",
        y="Artist Name",
        orientation="h",
        color="Primary Genre",
        hover_data=["Artist Type", "Country of Origin"],
        text="Total Streams (in millions)",
    )
    fig_top.update_traces(texttemplate="%{text:,.0f}M", textposition="outside")
    fig_top.update_layout(height=max(420, 22 * top_n), yaxis_title="", showlegend=True, legend_title="Genre")
    st.plotly_chart(fig_top, use_container_width=True)

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.markdown('<div class="section-title">Artist Type Split</div>', unsafe_allow_html=True)
        donut_dim = st.radio("Breakdown by", ["Artist Type", "Sex"], horizontal=True, key="donut_dim")
        pie_data = df[donut_dim].value_counts().reset_index()
        pie_data.columns = [donut_dim, "Count"]
        fig_donut = px.pie(pie_data, names=donut_dim, values="Count", hole=0.55)
        fig_donut.update_traces(textinfo="percent+label")
        fig_donut.update_layout(height=380)
        st.plotly_chart(fig_donut, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Streaming Pulse</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">Filtered avg total streams vs. the full dataset avg</div>',
            unsafe_allow_html=True,
        )
        gauge_max = float(raw_df["Total Streams (in millions)"].max())
        filtered_avg = float(df["Total Streams (in millions)"].mean())
        overall_avg = float(raw_df["Total Streams (in millions)"].mean())
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=filtered_avg,
                number={"suffix": "M"},
                delta={"reference": overall_avg, "relative": False},
                gauge={
                    "axis": {"range": [0, gauge_max]},
                    "bar": {"color": "#1DB954"},
                    "steps": [
                        {"range": [0, gauge_max / 3], "color": "rgba(29,185,84,0.12)"},
                        {"range": [gauge_max / 3, 2 * gauge_max / 3], "color": "rgba(29,185,84,0.25)"},
                        {"range": [2 * gauge_max / 3, gauge_max], "color": "rgba(29,185,84,0.4)"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.8,
                        "value": overall_avg,
                    },
                },
            )
        )
        fig_gauge.update_layout(height=380, template=plotly_template, margin=dict(t=30, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with c3:
        st.markdown('<div class="section-title">Debut Year Trend</div>', unsafe_allow_html=True)
        if has_debut_year:
            trend = (
                df.groupby("Debut Year")
                .agg(Artists=("Artist Name", "count"), Streams=("Total Streams (in millions)", "sum"))
                .reset_index()
                .sort_values("Debut Year")
            )
            fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
            fig_trend.add_trace(
                go.Bar(x=trend["Debut Year"], y=trend["Artists"], name="Artists debuting", opacity=0.55),
                secondary_y=False,
            )
            fig_trend.add_trace(
                go.Scatter(
                    x=trend["Debut Year"], y=trend["Streams"], name="Total streams (M)",
                    mode="lines+markers", line=dict(width=3),
                ),
                secondary_y=True,
            )
            fig_trend.update_layout(
                height=380, template=plotly_template, legend=dict(orientation="h", y=1.12)
            )
            fig_trend.update_yaxes(title_text="Artists", secondary_y=False)
            fig_trend.update_yaxes(title_text="Streams (M)", secondary_y=True)
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("No debut-year column available in this dataset.")

    st.write("")
    st.markdown('<div class="section-title">🎬 Genre Race — Cumulative Streams by Debut Year</div>', unsafe_allow_html=True)
    show_race = st.checkbox(
        "▶️ Build & play the animated race chart", value=False, key="show_race",
        help="Renders an animated bar-chart race — hit the ▶ Play button on the chart once it loads.",
    )
    if show_race and has_debut_year:
        with st.spinner("Building animation frames..."):
            top_genres_race = (
                df.groupby("Primary Genre")["Total Streams (in millions)"].sum().nlargest(8).index.tolist()
            )
            years_sorted = sorted(df["Debut Year"].unique())
            frames = []
            for y in years_sorted:
                cum = (
                    df[df["Debut Year"] <= y]
                    .groupby("Primary Genre")["Total Streams (in millions)"]
                    .sum()
                    .reindex(top_genres_race, fill_value=0)
                    .reset_index()
                )
                cum["Debut Year"] = y
                frames.append(cum)
            race_df = pd.concat(frames, ignore_index=True)

        fig_race = px.bar(
            race_df, x="Total Streams (in millions)", y="Primary Genre", color="Primary Genre",
            orientation="h", animation_frame="Debut Year",
            range_x=[0, race_df["Total Streams (in millions)"].max() * 1.05],
        )
        fig_race.update_layout(
            height=520, showlegend=False, template=plotly_template, yaxis_title="",
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig_race, use_container_width=True)
    elif show_race:
        st.info("No debut-year column available to animate.")

# ------------------------------------------------------------------
# TAB 2 — GENRE & COUNTRY (with cross-filtering)
# ------------------------------------------------------------------
with tab_genre:
    st.markdown('<div class="section-title">Streams by Genre (Treemap)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Click a genre block to cross-filter the charts below</div>',
        unsafe_allow_html=True,
    )

    genre_agg = df.groupby("Primary Genre", as_index=False)["Total Streams (in millions)"].sum()
    fig_tree = px.treemap(
        genre_agg, path=["Primary Genre"], values="Total Streams (in millions)",
        color="Total Streams (in millions)",
    )
    fig_tree.update_layout(height=420, margin=dict(t=10, l=0, r=0, b=0))
    tree_event = st.plotly_chart(
        fig_tree, use_container_width=True, key="genre_treemap",
        on_select="rerun", selection_mode="points",
    )

    selected_genres = []
    if tree_event and tree_event.get("selection", {}).get("points"):
        selected_genres = sorted(
            {p.get("label") for p in tree_event["selection"]["points"] if p.get("label")}
        )
    df_genre_cross = df[df["Primary Genre"].isin(selected_genres)] if selected_genres else df

    if selected_genres:
        st.info(f"Cross-filtered to genre(s): **{', '.join(selected_genres)}** ({len(df_genre_cross)} artists)")

    g1, g2 = st.columns(2)
    with g1:
        st.markdown('<div class="section-title">Stream Distribution by Genre</div>', unsafe_allow_html=True)
        fig_box = px.box(
            df_genre_cross, x="Primary Genre", y="Total Streams (in millions)", color="Primary Genre",
        )
        fig_box.update_layout(height=400, showlegend=False, xaxis_title="", xaxis_tickangle=-35)
        st.plotly_chart(fig_box, use_container_width=True)

    with g2:
        st.markdown('<div class="section-title">Top Countries by Avg Streams</div>', unsafe_allow_html=True)
        country_agg = (
            df_genre_cross.groupby("Country of Origin")["Total Streams (in millions)"]
            .mean()
            .nlargest(min(top_n, 15))
            .sort_values()
            .reset_index()
        )
        fig_country = px.bar(
            country_agg, x="Total Streams (in millions)", y="Country of Origin", orientation="h",
            color="Total Streams (in millions)",
        )
        fig_country.update_layout(height=400, yaxis_title="", coloraxis_showscale=False)
        st.plotly_chart(fig_country, use_container_width=True)

    st.markdown('<div class="section-title">Country → Genre Sunburst</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Top 10 countries by total streams, broken down by genre</div>',
        unsafe_allow_html=True,
    )
    top_countries = (
        df.groupby("Country of Origin")["Total Streams (in millions)"].sum().nlargest(10).index
    )
    sun_df = df[df["Country of Origin"].isin(top_countries)]
    fig_sun = px.sunburst(
        sun_df, path=["Country of Origin", "Primary Genre"], values="Total Streams (in millions)",
    )
    fig_sun.update_layout(height=520, margin=dict(t=10, l=0, r=0, b=0))
    st.plotly_chart(fig_sun, use_container_width=True)

# ------------------------------------------------------------------
# TAB 3 — DEEP DIVE & CORRELATION (with cross-filtering)
# ------------------------------------------------------------------
with tab_deep:
    st.markdown('<div class="section-title">Solo vs Collaborative Streams</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Box-select or lasso-select points to cross-filter the bubble chart & heatmap below</div>',
        unsafe_allow_html=True,
    )

    fig_scatter = px.scatter(
        df, x="Solo Streams (in millions)", y="Collaborative Streams (in millions)",
        color="Primary Genre", size="Total Streams (in millions)",
        hover_name="Artist Name", size_max=40,
    )
    fig_scatter.update_layout(height=460)
    scatter_event = st.plotly_chart(
        fig_scatter, use_container_width=True, key="solo_collab_scatter",
        on_select="rerun", selection_mode=["points", "box", "lasso"],
    )

    selected_artists = []
    if scatter_event and scatter_event.get("selection", {}).get("points"):
        idxs = [p.get("point_index") for p in scatter_event["selection"]["points"] if p.get("point_index") is not None]
        if idxs:
            selected_artists = df.reset_index(drop=True).loc[idxs, "Artist Name"].tolist()
    df_deep_cross = df[df["Artist Name"].isin(selected_artists)] if selected_artists else df

    if selected_artists:
        st.info(f"Cross-filtered to **{len(selected_artists)}** selected artist(s) from the scatter plot.")

    d1, d2 = st.columns(2)
    with d1:
        st.markdown('<div class="section-title">Debut Year vs Total Streams (Bubble)</div>', unsafe_allow_html=True)
        if has_debut_year:
            fig_bubble = px.scatter(
                df_deep_cross, x="Debut Year", y="Total Streams (in millions)",
                size="% of Solo Streams", color="Artist Type",
                hover_name="Artist Name", size_max=35,
            )
            fig_bubble.update_layout(height=420)
            st.plotly_chart(fig_bubble, use_container_width=True)
        else:
            st.info("No debut-year column available.")

    with d2:
        st.markdown('<div class="section-title">Correlation Heatmap</div>', unsafe_allow_html=True)
        corr_cols = [c for c in NUMERIC_COLS if c in df_deep_cross.columns]
        corr = df_deep_cross[corr_cols].corr().round(2)
        fig_heat = px.imshow(
            corr, text_auto=True, aspect="auto",
            color_continuous_scale=PALETTES[palette_name],
        )
        fig_heat.update_layout(height=420)
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown('<div class="section-title">Artist Comparison (Radar)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Pick up to 5 artists — metrics are min-max scaled within the filtered data for comparability</div>',
        unsafe_allow_html=True,
    )
    radar_metrics = [
        "Total Streams (in millions)", "Lead Streams (in millions)", "Feature Streams (in millions)",
        "Solo Streams (in millions)", "Collaborative Streams (in millions)",
    ]
    default_radar = df.nlargest(3, "Total Streams (in millions)")["Artist Name"].tolist()
    radar_artists = st.multiselect(
        "Select artists to compare", sorted(df["Artist Name"].unique()),
        default=default_radar, max_selections=5,
    )
    if radar_artists:
        scale_base = df[radar_metrics]
        mins, maxs = scale_base.min(), scale_base.max()
        fig_radar = go.Figure()
        for name in radar_artists:
            row = df[df["Artist Name"] == name][radar_metrics].iloc[0]
            scaled = ((row - mins) / (maxs - mins).replace(0, 1)).clip(0, 1)
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=scaled.values.tolist() + [scaled.values[0]],
                    theta=radar_metrics + [radar_metrics[0]],
                    fill="toself", name=name,
                )
            )
        fig_radar.update_layout(
            height=480, template=plotly_template,
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            legend=dict(orientation="h", y=-0.1),
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info("Select at least one artist above to draw the radar chart.")

# ------------------------------------------------------------------
# TAB 4 — BATTLE MODE (head-to-head artist comparison, gamified)
# ------------------------------------------------------------------
with tab_battle:
    st.markdown('<div class="section-title">⚔️ Artist Battle Mode</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Pick two artists from the filtered pool and see who wins across each metric</div>',
        unsafe_allow_html=True,
    )

    names_sorted = sorted(df["Artist Name"].unique())
    bcol1, bcol_vs, bcol2 = st.columns([2, 1, 2])
    with bcol1:
        artist_a = st.selectbox("Fighter 1", names_sorted, index=0, key="battle_a")
    with bcol_vs:
        st.markdown("<h2 style='text-align:center;margin-top:28px;'>⚡ VS ⚡</h2>", unsafe_allow_html=True)
    with bcol2:
        default_idx = 1 if len(names_sorted) > 1 else 0
        artist_b = st.selectbox("Fighter 2", names_sorted, index=default_idx, key="battle_b")

    if artist_a == artist_b:
        st.info("👆 Pick two different artists to start the battle.")
    else:
        row_a = df[df["Artist Name"] == artist_a].iloc[0]
        row_b = df[df["Artist Name"] == artist_b].iloc[0]
        battle_metrics = [
            "Total Streams (in millions)", "Lead Streams (in millions)", "Feature Streams (in millions)",
            "Solo Streams (in millions)", "Collaborative Streams (in millions)", "% of Solo Streams",
        ]
        wins_a = wins_b = 0
        score_rows = []
        for m in battle_metrics:
            va, vb = float(row_a[m]), float(row_b[m])
            if va > vb:
                wins_a += 1
                winner = f"🏅 {artist_a}"
            elif vb > va:
                wins_b += 1
                winner = f"🏅 {artist_b}"
            else:
                winner = "🤝 Tie"
            score_rows.append({m: round(va, 1), "vs": "→", artist_b: round(vb, 1), "Winner": winner})

        sc1, sc2, sc3 = st.columns(3)
        kpi_card(sc1, artist_a, str(wins_a), "categories won")
        kpi_card(sc2, "SCORE", f"{wins_a} — {wins_b}", "head-to-head")
        kpi_card(sc3, artist_b, str(wins_b), "categories won")
        st.write("")

        radar_metrics2 = [
            "Total Streams (in millions)", "Lead Streams (in millions)", "Feature Streams (in millions)",
            "Solo Streams (in millions)", "Collaborative Streams (in millions)",
        ]
        scale_base2 = df[radar_metrics2]
        mins2, maxs2 = scale_base2.min(), scale_base2.max()
        fig_battle_radar = go.Figure()
        for name, row in [(artist_a, row_a), (artist_b, row_b)]:
            scaled = ((row[radar_metrics2] - mins2) / (maxs2 - mins2).replace(0, 1)).clip(0, 1)
            fig_battle_radar.add_trace(
                go.Scatterpolar(
                    r=scaled.values.tolist() + [scaled.values[0]],
                    theta=radar_metrics2 + [radar_metrics2[0]],
                    fill="toself", name=name,
                )
            )
        fig_battle_radar.update_layout(
            height=440, template=plotly_template,
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            legend=dict(orientation="h", y=-0.1),
        )
        st.plotly_chart(fig_battle_radar, use_container_width=True)

        st.markdown('<div class="section-title">Head-to-Head Scoreboard</div>', unsafe_allow_html=True)
        score_table = pd.DataFrame(
            [
                {"Metric": m, artist_a: round(float(row_a[m]), 1), artist_b: round(float(row_b[m]), 1), "Winner": w}
                for m, w in zip(battle_metrics, [r["Winner"] for r in score_rows])
            ]
        )
        st.dataframe(score_table, use_container_width=True, hide_index=True)

        if wins_a > wins_b:
            st.success(f"🏆 **{artist_a}** wins the battle, {wins_a}–{wins_b}!")
        elif wins_b > wins_a:
            st.success(f"🏆 **{artist_b}** wins the battle, {wins_b}–{wins_a}!")
        else:
            st.info("🤝 It's a dead heat — both artists are evenly matched!")

        if st.button("🎉 Celebrate the winner"):
            st.balloons()

# ------------------------------------------------------------------
# TAB 5 — DATA EXPLORER
# ------------------------------------------------------------------
with tab_data:
    st.markdown('<div class="section-title">Full Filtered Dataset</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-sub">{len(df):,} rows · sortable, searchable — click a column header to sort</div>',
        unsafe_allow_html=True,
    )
    search = st.text_input("🔍 Search artist name")
    show_df = df.copy()
    if search.strip():
        show_df = show_df[show_df["Artist Name"].str.contains(search.strip(), case=False, na=False)]

    all_cols = list(df.columns)
    show_cols = st.multiselect("Columns to display", all_cols, default=all_cols)
    st.dataframe(
        show_df[show_cols].sort_values("Total Streams (in millions)", ascending=False),
        use_container_width=True,
        height=560,
    )

st.divider()
st.caption(
    "Built with Streamlit + Plotly · all charts and cards respond live to the sidebar filters "
    "and, in select tabs, to your on-chart selections."
)