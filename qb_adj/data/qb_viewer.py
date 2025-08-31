
import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="QB Game Viewer — GitHub Dark (Line+Dots Scaled)", layout="wide")

GITHUB_DARK_CSS = """
<style>
:root{
  --bg:#0d1117; --panel:#0d1117; --card:#161b22; --border:#30363d;
  --text:#c9d1d9; --muted:#8b949e; --accent:#1f6feb; --accent-hover:#388bfd;
  --shadow:0 1px 0 rgba(1,4,9,0.1), 0 1px 3px rgba(1,4,9,0.2);
  --radius:12px; --code:#79c0ff;
  --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif, "Apple Color Emoji","Segoe UI Emoji";
}
html, body, [data-testid="stAppViewContainer"]{ background: var(--bg); color: var(--text); font-family: var(--sans); }
[data-testid="stHeader"]{ background: var(--panel); border-bottom: 1px solid var(--border); }
[data-testid="stSidebar"]{ background: var(--card); border-right:1px solid var(--border); }
.gh-card{ background: var(--card); border:1px solid var(--border); border-radius: var(--radius); box-shadow: var(--shadow); padding: 1rem; }
.gh-subtle{ color: var(--muted); font-size: 0.9rem; }
.gh-badge{ display:inline-block; padding:2px 8px; border:1px solid var(--border); border-radius:999px; background:transparent; font-size:12px; color: var(--muted); }
.stButton>button, .stDownloadButton>button{ background: var(--accent) !important; color: #fff !important; border:1px solid var(--accent) !important; border-radius:8px !important; box-shadow: var(--shadow) !important; }
.stButton>button:hover, .stDownloadButton>button:hover{ background: var(--accent-hover) !important; }
[data-testid="stTable"] table{ background: var(--card) !important; color: var(--text) !important; border-collapse: collapse !important; }
[data-testid="stTable"] th, [data-testid="stTable"] td{ border:1px solid var(--border) !important; padding: 0.5rem !important; }
[data-testid="stTable"] thead tr{ background: #0f1621 !important; }
[data-testid="stDataFrame"]{ background: var(--card) !important; border:1px solid var(--border) !important; border-radius: var(--radius) !important; overflow:hidden !important; box-shadow: var(--shadow) !important; }
[data-testid="stDataFrame"] *{ color: var(--text) !important; }
</style>
"""
st.markdown(GITHUB_DARK_CSS, unsafe_allow_html=True)

@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "gameday" in df.columns:
        try:
            df["gameday"] = pd.to_datetime(df["gameday"])
        except Exception:
            pass
    long_home = df.rename(columns={
        "qb1_id":"qb_id","qb1":"qb_name","qb1_value_pre":"qb_value_pre",
        "qb1_adj":"qb_adj","qb1_game_value":"qb_game_value","qb1_value_post":"qb_value_post"
    }).copy()
    long_home["qb_side"] = "home"
    long_home["qb_team"] = long_home.get("home_team", np.nan)
    long_away = df.rename(columns={
        "qb2_id":"qb_id","qb2":"qb_name","qb2_value_pre":"qb_value_pre",
        "qb2_adj":"qb_adj","qb2_game_value":"qb_game_value","qb2_value_post":"qb_value_post"
    }).copy()
    long_away["qb_side"] = "away"
    long_away["qb_team"] = long_away.get("away_team", np.nan)
    keep_cols = [
        "game_id","season","game_type","week","gameday","weekday","gametime",
        "home_team","home_score","away_team","away_score","location","result","total",
        "qb_id","qb_name","qb_team","qb_side",
        "qb_value_pre","qb_adj","qb_game_value","qb_value_post"
    ]
    long_df = pd.concat([long_home[keep_cols], long_away[keep_cols]], ignore_index=True)
    return long_df

def table_format(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [c for c in ["gameday","season","week"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)
    def opponent(row):
        return row.get("away_team") if row.get("qb_side") == "home" else row.get("home_team")
    df["opponent"] = df.apply(opponent, axis=1)
    def outcome(row):
        hs, as_ = row.get("home_score"), row.get("away_score")
        if pd.isna(hs) or pd.isna(as_): return np.nan
        diff = (hs - as_) if row.get("qb_side") == "home" else (as_ - hs)
        if diff > 0: return f"W (+{int(diff)})"
        if diff < 0: return f"L ({int(diff)})"
        return "T (0)"
    df["outcome"] = df.apply(outcome, axis=1)
    show = [
        "gameday","season","week","game_type","qb_name","qb_team","qb_side",
        "opponent","outcome",
        "qb_value_pre","qb_adj","qb_game_value","qb_value_post",
        "home_team","away_team","home_score","away_score","game_id"
    ]
    existing = [c for c in show if c in df.columns]
    return df[existing]

st.markdown('<div class="gh-card" style="margin-bottom:1rem;">'
            '<div style="display:flex;gap:12px;align-items:center;">'
            '<span class="gh-badge">NFL</span>'
            '<h1 style="margin:0;">QB Game Viewer</h1>'
            '</div>'
            '<div class="gh-subtle">QB Rating • Auto x-axis font sizing</div>'
            '</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("### Filters")
csv_default = "new_elo_file.csv"
csv_path = st.sidebar.text_input("CSV path", value=csv_default)
load_btn = st.sidebar.button("Load / Reload", type="primary")

if "data" not in st.session_state or load_btn:
    try:
        data = load_data(csv_path if os.path.exists(csv_path) else csv_default)
        st.session_state["data"] = data
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

data = st.session_state["data"]
st.sidebar.caption(f"Loaded **{len(data):,}** QB-game rows")

with st.sidebar:
    seasons = sorted([s for s in data["season"].dropna().unique().tolist() if pd.notna(s)])
    season_sel = st.multiselect("Season", seasons, default=seasons)
    qbs = sorted([q for q in data["qb_name"].dropna().unique().tolist() if isinstance(q, str)])
    qb_sel = st.multiselect("Quarterback", qbs, default=[])
    teams = sorted([t for t in pd.unique(pd.concat([data["home_team"], data["away_team"], data["qb_team"]], ignore_index=True).dropna()) if isinstance(t, str)])
    team_sel = st.multiselect("Team", teams, default=[])
    side_sel = st.multiselect("Side", ["home","away"], default=["home","away"])
    game_type_sel = st.multiselect("Game Type", sorted([gt for gt in data["game_type"].dropna().unique().tolist() if isinstance(gt, str)]), default=[])
    week_min, week_max = int(data["week"].min()), int(data["week"].max())
    week_range = st.slider("Week range", min_value=week_min, max_value=week_max, value=(week_min, week_max))

# Main table
st.markdown('<div class="gh-card">', unsafe_allow_html=True)
st.subheader("Filtered Games")
f = data.copy()
if season_sel: f = f[f["season"].isin(season_sel)]
if qb_sel: f = f[f["qb_name"].isin(qb_sel)]
if team_sel: f = f[(f["qb_team"].isin(team_sel)) | (f["home_team"].isin(team_sel)) | (f["away_team"].isin(team_sel))]
if side_sel: f = f[f["qb_side"].isin(side_sel)]
if game_type_sel: f = f[f["game_type"].isin(game_type_sel)]
f = f[(f["week"] >= week_range[0]) & (f["week"] <= week_range[1])]

disp = table_format(f)
st.dataframe(disp, use_container_width=True, height=450)
st.markdown('</div>', unsafe_allow_html=True)

# Chart
st.markdown('<div class="gh-card" style="margin-top:1rem;">', unsafe_allow_html=True)
st.subheader("Per-QB Trajectory (Pre-line + In-game Dots ×3.3)")
group_choice = st.selectbox("Choose a QB to chart", ["(none)"] + sorted(disp["qb_name"].dropna().unique().tolist()))
if group_choice and group_choice != "(none)":
    qb_df = disp[disp["qb_name"] == group_choice].copy().sort_values(["season","week"])
    chart_df = qb_df[["season","week","qb_value_pre","qb_game_value"]].copy()
    chart_df["label"] = qb_df["season"].astype(str) + " wk" + qb_df["week"].astype(int).astype(str)
    chart_df = chart_df.set_index("label")

    # Scale the in-game values by 3.3
    scaled_game_values = chart_df["qb_game_value"] * 1

    # Adaptive x-axis font size to avoid squish
    n = len(chart_df.index)
    if n <= 12:
        tick_fontsize = 12
    elif n <= 20:
        tick_fontsize = 10
    elif n <= 35:
        tick_fontsize = 9
    elif n <= 60:
        tick_fontsize = 8
    else:
        tick_fontsize = 7

    fig = plt.figure()
    x = range(n)
    # Pre-game rating line
    plt.plot(x, chart_df["qb_value_pre"], label="Pre-game rating")
    # In-game dots (scaled)
    plt.scatter(x, scaled_game_values, label="In-game value ×3.3")
    # Labels
    plt.xticks(x, chart_df.index, rotation=45, ha="right", fontsize=tick_fontsize)
    plt.title(f"{group_choice}: Pre-game (line) vs In-game ×3.3 (dots)")
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("**QB game log**")
    st.dataframe(qb_df, use_container_width=True, height=350)
st.markdown('</div>', unsafe_allow_html=True)

# Summary
st.markdown('<div class="gh-card" style="margin-top:1rem;">', unsafe_allow_html=True)
st.subheader("Summary (Filtered Set)")
summ = f.groupby("qb_name", dropna=True).agg(
    games=("game_id","nunique"),
    avg_pre=("qb_value_pre","mean"),
    avg_game=("qb_game_value","mean"),
    avg_post=("qb_value_post","mean")
).reset_index().sort_values("games", ascending=False)
st.dataframe(summ, use_container_width=True, height=350)
st.markdown('</div>', unsafe_allow_html=True)

st.caption("Trajectory uses pre-game rating as a line and in-game value (×3.3) as dots. X-axis font size adapts to number of games.")
