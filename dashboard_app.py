"""
Professional Streamlit dashboard for the MLB Sports Model Pro Template.

v14 updates:
- Dark professional UI
- Dark styled HTML tables instead of default white Streamlit tables
- Player cards with headshots/team logos
- Live schedule cards
- Starter season data cards from included Pro CSVs
- Player search for season totals, rolling trends, and history
- Recomputes season totals from master game logs so included demo totals stay realistic
- Dark Streamlit native controls: select boxes, search fields, sidebar command boxes
- Team pages, matchup lab, trend charts, and model builder tab
- No subscriber gate, no payment logic, all tabs available
"""

from __future__ import annotations

import html
import os
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DAILY_OUT = ROOT / "daily_out"
RESULTS_DIR = ROOT / "results"
LOGS_DIR = ROOT / "logs"
ACCURACY_DIR = ROOT / "accuracy_out"
SLATES = ["full", "early", "mid", "late"]
TOP10_MARKETS = ["H", "HRR", "K"]

HEADSHOTS_PATH = DATA_DIR / "player_headshots.csv"
TEAM_LOGOS_PATH = DATA_DIR / "team_logos.csv"

st.set_page_config(page_title="MLB Model Dashboard", page_icon="⚾", layout="wide")


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root{
          --bg:#0b1220;
          --panel:#111827;
          --panel2:#0f172a;
          --muted:#94a3b8;
          --text:#e5e7eb;
          --line:rgba(148,163,184,.18);
          --accent:#60a5fa;
          --good:#22c55e;
          --bad:#ef4444;
          --warn:#f59e0b;
        }
        .stApp{background:linear-gradient(180deg,#0b1220 0%,#0f172a 45%,#0b1220 100%); color:var(--text);}
        h1,h2,h3,h4,p,span,div,label{color:var(--text);}
        [data-testid="stSidebar"]{background:#070d18;border-right:1px solid var(--line);}
        [data-testid="stMetric"]{background:rgba(17,24,39,.9);border:1px solid var(--line);border-radius:16px;padding:16px;box-shadow:0 12px 28px rgba(0,0,0,.35);}
        [data-testid="stMetricLabel"] p{color:var(--muted)!important;font-weight:700;}
        [data-testid="stMetricValue"]{color:#fff!important;}
        .stTabs [data-baseweb="tab-list"]{gap:8px;border-bottom:1px solid var(--line);}
        .stTabs [data-baseweb="tab"]{background:rgba(15,23,42,.75);border:1px solid rgba(148,163,184,.12);border-radius:12px 12px 0 0;padding:10px 16px;color:#cbd5e1;}
        .stTabs [aria-selected="true"]{background:#1e293b;color:#fff;border-color:rgba(96,165,250,.45);}
        .stDataFrame, [data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:14px;overflow:hidden;}

        /* Darken Streamlit native inputs/selects/date controls/code boxes */
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-testid="stNumberInput"] input,
        [data-testid="stDateInput"] input,
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        [data-baseweb="select"] > div,
        [data-baseweb="input"] input,
        [data-baseweb="base-input"] input,
        div[data-baseweb="select"] div,
        div[data-baseweb="popover"] div,
        div[data-baseweb="menu"],
        ul[role="listbox"],
        li[role="option"]{
          background-color:#111827!important;
          color:#f8fafc!important;
          border-color:rgba(148,163,184,.28)!important;
        }
        [data-testid="stTextInput"] input,
        [data-testid="stDateInput"] input,
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div{
          border:1px solid rgba(148,163,184,.28)!important;
          border-radius:12px!important;
          box-shadow:none!important;
        }
        [data-testid="stTextInput"] input:focus,
        [data-testid="stDateInput"] input:focus,
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within{
          border-color:rgba(96,165,250,.70)!important;
          box-shadow:0 0 0 1px rgba(96,165,250,.35)!important;
        }
        [data-testid="stTextInput"] input::placeholder,
        input::placeholder{
          color:#64748b!important;
          opacity:1!important;
        }
        [data-testid="stSelectbox"] svg,
        [data-baseweb="select"] svg{
          fill:#cbd5e1!important;
          color:#cbd5e1!important;
        }
        [data-testid="stCodeBlock"],
        [data-testid="stCodeBlock"] pre,
        [data-testid="stCodeBlock"] code,
        pre, code{
          background:#111827!important;
          color:#dbeafe!important;
          border:1px solid rgba(148,163,184,.20)!important;
          border-radius:12px!important;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label{
          color:#cbd5e1!important;
        }
        .hero-card{background:linear-gradient(135deg,rgba(17,24,39,.98),rgba(30,41,59,.92));border:1px solid rgba(96,165,250,.22);border-radius:24px;padding:24px;margin:8px 0 20px;box-shadow:0 20px 50px rgba(0,0,0,.42);}
        .hero-title{font-size:2.35rem;font-weight:900;line-height:1.05;margin-bottom:6px;color:#fff;letter-spacing:-.02em;}
        .hero-sub{color:#a8b3c7;font-size:1rem;}
        .small-note{color:#94a3b8;font-size:.9rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_theme()


def norm_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def esc(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return html.escape(str(x))


@st.cache_data(show_spinner=False)
def load_csv_cached(path_str: str, mtime: float) -> pd.DataFrame:
    path = Path(path_str)
    df = pd.read_csv(path, low_memory=False)
    for col in ["DATE", "GAME_DATE", "COMMENCE_TIME", "START_TIME_CT", "START_TIME_LOCAL"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_csv(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return load_csv_cached(str(path), path.stat().st_mtime)
    except Exception as exc:
        st.warning(f"Could not load {path}: {exc}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_mlb_schedule_live(day: date) -> pd.DataFrame:
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {"sportId": 1, "date": day.strftime("%Y-%m-%d"), "hydrate": "team,venue,probablePitcher"}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        js = r.json()
        rows = []
        for d in js.get("dates", []):
            for g in d.get("games", []):
                home = g.get("teams", {}).get("home", {})
                away = g.get("teams", {}).get("away", {})
                hp = home.get("probablePitcher", {}) or {}
                ap = away.get("probablePitcher", {}) or {}
                rows.append({
                    "GAME_DATE": d.get("date"),
                    "GAME_PK": g.get("gamePk"),
                    "AWAY_TEAM": away.get("team", {}).get("name", ""),
                    "HOME_TEAM": home.get("team", {}).get("name", ""),
                    "VENUE": g.get("venue", {}).get("name", ""),
                    "START_TIME_LOCAL": g.get("gameDate"),
                    "STATUS": g.get("status", {}).get("detailedState", ""),
                    "AWAY_PROBABLE_PITCHER_NAME": ap.get("fullName", ""),
                    "HOME_PROBABLE_PITCHER_NAME": hp.get("fullName", ""),
                })
        df = pd.DataFrame(rows)
        if not df.empty and "START_TIME_LOCAL" in df.columns:
            utc = pd.to_datetime(df["START_TIME_LOCAL"], errors="coerce", utc=True)
            df["START_TIME_CT"] = utc.dt.tz_convert("America/Chicago").dt.strftime("%-I:%M %p CT")
        return df
    except Exception:
        return pd.DataFrame()


def load_schedule_for_today() -> pd.DataFrame:
    local = load_csv(DATA_DIR / "mlb_schedule_today.csv")
    if not local.empty:
        if "START_TIME_LOCAL" in local.columns and "START_TIME_CT" not in local.columns:
            utc = pd.to_datetime(local["START_TIME_LOCAL"], errors="coerce", utc=True)
            local["START_TIME_CT"] = utc.dt.tz_convert("America/Chicago").dt.strftime("%-I:%M %p CT")
        return local
    return fetch_mlb_schedule_live(date.today())


@st.cache_data(show_spinner=False)
def asset_maps(headshot_mtime: float | None, logo_mtime: float | None) -> tuple[dict, dict, dict]:
    headshots_by_id = {}
    headshots_by_name = {}
    logos_by_team = {}

    if HEADSHOTS_PATH.exists():
        hs = pd.read_csv(HEADSHOTS_PATH, low_memory=False)
        for _, r in hs.iterrows():
            url = r.get("headshot_url", "")
            if not str(url).strip():
                continue
            try:
                pid = int(float(r.get("player_id")))
                headshots_by_id[pid] = url
            except Exception:
                pass
            name = norm_text(r.get("player"))
            if name:
                headshots_by_name[name] = url

    if TEAM_LOGOS_PATH.exists():
        logos = pd.read_csv(TEAM_LOGOS_PATH, low_memory=False)
        for _, r in logos.iterrows():
            url = r.get("logo_url", "")
            if not str(url).strip():
                continue
            full = norm_text(r.get("team_full"))
            abbr = norm_text(r.get("team_abbr"))
            if full:
                logos_by_team[full] = {"url": url, "abbr": r.get("team_abbr", ""), "primary": r.get("primary_color", "#1f2937")}
            if abbr:
                logos_by_team[abbr] = {"url": url, "abbr": r.get("team_abbr", ""), "primary": r.get("primary_color", "#1f2937")}
        # Common alias for current Athletics naming.
        if "athletics" in logos_by_team:
            logos_by_team["oakland athletics"] = logos_by_team["athletics"]
    return headshots_by_id, headshots_by_name, logos_by_team


def get_assets():
    hs_m = HEADSHOTS_PATH.stat().st_mtime if HEADSHOTS_PATH.exists() else None
    lg_m = TEAM_LOGOS_PATH.stat().st_mtime if TEAM_LOGOS_PATH.exists() else None
    return asset_maps(hs_m, lg_m)


def headshot_url(player_id=None, player_name=None) -> str:
    by_id, by_name, _ = get_assets()
    try:
        if pd.notna(player_id):
            url = by_id.get(int(float(player_id)), "")
            if url:
                return url
    except Exception:
        pass
    return by_name.get(norm_text(player_name), "")


def logo_info(team_name=None) -> dict:
    _, _, logos = get_assets()
    return logos.get(norm_text(team_name), {})


def fmt_num(x, digits=3, blank="—") -> str:
    try:
        if pd.isna(x):
            return blank
        return f"{float(x):.{digits}f}"
    except Exception:
        return blank


def fmt_int(x, blank="0") -> str:
    try:
        if pd.isna(x):
            return blank
        return str(int(float(x)))
    except Exception:
        return blank


def _format_cell(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        # Keep probabilities readable but avoid huge floats.
        if abs(value) < 1 and value != 0:
            return f"{value:.3f}"
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:.2f}".rstrip("0").rstrip(".")
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    text = str(value)
    if len(text) > 80:
        text = text[:77] + "..."
    return text


def dark_table_html(df: pd.DataFrame, max_rows: int = 200, table_id: str = "dark-table") -> str:
    if df is None or df.empty:
        return ""
    d = df.head(max_rows).copy()
    # Convert datetimes to friendly strings.
    for c in d.columns:
        if pd.api.types.is_datetime64_any_dtype(d[c]):
            d[c] = d[c].dt.strftime("%Y-%m-%d %H:%M").fillna("")
    headers = "".join(f"<th>{esc(c)}</th>" for c in d.columns)
    rows = []
    for _, row in d.iterrows():
        cells = "".join(f"<td>{esc(_format_cell(row[c]))}</td>" for c in d.columns)
        rows.append(f"<tr>{cells}</tr>")
    return f"""
    <div class="table-wrap" id="{esc(table_id)}">
      <table class="dark-table">
        <thead><tr>{headers}</tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    """


def show_df(title: str, df: pd.DataFrame, max_rows: int = 200):
    st.subheader(title)
    if df is None or df.empty:
        st.info("No file found yet. Run the pipeline first or check the selected date/slate.")
        return
    st.caption(f"Rows: {len(df):,}")
    html_block = """
    <style>
      .table-wrap{width:100%;max-height:520px;overflow:auto;border:1px solid rgba(148,163,184,.18);border-radius:14px;background:#0b1220;box-shadow:0 12px 28px rgba(0,0,0,.35);}
      .dark-table{width:100%;border-collapse:separate;border-spacing:0;color:#e5e7eb;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:13px;min-width:900px;}
      .dark-table thead th{position:sticky;top:0;background:#111827;color:#cbd5e1;text-transform:uppercase;letter-spacing:.04em;font-size:11px;font-weight:800;padding:11px 12px;border-bottom:1px solid rgba(148,163,184,.18);text-align:left;white-space:nowrap;z-index:2;}
      .dark-table tbody td{padding:10px 12px;border-bottom:1px solid rgba(148,163,184,.10);color:#e5e7eb;white-space:nowrap;}
      .dark-table tbody tr:nth-child(even){background:rgba(15,23,42,.65);}
      .dark-table tbody tr:nth-child(odd){background:rgba(17,24,39,.45);}
      .dark-table tbody tr:hover{background:rgba(96,165,250,.14);}
    </style>
    """ + dark_table_html(df, max_rows=max_rows, table_id=re.sub(r"[^a-z0-9]+", "-", title.lower()))
    st.markdown(html_block, unsafe_allow_html=True)


def metric_card(label: str, value):
    st.metric(label, value if value is not None else "-")


def render_html_block(html_text: str, height: int = 800):
    components.html(html_text, height=height, scrolling=True)


def base_html_start() -> str:
    return """
    <html><head><style>
      body{margin:0;background:#0b1220;color:#e5e7eb;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}
      .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px;padding:8px 2px 18px;}
      .game-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px;padding:8px 2px 18px;}
      .card{position:relative;background:#111827;border:1px solid rgba(148,163,184,.16);border-radius:18px;padding:16px;box-shadow:0 14px 34px rgba(0,0,0,.45);overflow:hidden;min-height:178px;}
      .card:before{content:"";position:absolute;inset:0;background:radial-gradient(circle at 90% 10%,rgba(96,165,250,.12),transparent 35%);pointer-events:none;}
      .row{display:flex;gap:14px;align-items:flex-start;position:relative;z-index:1;}
      .hs{width:68px;height:68px;border-radius:50%;object-fit:cover;border:2px solid rgba(148,163,184,.25);background:#0b1220;flex:0 0 auto;}
      .logo{width:42px;height:42px;object-fit:contain;filter:drop-shadow(0 4px 10px rgba(0,0,0,.45));}
      .title{font-size:1.08rem;font-weight:900;color:#fff;line-height:1.1;margin-bottom:4px;}
      .subtitle{color:#cbd5e1;font-size:.86rem;line-height:1.25;}
      .muted{color:#94a3b8;font-size:.82rem;line-height:1.3;}
      .chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px;position:relative;z-index:1;}
      .chip{background:#1f2937;border:1px solid rgba(148,163,184,.18);border-radius:999px;padding:6px 10px;color:#e5e7eb;font-size:.82rem;white-space:nowrap;}
      .chip strong{color:#fff;margin-left:5px;}
      .hit{background:rgba(34,197,94,.18);border-color:rgba(34,197,94,.38);}
      .miss{background:rgba(239,68,68,.18);border-color:rgba(239,68,68,.38);}
      .warn{background:rgba(245,158,11,.16);border-color:rgba(245,158,11,.34);}
      .teams{display:flex;align-items:center;justify-content:space-between;gap:10px;margin:8px 0 10px;position:relative;z-index:1;}
      .team{display:flex;align-items:center;gap:9px;min-width:0;}
      .team-name{font-weight:800;color:#fff;font-size:.98rem;line-height:1.1;}
      .vs{color:#94a3b8;font-weight:900;letter-spacing:.08em;}
      .section-label{font-size:.78rem;color:#93c5fd;text-transform:uppercase;letter-spacing:.08em;font-weight:800;margin-bottom:8px;position:relative;z-index:1;}
    </style></head><body>
    """


def render_schedule_cards(schedule: pd.DataFrame, max_cards: int = 18):
    if schedule.empty:
        st.info("No MLB games found for today.")
        return
    parts = [base_html_start(), '<div class="game-grid">']
    for _, r in schedule.head(max_cards).iterrows():
        away = r.get("AWAY_TEAM", "")
        home = r.get("HOME_TEAM", "")
        away_logo = logo_info(away).get("url", "")
        home_logo = logo_info(home).get("url", "")
        time = r.get("START_TIME_CT", "")
        venue = r.get("VENUE", "")
        status = r.get("STATUS", "")
        ap = r.get("AWAY_PROBABLE_PITCHER_NAME", "")
        hp = r.get("HOME_PROBABLE_PITCHER_NAME", "")
        parts.append(f"""
        <div class="card">
          <div class="section-label">Today's Game</div>
          <div class="teams">
            <div class="team"><img src="{esc(away_logo)}" class="logo" onerror="this.style.display='none'"/><div class="team-name">{esc(away)}</div></div>
            <div class="vs">@</div>
            <div class="team" style="justify-content:flex-end;text-align:right;"><div class="team-name">{esc(home)}</div><img src="{esc(home_logo)}" class="logo" onerror="this.style.display='none'"/></div>
          </div>
          <div class="chips">
            <span class="chip">Time <strong>{esc(time)}</strong></span>
            <span class="chip">Status <strong>{esc(status)}</strong></span>
          </div>
          <div class="muted" style="margin-top:12px;">{esc(venue)}</div>
          <div class="muted" style="margin-top:8px;">Probables: {esc(ap or 'TBD')} vs {esc(hp or 'TBD')}</div>
        </div>
        """)
    parts.append("</div></body></html>")
    height = min(1300, 250 + 260 * ((min(len(schedule), max_cards) + 1) // 2))
    render_html_block("".join(parts), height=height)


def render_batter_cards(title: str, df: pd.DataFrame, max_cards: int = 24):
    st.subheader(title)
    if df is None or df.empty:
        st.info("No player rows found.")
        return
    d = df.copy().head(max_cards)
    parts = [base_html_start(), '<div class="grid">']
    for _, r in d.iterrows():
        pid = r.get("PLAYER_ID", r.get("ENTITY_ID", np.nan))
        name = r.get("PLAYER_NAME", r.get("ENTITY_NAME", "Unknown Player"))
        team = r.get("TEAM", "")
        opp = r.get("OPP", "")
        img = headshot_url(pid, name)
        logo = logo_info(team).get("url", "")
        team_abbr = logo_info(team).get("abbr", team)
        game_date = r.get("DATE", "")
        try:
            game_date = pd.to_datetime(game_date).strftime("%b %d, %Y")
        except Exception:
            pass
        parts.append(f"""
        <div class="card">
          <div class="row">
            <img src="{esc(img)}" class="hs" onerror="this.style.display='none'"/>
            <div style="flex:1;min-width:0;">
              <div class="title">{esc(name)}</div>
              <div class="subtitle">{esc(game_date)} • vs {esc(opp)}</div>
              <div class="muted">{esc(team)}</div>
            </div>
            <img src="{esc(logo)}" class="logo" onerror="this.style.display='none'"/>
          </div>
          <div class="chips">
            <span class="chip">H <strong>{fmt_int(r.get('HITS', r.get('H_R5', np.nan)))}</strong></span>
            <span class="chip">TB <strong>{fmt_int(r.get('TOTAL_BASES', r.get('TB_R5', np.nan)))}</strong></span>
            <span class="chip">HR <strong>{fmt_int(r.get('HOME_RUNS', r.get('HR_R5', np.nan)))}</strong></span>
            <span class="chip">K <strong>{fmt_int(r.get('STRIKEOUTS', r.get('K_R5', np.nan)))}</strong></span>
            <span class="chip">BB <strong>{fmt_int(r.get('WALKS', r.get('BB_R5', np.nan)))}</strong></span>
            <span class="chip">SB <strong>{fmt_int(r.get('STOLEN_BASES', r.get('SB_R5', np.nan)))}</strong></span>
          </div>
        </div>
        """)
    parts.append("</div></body></html>")
    height = min(1800, 260 + 260 * ((min(len(d), max_cards) + 2) // 3))
    render_html_block("".join(parts), height=height)


def render_pick_cards(title: str, df: pd.DataFrame, max_cards: int = 18):
    st.subheader(title)
    if df is None or df.empty:
        st.info("No pick rows found yet. Run the daily pipeline to populate this board.")
        return
    d = df.copy().head(max_cards)
    parts = [base_html_start(), '<div class="grid">']
    for _, r in d.iterrows():
        pid = r.get("ENTITY_ID", r.get("PLAYER_ID", np.nan))
        name = r.get("PLAYER_NAME", r.get("ENTITY_NAME", "Unknown Player"))
        team = r.get("TEAM", "")
        img = headshot_url(pid, name)
        logo = logo_info(team).get("url", "")
        market = r.get("MARKET_FAMILY", "")
        line = r.get("LINE", "")
        odds = r.get("ODDS", "")
        prob = r.get("MODEL_PROB_FINAL", r.get("MODEL_PROB_USED", r.get("MODEL_PROB_CAL", np.nan)))
        edge = r.get("EDGE", np.nan)
        tier = r.get("CONFIDENCE_TIER", "")
        parts.append(f"""
        <div class="card">
          <div class="row">
            <img src="{esc(img)}" class="hs" onerror="this.style.display='none'"/>
            <div style="flex:1;min-width:0;">
              <div class="title">{esc(name)}</div>
              <div class="subtitle">{esc(team)}</div>
              <div class="muted">{esc(market)} over {esc(line)} • {esc(odds)}</div>
            </div>
            <img src="{esc(logo)}" class="logo" onerror="this.style.display='none'"/>
          </div>
          <div class="chips">
            <span class="chip">Model <strong>{fmt_num(prob)}</strong></span>
            <span class="chip">Edge <strong>{fmt_num(edge)}</strong></span>
            <span class="chip">Tier <strong>{esc(tier)}</strong></span>
          </div>
        </div>
        """)
    parts.append("</div></body></html>")
    height = min(1600, 260 + 260 * ((min(len(d), max_cards) + 2) // 3))
    render_html_block("".join(parts), height=height)




def _safe_get(row: pd.Series, *names, default=""):
    for name in names:
        if name in row.index:
            val = row.get(name)
            if pd.notna(val) and str(val).strip() != "":
                return val
    return default


def _format_odds(x) -> str:
    try:
        if pd.isna(x):
            return "—"
        return f"{int(float(x)):+d}"
    except Exception:
        return esc(x) if x not in [None, ""] else "—"


def _extract_parlay_legs(row: pd.Series) -> list[dict]:
    # Return P1/P2/... parlay legs, or a single-leg fallback for simple files.
    legs = []
    for i in range(1, 9):
        player = _safe_get(row, f"P{i}_PLAYER", f"LEG{i}_PLAYER", default="")
        if not str(player).strip():
            continue
        legs.append({
            "player": player,
            "team": _safe_get(row, f"P{i}_TEAM", f"LEG{i}_TEAM", default=""),
            "market": _safe_get(row, f"P{i}_MARKET", f"LEG{i}_MARKET", default=""),
            "line": _safe_get(row, f"P{i}_LINE", f"LEG{i}_LINE", default=""),
            "odds": _safe_get(row, f"P{i}_ODDS", f"LEG{i}_ODDS", default=""),
            "prob": _safe_get(row, f"P{i}_PROB", f"P{i}_MODEL_PROB", f"LEG{i}_PROB", default=np.nan),
            "edge": _safe_get(row, f"P{i}_EDGE", f"LEG{i}_EDGE", default=np.nan),
            "status": _safe_get(row, f"P{i}_STATUS", f"LEG{i}_STATUS", default=""),
            "actual": _safe_get(row, f"P{i}_ACTUAL", f"LEG{i}_ACTUAL", default=""),
            "result_line": _safe_get(row, f"P{i}_RESULT_LINE", f"LEG{i}_RESULT_LINE", default=""),
            "hit": _safe_get(row, f"P{i}_HIT", f"LEG{i}_HIT", default=""),
            "pid": _safe_get(row, f"P{i}_ENTITY_ID", f"P{i}_PLAYER_ID", f"LEG{i}_ENTITY_ID", default=np.nan),
            "game": _safe_get(row, f"P{i}_GAME", "GAME_KEY", "GAME_ID", default=""),
        })

    if not legs:
        player = _safe_get(row, "PLAYER_NAME", "PLAYER", "ENTITY_NAME", default="")
        if str(player).strip():
            legs.append({
                "player": player,
                "team": _safe_get(row, "TEAM", "TEAM_OUT", default=""),
                "market": _safe_get(row, "MARKET_FAMILY", "MARKET", default=""),
                "line": _safe_get(row, "LINE", "LINE_OUT", default=""),
                "odds": _safe_get(row, "ODDS", "ODDS_OUT", default=""),
                "prob": _safe_get(row, "MODEL_PROB_FINAL", "MODEL_PROB_USED", "MODEL_PROB_META", default=np.nan),
                "edge": _safe_get(row, "EDGE", "EDGE_META", default=np.nan),
                "status": _safe_get(row, "RESULT_STATUS", "STATUS", default=""),
                "actual": _safe_get(row, "ACTUAL", "RESULT_ACTUAL", default=""),
                "result_line": _safe_get(row, "RESULT_LINE", default=""),
                "hit": _safe_get(row, "PICK_HIT", "HIT", default=""),
                "pid": _safe_get(row, "ENTITY_ID", "PLAYER_ID", default=np.nan),
                "game": _safe_get(row, "GAME_KEY", "GAME_ID", default=""),
            })
    return legs



def _result_status_from_values(status="", hit="") -> str:
    status_txt = str(status).strip().upper()
    if status_txt in {"HIT", "MISS", "PENDING", "DNP", "VOID/REDUCED", "VOID", "PUSH"}:
        return status_txt
    hit_txt = str(hit).strip().upper()
    if hit_txt in {"1", "1.0", "TRUE", "YES"}:
        return "HIT"
    if hit_txt in {"0", "0.0", "FALSE", "NO"}:
        return "MISS"
    return "PENDING"


def _status_class(status: str) -> str:
    s = str(status).strip().upper()
    if s == "HIT":
        return "result-hit"
    if s == "MISS":
        return "result-miss"
    if s in {"DNP", "VOID", "VOID/REDUCED", "PUSH"}:
        return "result-void"
    return "result-pending"

def _parlay_card_html(row: pd.Series, label: str, max_legs: int = 8) -> str:
    legs = _extract_parlay_legs(row)[:max_legs]
    style = _safe_get(row, "STYLE", "PRODUCT", "ACTION", default=label)
    num_legs = _safe_get(row, "NUM_LEGS", default=len(legs))
    model_prob = _safe_get(row, "PARLAY_MODEL_PROB", "MODEL_PROB_FINAL", "MODEL_PROB_META", default=np.nan)
    book_prob = _safe_get(row, "PARLAY_BOOK_PROB", default=np.nan)
    decimal = _safe_get(row, "PARLAY_DECIMAL", default=np.nan)
    american = _safe_get(row, "PARLAY_AMERICAN", default=np.nan)
    edge = _safe_get(row, "PARLAY_EDGE", "EDGE", "EDGE_META", default=np.nan)
    score = _safe_get(row, "PARLAY_SCORE", "META_SCORE", default=np.nan)
    reason = _safe_get(row, "REASON", default="")
    parlay_status = _safe_get(row, "PARLAY_STATUS", "RESULT_STATUS", default="")
    parlay_status = _result_status_from_values(parlay_status, _safe_get(row, "PARLAY_HIT", default=""))
    resolved_legs = _safe_get(row, "PARLAY_RESOLVED_LEGS", default="")
    parlay_hits = _safe_get(row, "PARLAY_HITS", default="")
    parlay_misses = _safe_get(row, "PARLAY_MISSES", default="")

    leg_html = []
    for leg in legs:
        img = headshot_url(leg.get("pid"), leg.get("player"))
        logo = logo_info(leg.get("team")).get("url", "")
        market = str(leg.get("market", "")).upper()
        line = leg.get("line", "")
        line_txt = f"o{line}" if str(line).strip() and str(line).lower() != "nan" else ""
        odds_txt = _format_odds(leg.get("odds"))
        play = " ".join([x for x in [market, line_txt, odds_txt] if str(x).strip() and str(x) != "—"])
        leg_status = _result_status_from_values(leg.get("status"), leg.get("hit"))
        actual_txt = leg.get("actual", "")
        result_line_txt = leg.get("result_line", "") or leg.get("line", "")
        leg_html.append(f"""
          <div class="leg-card {esc(_status_class(leg_status))}">
            <div class="leg-row">
              <img src="{esc(img)}" class="leg-headshot" onerror="this.style.display='none'"/>
              <div class="leg-main">
                <div class="leg-player">{esc(leg.get('player'))}</div>
                <div class="leg-play">{esc(play)}</div>
                <div class="leg-sub">{esc(leg.get('team'))}</div>
              </div>
              <img src="{esc(logo)}" class="leg-logo" onerror="this.style.display='none'"/>
            </div>
            <div class="mini-chips">
              <span>Model <b>{fmt_num(leg.get('prob'))}</b></span>
              <span>Edge <b>{fmt_num(leg.get('edge'))}</b></span>
              <span class="{esc(_status_class(leg_status))}">Result <b>{esc(leg_status)}</b></span>
              <span>Actual <b>{esc(actual_txt)}</b></span>
              <span>Line <b>{esc(result_line_txt)}</b></span>
            </div>
          </div>
        """)

    if not leg_html:
        leg_html.append('<div class="empty-leg">No leg details found for this row.</div>')

    return f"""
      <div class="parlay-card">
        <div class="parlay-top">
          <div>
            <div class="parlay-eyebrow">{esc(label)}</div>
            <div class="parlay-title">{esc(style)} • {esc(num_legs)} legs</div>
            <div class="parlay-reason">{esc(reason)}</div>
          </div>
          <div class="parlay-side">
            <div class="parlay-status {esc(_status_class(parlay_status))}">{esc(parlay_status)}</div>
            <div class="parlay-odds">{_format_odds(american)}</div>
          </div>
        </div>
        <div class="parlay-stats">
          <span>Model <b>{fmt_num(model_prob)}</b></span>
          <span>Book <b>{fmt_num(book_prob)}</b></span>
          <span>Decimal <b>{fmt_num(decimal, 2)}</b></span>
          <span>Edge <b>{fmt_num(edge)}</b></span>
          <span>Score <b>{fmt_num(score)}</b></span>
          <span>Legs <b>{esc(resolved_legs)}</b></span>
          <span>Hits <b>{esc(parlay_hits)}</b></span>
          <span>Misses <b>{esc(parlay_misses)}</b></span>
        </div>
        <div class="legs-grid">{''.join(leg_html)}</div>
      </div>
    """


def render_parlay_card_board(title: str, df: pd.DataFrame, *, label: str | None = None, max_cards: int = 12):
    st.subheader(title)
    if df is None or df.empty:
        st.info("No parlay/meta rows found yet. Run the daily pipeline to populate this board.")
        return
    label = label or title
    d = df.head(max_cards).copy()
    html_parts = [base_html_start(), """
    <style>
      .parlay-board{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:18px;padding:8px 2px 18px;}
      .parlay-card{background:linear-gradient(135deg,#111827,#0f172a);border:1px solid rgba(148,163,184,.18);border-radius:20px;padding:16px;box-shadow:0 16px 38px rgba(0,0,0,.42);overflow:hidden;}
      .parlay-top{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;margin-bottom:12px;}
      .parlay-eyebrow{font-size:.78rem;color:#93c5fd;text-transform:uppercase;letter-spacing:.09em;font-weight:900;margin-bottom:5px;}
      .parlay-title{font-size:1.2rem;color:#fff;font-weight:900;line-height:1.1;}
      .parlay-reason{font-size:.83rem;color:#94a3b8;margin-top:6px;line-height:1.35;max-width:720px;}
      .parlay-side{display:flex;flex-direction:column;align-items:flex-end;gap:8px;}
      .parlay-odds{background:linear-gradient(135deg,#1e3a8a,#0f172a);border:1px solid rgba(96,165,250,.35);border-radius:14px;padding:10px 14px;color:#fff;font-size:1.12rem;font-weight:900;white-space:nowrap;}
      .parlay-status{border-radius:999px;padding:7px 12px;font-size:.82rem;font-weight:900;text-transform:uppercase;letter-spacing:.05em;border:1px solid rgba(148,163,184,.20);}
      .parlay-stats{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0 14px;}
      .parlay-stats span,.mini-chips span{background:#1f2937;border:1px solid rgba(148,163,184,.18);border-radius:999px;padding:6px 10px;color:#cbd5e1;font-size:.82rem;}
      .parlay-stats b,.mini-chips b{color:#fff;margin-left:5px;}
      .legs-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(235px,1fr));gap:10px;}
      .leg-card{background:#0b1220;border:1px solid rgba(148,163,184,.14);border-radius:16px;padding:12px;min-height:116px;}
      .result-hit{background:rgba(22,163,74,.18)!important;border-color:rgba(34,197,94,.38)!important;color:#bbf7d0!important;}
      .result-miss{background:rgba(220,38,38,.16)!important;border-color:rgba(248,113,113,.35)!important;color:#fecaca!important;}
      .result-pending{background:rgba(245,158,11,.13)!important;border-color:rgba(251,191,36,.30)!important;color:#fde68a!important;}
      .result-void{background:rgba(148,163,184,.13)!important;border-color:rgba(148,163,184,.28)!important;color:#e2e8f0!important;}
      .leg-row{display:flex;align-items:flex-start;gap:10px;}
      .leg-headshot{width:52px;height:52px;border-radius:50%;object-fit:cover;border:2px solid rgba(148,163,184,.25);background:#111827;flex:0 0 auto;}
      .leg-logo{width:34px;height:34px;object-fit:contain;filter:drop-shadow(0 4px 10px rgba(0,0,0,.45));margin-left:auto;}
      .leg-main{min-width:0;flex:1;}
      .leg-player{font-weight:900;color:#fff;font-size:.98rem;line-height:1.1;}
      .leg-play{display:inline-block;margin-top:6px;background:rgba(96,165,250,.13);border:1px solid rgba(96,165,250,.28);color:#dbeafe;padding:5px 8px;border-radius:10px;font-weight:800;font-size:.82rem;}
      .leg-sub{color:#94a3b8;font-size:.8rem;margin-top:5px;line-height:1.2;}
      .mini-chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px;}
      .empty-leg{color:#94a3b8;border:1px dashed rgba(148,163,184,.25);border-radius:14px;padding:16px;}
      @media(max-width:700px){.parlay-board{grid-template-columns:1fr}.legs-grid{grid-template-columns:1fr}}
    </style>
    <div class="parlay-board">
    """]
    for _, row in d.iterrows():
        html_parts.append(_parlay_card_html(row, label=label))
    html_parts.append('</div></body></html>')
    height = min(2200, 320 + 310 * len(d))
    render_html_block("".join(html_parts), height=height)

def render_results_cards(title: str, df: pd.DataFrame, max_cards: int = 18):
    st.subheader(title)
    if df is None or df.empty:
        st.info("No results rows found.")
        return
    d = df.copy().head(max_cards)
    parts = [base_html_start(), '<div class="grid">']
    for _, r in d.iterrows():
        pid = r.get("ENTITY_ID", r.get("PLAYER_ID", np.nan))
        name = r.get("PLAYER_NAME", "Unknown Player")
        team = r.get("TEAM", "")
        img = headshot_url(pid, name)
        logo = logo_info(team).get("url", "")
        market = r.get("MARKET_FAMILY", "")
        line = r.get("LINE", r.get("RESULT_LINE", ""))
        actual = r.get("ACTUAL", r.get("RESULT_ACTUAL", ""))
        hit = r.get("HIT", r.get("PICK_HIT", np.nan))
        raw_status = str(r.get("RESULT_STATUS", "")).strip().upper()
        if raw_status in {"HIT", "MISS", "DNP", "PENDING", "VOID/REDUCED"}:
            status = raw_status
        else:
            status = "HIT" if str(hit).strip() in {"1", "1.0", "True", "TRUE"} else "MISS" if str(hit).strip() in {"0", "0.0", "False", "FALSE"} else "PENDING"
        status_class = "hit" if status == "HIT" else "miss" if status == "MISS" else "warn"
        parts.append(f"""
        <div class="card">
          <div class="row">
            <img src="{esc(img)}" class="hs" onerror="this.style.display='none'"/>
            <div style="flex:1;min-width:0;">
              <div class="title">{esc(name)}</div>
              <div class="subtitle">{esc(team)}</div>
              <div class="muted">{esc(market)} over {esc(line)}</div>
            </div>
            <img src="{esc(logo)}" class="logo" onerror="this.style.display='none'"/>
          </div>
          <div class="chips">
            <span class="chip {status_class}">Result <strong>{status}</strong></span>
            <span class="chip">Actual <strong>{esc(actual)}</strong></span>
            <span class="chip">Line <strong>{esc(line)}</strong></span>
          </div>
        </div>
        """)
    parts.append("</div></body></html>")
    height = min(1600, 260 + 250 * ((min(len(d), max_cards) + 2) // 3))
    render_html_block("".join(parts), height=height)




def load_product_result_file(prefix: str, run_date: str, slate: str, market: str | None = None) -> pd.DataFrame:
    """Load scored product/result files from results/, daily_out/, data/, or logs/."""
    p = find_file(prefix, run_date, slate, market)
    return load_csv(p) if p else pd.DataFrame()


def render_product_result_section(label: str, df: pd.DataFrame, *, kind: str = "results", max_rows: int = 150):
    """Render one scored product result section with cards plus expandable raw table."""
    st.markdown(f"### {label}")
    if df is None or df.empty:
        st.info(f"No {label.lower()} file found for the selected date/slate.")
        return

    if kind == "parlay":
        render_parlay_card_board(label, df, label=label, max_cards=10)
    elif kind == "summary":
        show_df(label, df, max_rows=max_rows)
    else:
        sort_df = df.copy()
        hit_col = "PICK_HIT" if "PICK_HIT" in sort_df.columns else "HIT" if "HIT" in sort_df.columns else None
        if hit_col:
            sort_df[hit_col] = pd.to_numeric(sort_df[hit_col], errors="coerce")
            sort_df = sort_df.sort_values(hit_col, ascending=False, na_position="last")
        render_results_cards(label, sort_df, max_cards=18)

    with st.expander(f"Raw table: {label}", expanded=False):
        show_df(label, df, max_rows=max_rows)


def render_product_results_breakdown(run_date: str, slate: str):
    """Show all scored product breakdown files created by run_mlb_results.py."""
    st.markdown("## Product results breakdown")
    st.caption("These sections read the scored product CSVs from the dashboard_publish/results folder after running the results pipeline.")

    meta_compare = load_product_result_file("results_mlb_meta_performance_compare", run_date, slate)
    if not meta_compare.empty:
        render_product_result_section("Meta performance compare", meta_compare, kind="summary", max_rows=80)

    product_sections = [
        ("Meta play card results", "results_mlb_meta_play_card", "results"),
        ("Best 5 results", "results_best5", "results"),
        ("Daily candidates results", "results_daily_candidates", "results"),
        ("Safe parlay results", "results_mlb_safe_parlay", "parlay"),
        ("Ladder parlay results", "results_mlb_ladder_parlay", "parlay"),
        ("Risky parlay results", "results_mlb_risky_parlay", "parlay"),
        ("Same-game parlay results", "results_mlb_specific_game_parlays", "parlay"),
        ("Best single per game results", "results_mlb_best_single_per_game", "results"),
        ("Best single per game parlay results", "results_mlb_best_single_per_game_parlay", "parlay"),
    ]

    for label, prefix, kind in product_sections:
        df = load_product_result_file(prefix, run_date, slate)
        if not df.empty:
            render_product_result_section(label, df, kind=kind, max_rows=150)

    st.markdown("### Top 10 market results")
    any_market = False
    for market in TOP10_MARKETS:
        df = load_product_result_file("results_top10", run_date, slate, market)
        if not df.empty:
            any_market = True
            render_product_result_section(f"Top 10 {market} results", df, kind="results", max_rows=80)
    if not any_market:
        st.info("No scored top-10 market result files found yet for this selected date/slate.")

def latest_batter_sample() -> pd.DataFrame:
    df = load_csv(DATA_DIR / "mlb_batter_gamelogs_master.csv")
    if df.empty or "DATE" not in df.columns:
        return df
    latest = df["DATE"].max()
    d = df[df["DATE"] == latest].copy()
    if "TOTAL_BASES" in d.columns:
        d = d.sort_values(["TOTAL_BASES", "HITS", "HOME_RUNS"], ascending=False)
    return d


def discover_run_dates() -> list[str]:
    found = set()
    pattern = re.compile(r".*_(\d{4}-\d{2}-\d{2})_(full|early|mid|late).*")
    for folder in [DATA_DIR, DAILY_OUT, RESULTS_DIR, LOGS_DIR, ACCURACY_DIR]:
        if not folder.exists():
            continue
        for path in folder.glob("*.csv"):
            m = pattern.match(path.name)
            if m:
                found.add(m.group(1))
    history = load_csv(LOGS_DIR / "mlb_results_history.csv")
    if not history.empty and "DATE" in history.columns:
        found.update(history["DATE"].dropna().dt.strftime("%Y-%m-%d").unique().tolist())
    today = date.today().strftime("%Y-%m-%d")
    if not found:
        found.add(today)
    return sorted(found)


def find_file(prefix: str, run_date: str, slate: str, market: str | None = None) -> Path | None:
    names = []
    if market:
        names += [f"{prefix}_{run_date}_{slate}_{market}.csv", f"{prefix}_{market}_today.csv"]
    else:
        names += [f"{prefix}_{run_date}_{slate}.csv", f"{prefix}_today.csv"]
    for folder in [DAILY_OUT, DATA_DIR, RESULTS_DIR, LOGS_DIR, ACCURACY_DIR]:
        for name in names:
            p = folder / name
            if p.exists() and p.stat().st_size > 0:
                return p
    return None



def convert_baseball_ip(ip) -> float:
    """Convert MLB innings strings like 5.1 / 5.2 to decimal innings."""
    try:
        if pd.isna(ip):
            return 0.0
        text = str(ip).strip()
        if not text:
            return 0.0
        val = float(text)
        whole = int(val)
        frac = round(val - whole, 1)
        if frac == 0.1:
            return whole + (1.0 / 3.0)
        if frac == 0.2:
            return whole + (2.0 / 3.0)
        return val
    except Exception:
        return 0.0


def recompute_batter_season_totals_from_logs(batter_master: pd.DataFrame) -> pd.DataFrame:
    """
    Build season totals directly from mlb_batter_gamelogs_master.csv.
    This avoids showing stale/projection/full-season CSV totals in the Pro demo.
    """
    if batter_master is None or batter_master.empty or "PLAYER_ID" not in batter_master.columns:
        return pd.DataFrame()
    df = batter_master.copy()
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df = df[df["DATE"].notna()].copy()
    numeric_cols = [
        "AT_BATS", "HITS", "DOUBLES", "TRIPLES", "HOME_RUNS", "TOTAL_BASES",
        "RUNS", "RBI", "WALKS", "STRIKEOUTS", "STOLEN_BASES", "HBP", "SAC_FLIES", "PA",
    ]
    for c in numeric_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    group_cols = ["PLAYER_ID"]
    if "PLAYER_NAME" in df.columns:
        group_cols.append("PLAYER_NAME")
    if "TEAM" in df.columns:
        group_cols.append("TEAM")
    agg = {c: "sum" for c in numeric_cols}
    if "DATE" in df.columns:
        agg["DATE"] = "max"
    out = df.groupby(group_cols, dropna=False, as_index=False).agg(agg)
    out["AVG"] = np.where(out["AT_BATS"] > 0, out["HITS"] / out["AT_BATS"], np.nan)
    out["OBP"] = np.where(
        (out["AT_BATS"] + out["WALKS"] + out["HBP"] + out["SAC_FLIES"]) > 0,
        (out["HITS"] + out["WALKS"] + out["HBP"]) / (out["AT_BATS"] + out["WALKS"] + out["HBP"] + out["SAC_FLIES"]),
        np.nan,
    )
    out["SLG"] = np.where(out["AT_BATS"] > 0, out["TOTAL_BASES"] / out["AT_BATS"], np.nan)
    out["OPS"] = out["OBP"] + out["SLG"]
    out["HRR"] = out["HITS"] + out["RUNS"] + out["RBI"]
    sort_cols = [c for c in ["HOME_RUNS", "TOTAL_BASES", "HITS"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=False)
    return out.reset_index(drop=True)


def recompute_pitcher_season_totals_from_logs(pitcher_master: pd.DataFrame) -> pd.DataFrame:
    """Build pitcher season totals directly from mlb_pitcher_gamelogs_master.csv."""
    if pitcher_master is None or pitcher_master.empty or "PITCHER_ID" not in pitcher_master.columns:
        return pd.DataFrame()
    df = pitcher_master.copy()
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df = df[df["DATE"].notna()].copy()
    if "INNINGS" not in df.columns:
        df["INNINGS"] = 0
    df["IP_DEC"] = df["INNINGS"].apply(convert_baseball_ip)
    numeric_cols = [
        "STRIKEOUTS", "WALKS_ALLOWED", "HITS_ALLOWED", "HOME_RUNS_ALLOWED",
        "EARNED_RUNS", "BATTERS_FACED", "PITCH_COUNT",
    ]
    for c in numeric_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    group_cols = ["PITCHER_ID"]
    if "PITCHER_NAME" in df.columns:
        group_cols.append("PITCHER_NAME")
    if "TEAM" in df.columns:
        group_cols.append("TEAM")
    agg = {c: "sum" for c in numeric_cols}
    agg["IP_DEC"] = "sum"
    if "DATE" in df.columns:
        agg["DATE"] = "max"
    out = df.groupby(group_cols, dropna=False, as_index=False).agg(agg)
    out = out.rename(columns={"IP_DEC": "INNINGS"})
    out["K_PER_9"] = np.where(out["INNINGS"] > 0, out["STRIKEOUTS"] * 9 / out["INNINGS"], np.nan)
    out["BB_PER_9"] = np.where(out["INNINGS"] > 0, out["WALKS_ALLOWED"] * 9 / out["INNINGS"], np.nan)
    out["HR_PER_9"] = np.where(out["INNINGS"] > 0, out["HOME_RUNS_ALLOWED"] * 9 / out["INNINGS"], np.nan)
    out["ERA"] = np.where(out["INNINGS"] > 0, out["EARNED_RUNS"] * 9 / out["INNINGS"], np.nan)
    out = out.sort_values(["STRIKEOUTS", "INNINGS"], ascending=False).reset_index(drop=True)
    return out


def summarize_results_history(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty or "HIT" not in history.columns:
        return pd.DataFrame()
    df = history.copy()
    df["HIT"] = pd.to_numeric(df["HIT"], errors="coerce")
    if "DID_PLAY" in df.columns:
        did = df["DID_PLAY"].astype(str).str.upper().isin(["TRUE", "1", "YES", "Y"])
        df = df[did | (df["DID_PLAY"] == 1)].copy()
    df = df[df["HIT"].isin([0, 1])].copy()
    if df.empty or "MARKET_FAMILY" not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby("MARKET_FAMILY", dropna=False)
        .agg(bets=("HIT", "size"), hits=("HIT", "sum"), hit_rate=("HIT", "mean"))
        .reset_index()
        .sort_values(["hit_rate", "bets"], ascending=[False, False])
    )


def filter_player_rows(df: pd.DataFrame, query: str, name_cols: list[str] | None = None) -> pd.DataFrame:
    if df is None or df.empty or not query.strip():
        return df.copy() if df is not None else pd.DataFrame()
    q = query.strip().lower()
    name_cols = name_cols or ["PLAYER_NAME", "player", "PITCHER_NAME", "ENTITY_NAME"]
    mask = pd.Series(False, index=df.index)
    for c in name_cols:
        if c in df.columns:
            mask = mask | df[c].astype(str).str.lower().str.contains(q, na=False)
    return df[mask].copy()


def player_name_options(*frames: pd.DataFrame) -> list[str]:
    names = set()
    for df in frames:
        if df is None or df.empty:
            continue
        for c in ["PLAYER_NAME", "PITCHER_NAME", "ENTITY_NAME", "player"]:
            if c in df.columns:
                vals = df[c].dropna().astype(str).str.strip()
                names.update(v for v in vals if v)
    return [""] + sorted(names)


def render_player_profile(query: str, batter_season: pd.DataFrame, pitcher_season: pd.DataFrame, rolling_latest: pd.DataFrame, batter_master: pd.DataFrame, pitcher_master: pd.DataFrame, history: pd.DataFrame):
    if not query.strip():
        st.info("Search or select a player to view season totals, rolling trends, game logs, and model result history.")
        return

    q = query.strip()
    b_season = filter_player_rows(batter_season, q)
    p_season = filter_player_rows(pitcher_season, q)
    rolling = filter_player_rows(rolling_latest, q)
    b_logs = filter_player_rows(batter_master, q)
    p_logs = filter_player_rows(pitcher_master, q, ["PITCHER_NAME", "PLAYER_NAME"])
    hist = filter_player_rows(history, q)

    st.markdown(f"### Player search: {esc(q)}")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Batter log rows", f"{len(b_logs):,}")
    with c2:
        metric_card("Pitcher log rows", f"{len(p_logs):,}")
    with c3:
        metric_card("Rolling rows", f"{len(rolling):,}")
    with c4:
        metric_card("Result history rows", f"{len(hist):,}")

    # Show best available card summary first.
    if not b_season.empty:
        render_batter_cards("Player profile card", b_season, max_cards=3)
    elif not b_logs.empty:
        latest = b_logs.sort_values("DATE", ascending=False) if "DATE" in b_logs.columns else b_logs
        render_batter_cards("Player profile card", latest, max_cards=3)

    tab_a, tab_b, tab_c, tab_d, tab_e = st.tabs(["Season totals", "Rolling trends", "Batter game logs", "Pitcher game logs", "Result history"])
    with tab_a:
        show_df("Batter season total match", b_season, max_rows=50)
        show_df("Pitcher season total match", p_season, max_rows=50)
    with tab_b:
        render_player_trend_charts(q, b_logs, p_logs)
        show_df("Rolling trend match", rolling, max_rows=100)
    with tab_c:
        if not b_logs.empty and "DATE" in b_logs.columns:
            b_logs = b_logs.sort_values("DATE", ascending=False)
        show_df("Batter game log match", b_logs, max_rows=150)
    with tab_d:
        if not p_logs.empty and "DATE" in p_logs.columns:
            p_logs = p_logs.sort_values("DATE", ascending=False)
        show_df("Pitcher game log match", p_logs, max_rows=150)
    with tab_e:
        if not hist.empty and "DATE" in hist.columns:
            hist = hist.sort_values("DATE", ascending=False)
        show_df("Model result history match", hist, max_rows=200)




def render_player_trend_charts(player_query: str, b_logs: pd.DataFrame, p_logs: pd.DataFrame) -> None:
    st.markdown("#### Recent trend charts")
    shown = False
    if b_logs is not None and not b_logs.empty and "DATE" in b_logs.columns:
        b = b_logs.copy()
        b["DATE"] = pd.to_datetime(b["DATE"], errors="coerce")
        b = b.dropna(subset=["DATE"]).sort_values("DATE").tail(25)
        if not b.empty:
            for col in ["HITS", "TOTAL_BASES", "RUNS", "RBI", "WALKS", "STRIKEOUTS", "HOME_RUNS"]:
                if col in b.columns:
                    b[col] = pd.to_numeric(b[col], errors="coerce").fillna(0)
            if {"HITS", "RUNS", "RBI"}.issubset(b.columns):
                b["HRR"] = b["HITS"] + b["RUNS"] + b["RBI"]
            chart_cols = [c for c in ["HITS", "TOTAL_BASES", "HRR", "WALKS", "STRIKEOUTS", "HOME_RUNS"] if c in b.columns]
            if chart_cols:
                st.caption("Batter game-by-game trend - last 25 included games")
                st.line_chart(b.set_index("DATE")[chart_cols])
                shown = True
    if p_logs is not None and not p_logs.empty and "DATE" in p_logs.columns:
        p = p_logs.copy()
        p["DATE"] = pd.to_datetime(p["DATE"], errors="coerce")
        p = p.dropna(subset=["DATE"]).sort_values("DATE").tail(25)
        if not p.empty:
            for col in ["STRIKEOUTS", "PITCH_COUNT", "INNINGS", "WALKS_ALLOWED", "HITS_ALLOWED", "HOME_RUNS_ALLOWED"]:
                if col in p.columns:
                    p[col] = pd.to_numeric(p[col], errors="coerce").fillna(0)
            chart_cols = [c for c in ["STRIKEOUTS", "PITCH_COUNT", "INNINGS", "WALKS_ALLOWED", "HITS_ALLOWED", "HOME_RUNS_ALLOWED"] if c in p.columns]
            if chart_cols:
                st.caption("Pitcher game-by-game trend - last 25 included games")
                st.line_chart(p.set_index("DATE")[chart_cols])
                shown = True
    if not shown:
        st.info("No chartable game-log rows found for this player yet.")


def team_options_from_frames(*frames: pd.DataFrame) -> list[str]:
    teams = set()
    for df in frames:
        if df is None or df.empty:
            continue
        for c in ["TEAM", "HOME_TEAM", "AWAY_TEAM"]:
            if c in df.columns:
                teams.update(v for v in df[c].dropna().astype(str).str.strip() if v)
    return [""] + sorted(teams, key=lambda x: x.lower())


def filter_team_rows(df: pd.DataFrame, team: str) -> pd.DataFrame:
    if df is None or df.empty or not team.strip():
        return pd.DataFrame()
    t = team.strip().lower()
    mask = pd.Series(False, index=df.index)
    for c in ["TEAM", "HOME_TEAM", "AWAY_TEAM", "OPP"]:
        if c in df.columns:
            mask = mask | df[c].astype(str).str.lower().str.contains(re.escape(t), na=False)
    return df[mask].copy()


def render_team_page(batter_season: pd.DataFrame, pitcher_season: pd.DataFrame, batter_master: pd.DataFrame, pitcher_master: pd.DataFrame, history: pd.DataFrame, schedule: pd.DataFrame) -> None:
    st.markdown("### Team pages")
    st.caption("Search a team to view top hitters, pitchers, schedule context, and model result history from the included starter data.")
    c1, c2 = st.columns([2, 2])
    with c1:
        typed_team = st.text_input("Search team", placeholder="Example: New York Yankees, Dodgers, Arizona")
    with c2:
        team_choice = st.selectbox("Or choose team", team_options_from_frames(batter_season, pitcher_season, batter_master, pitcher_master, history, schedule), index=0, key="team_page_select")
    team = typed_team.strip() or team_choice.strip()
    if not team:
        st.info("Choose a team to open its team page.")
        return
    b = filter_team_rows(batter_season, team)
    p = filter_team_rows(pitcher_season, team)
    bl = filter_team_rows(batter_master, team)
    pl = filter_team_rows(pitcher_master, team)
    h = filter_team_rows(history, team)
    s = filter_team_rows(schedule, team)
    st.markdown(f"### Team search: {esc(team)}")
    m1, m2, m3, m4 = st.columns(4)
    with m1: metric_card("Batter rows", f"{len(b):,}")
    with m2: metric_card("Pitcher rows", f"{len(p):,}")
    with m3: metric_card("Game-log rows", f"{len(bl) + len(pl):,}")
    with m4: metric_card("Result rows", f"{len(h):,}")
    if not b.empty:
        sort_cols = [c for c in ["HOME_RUNS", "TOTAL_BASES", "HITS"] if c in b.columns]
        if sort_cols:
            b = b.sort_values(sort_cols, ascending=False)
        render_batter_cards("Top team hitter cards", b, max_cards=12)
    tabs = st.tabs(["Team summary", "Hitters", "Pitchers", "Recent game logs", "Model results", "Schedule"])
    with tabs[0]:
        if not bl.empty and "DATE" in bl.columns:
            tb = bl.copy()
            tb["DATE"] = pd.to_datetime(tb["DATE"], errors="coerce")
            for c in ["HITS", "TOTAL_BASES", "HOME_RUNS", "RUNS", "RBI", "WALKS", "STRIKEOUTS"]:
                if c in tb.columns:
                    tb[c] = pd.to_numeric(tb[c], errors="coerce").fillna(0)
            chart_cols = [c for c in ["HITS", "TOTAL_BASES", "HOME_RUNS", "RUNS", "RBI"] if c in tb.columns]
            if chart_cols:
                daily = tb.groupby("DATE", as_index=True)[chart_cols].sum().sort_index().tail(30)
                st.line_chart(daily)
        show_df("Team model summary", summarize_results_history(h), max_rows=50)
    with tabs[1]: show_df("Team hitters", b, max_rows=100)
    with tabs[2]: show_df("Team pitchers", p, max_rows=100)
    with tabs[3]:
        show_df("Team batter game logs", bl.sort_values("DATE", ascending=False) if not bl.empty and "DATE" in bl.columns else bl, max_rows=150)
        show_df("Team pitcher game logs", pl.sort_values("DATE", ascending=False) if not pl.empty and "DATE" in pl.columns else pl, max_rows=150)
    with tabs[4]: show_df("Team model result history", h.sort_values("DATE", ascending=False) if not h.empty and "DATE" in h.columns else h, max_rows=200)
    with tabs[5]:
        render_schedule_cards(s if not s.empty else schedule, max_cards=12)
        show_df("Team schedule rows", s, max_rows=50)


def _id_from_player_rows(df: pd.DataFrame, name: str, id_cols: list[str]) -> object:
    if df is None or df.empty or not name:
        return np.nan
    q = name.strip().lower()
    name_cols = [c for c in ["PLAYER_NAME", "PITCHER_NAME", "ENTITY_NAME", "player"] if c in df.columns]
    for nc in name_cols:
        rows = df[df[nc].astype(str).str.lower().eq(q)]
        if rows.empty:
            rows = df[df[nc].astype(str).str.lower().str.contains(re.escape(q), na=False)]
        if not rows.empty:
            for idc in id_cols:
                if idc in rows.columns and rows[idc].notna().any():
                    return rows[idc].dropna().iloc[0]
    return np.nan


def load_bvp_baseline() -> pd.DataFrame:
    """Load optional BvP baseline from the published app folder or bundled starter data."""
    candidates = [
        DATA_DIR / "mlb_bvp_baseline_master.csv",
        ROOT / "starter_data" / "data" / "mlb_bvp_baseline_master.csv",
        ROOT / "data_starter" / "mlb_bvp_baseline_master.csv",
    ]
    for p in candidates:
        df = load_csv(p)
        if not df.empty:
            df = df.copy()
            df.columns = [str(c).strip() for c in df.columns]
            return df
    return pd.DataFrame()


def _coalesce_first(row: pd.Series, cols: list[str], default=np.nan):
    for c in cols:
        if c in row.index and pd.notna(row.get(c)) and str(row.get(c)).strip() != "":
            return row.get(c)
    return default


def _bvp_id_columns(bvp: pd.DataFrame) -> tuple[str | None, str | None]:
    batter_cols = ["batter_id", "BATTER_ID", "PLAYER_ID", "ENTITY_ID"]
    pitcher_cols = ["pitcher_id", "PITCHER_ID", "OPP_PITCHER_ID", "MATCHUP_PITCHER_ID"]
    b_col = next((c for c in batter_cols if c in bvp.columns), None)
    p_col = next((c for c in pitcher_cols if c in bvp.columns), None)
    return b_col, p_col


def _bvp_name_columns(bvp: pd.DataFrame) -> tuple[str | None, str | None]:
    batter_cols = ["batter_name", "BATTER_NAME", "PLAYER_NAME", "player_name", "batter_name_norm"]
    pitcher_cols = ["pitcher_name", "PITCHER_NAME", "OPP_PITCHER_NAME", "pitcher_name_norm"]
    b_col = next((c for c in batter_cols if c in bvp.columns), None)
    p_col = next((c for c in pitcher_cols if c in bvp.columns), None)
    return b_col, p_col


def find_bvp_match(bvp: pd.DataFrame, batter: str, pitcher: str, bid, pid) -> pd.DataFrame:
    """Find exact BvP by IDs first, then fall back to normalized names."""
    if bvp is None or bvp.empty:
        return pd.DataFrame()
    d = bvp.copy()
    b_col, p_col = _bvp_id_columns(d)
    if b_col and p_col and pd.notna(bid) and pd.notna(pid):
        d[b_col] = pd.to_numeric(d[b_col], errors="coerce")
        d[p_col] = pd.to_numeric(d[p_col], errors="coerce")
        exact = d[(d[b_col] == float(bid)) & (d[p_col] == float(pid))].copy()
        if not exact.empty:
            return exact
    bn_col, pn_col = _bvp_name_columns(d)
    if bn_col and pn_col and batter and pitcher:
        bq = str(batter).strip().lower()
        pq = str(pitcher).strip().lower()
        d["_b_name"] = d[bn_col].astype(str).str.strip().str.lower()
        d["_p_name"] = d[pn_col].astype(str).str.strip().str.lower()
        exact = d[(d["_b_name"].eq(bq)) & (d["_p_name"].eq(pq))].copy()
        if exact.empty:
            exact = d[(d["_b_name"].str.contains(re.escape(bq), na=False)) & (d["_p_name"].str.contains(re.escape(pq), na=False))].copy()
        if not exact.empty:
            return exact.drop(columns=["_b_name", "_p_name"], errors="ignore")
    return pd.DataFrame()


def render_matchup_snapshot(batter: str, pitcher: str, bc: pd.DataFrame, pc: pd.DataFrame, b_logs: pd.DataFrame, p_logs: pd.DataFrame) -> None:
    """Always show a useful matchup summary even when exact BvP history is unavailable."""
    def num(frame, cols, default=np.nan):
        if frame is None or frame.empty:
            return default
        row = frame.iloc[0]
        for c in cols:
            if c in frame.columns and pd.notna(row.get(c)):
                return pd.to_numeric(pd.Series([row.get(c)]), errors="coerce").iloc[0]
        return default

    hitter_hr = num(bc, ["HOME_RUNS", "HR", "statcast_batter_hr"], 0)
    hitter_hits = num(bc, ["HITS", "hits", "statcast_batter_hits"], 0)
    hitter_tb = num(bc, ["TOTAL_BASES", "TB", "tb", "statcast_batter_tb"], 0)
    pitcher_k = num(pc, ["STRIKEOUTS", "so_recorded", "statcast_pitcher_so_recorded"], 0)
    pitcher_hr_allowed = num(pc, ["HOME_RUNS_ALLOWED", "hr_allowed", "statcast_pitcher_hr_allowed"], 0)
    pitcher_hits_allowed = num(pc, ["HITS_ALLOWED", "hits_allowed", "statcast_pitcher_hits_allowed"], 0)

    recent_h = pd.to_numeric(b_logs.get("HITS", pd.Series(dtype=float)), errors="coerce").tail(10).mean() if b_logs is not None and not b_logs.empty and "HITS" in b_logs.columns else np.nan
    recent_tb = pd.to_numeric(b_logs.get("TOTAL_BASES", b_logs.get("TB", pd.Series(dtype=float))), errors="coerce").tail(10).mean() if b_logs is not None and not b_logs.empty and ("TOTAL_BASES" in b_logs.columns or "TB" in b_logs.columns) else np.nan
    recent_pk = pd.to_numeric(p_logs.get("STRIKEOUTS", pd.Series(dtype=float)), errors="coerce").tail(10).mean() if p_logs is not None and not p_logs.empty and "STRIKEOUTS" in p_logs.columns else np.nan

    rows = [
        {"view": "Hitter season", "player": batter, "metric": "Hits", "value": hitter_hits},
        {"view": "Hitter season", "player": batter, "metric": "Total Bases", "value": hitter_tb},
        {"view": "Hitter season", "player": batter, "metric": "Home Runs", "value": hitter_hr},
        {"view": "Hitter recent", "player": batter, "metric": "Avg hits last 10", "value": recent_h},
        {"view": "Hitter recent", "player": batter, "metric": "Avg TB last 10", "value": recent_tb},
        {"view": "Pitcher season", "player": pitcher, "metric": "Strikeouts", "value": pitcher_k},
        {"view": "Pitcher season", "player": pitcher, "metric": "Hits Allowed", "value": pitcher_hits_allowed},
        {"view": "Pitcher season", "player": pitcher, "metric": "HR Allowed", "value": pitcher_hr_allowed},
        {"view": "Pitcher recent", "player": pitcher, "metric": "Avg K last 10", "value": recent_pk},
    ]
    snap = pd.DataFrame(rows)
    snap["value"] = pd.to_numeric(snap["value"], errors="coerce").round(3)
    show_df("Matchup snapshot from included season data", snap, max_rows=20)


def render_matchup_lab(batter_season: pd.DataFrame, pitcher_season: pd.DataFrame, batter_master: pd.DataFrame, pitcher_master: pd.DataFrame, rolling_latest: pd.DataFrame) -> None:
    st.markdown("### Matchup lab")
    st.caption("Pick a hitter and pitcher to review recent form, season profile, and optional BvP baseline if included.")
    batter_options = player_name_options(batter_season, batter_master)
    pitcher_options = player_name_options(pitcher_season, pitcher_master)
    c1, c2 = st.columns(2)
    with c1:
        batter = st.selectbox("Choose hitter", batter_options, index=0, key="matchup_batter")
    with c2:
        pitcher = st.selectbox("Choose pitcher", pitcher_options, index=0, key="matchup_pitcher")
    if not batter and not pitcher:
        st.info("Choose a hitter and pitcher to build a matchup card.")
        return
    bc = filter_player_rows(batter_season, batter) if batter else pd.DataFrame()
    pc = filter_player_rows(pitcher_season, pitcher, ["PITCHER_NAME", "PLAYER_NAME"]) if pitcher else pd.DataFrame()
    b_logs = filter_player_rows(batter_master, batter) if batter else pd.DataFrame()
    p_logs = filter_player_rows(pitcher_master, pitcher, ["PITCHER_NAME", "PLAYER_NAME"]) if pitcher else pd.DataFrame()
    c1, c2 = st.columns(2)
    with c1:
        if not bc.empty: render_batter_cards("Hitter profile", bc, max_cards=1)
        show_df("Hitter recent logs", b_logs.sort_values("DATE", ascending=False) if not b_logs.empty and "DATE" in b_logs.columns else b_logs, max_rows=10)
    with c2:
        if not pc.empty: show_df("Pitcher season profile", pc, max_rows=3)
        show_df("Pitcher recent logs", p_logs.sort_values("DATE", ascending=False) if not p_logs.empty and "DATE" in p_logs.columns else p_logs, max_rows=10)
    if batter and pitcher:
        bvp = load_bvp_baseline()
        bid = _id_from_player_rows(pd.concat([batter_season, batter_master, rolling_latest], ignore_index=True, sort=False), batter, ["PLAYER_ID", "ENTITY_ID", "batter_id"])
        pid = _id_from_player_rows(pd.concat([pitcher_season, pitcher_master, rolling_latest], ignore_index=True, sort=False), pitcher, ["PITCHER_ID", "PLAYER_ID", "ENTITY_ID", "pitcher_id"])
        st.markdown("#### Batter vs pitcher baseline")
        if not bvp.empty:
            match = find_bvp_match(bvp, batter, pitcher, bid, pid)
            if not match.empty:
                st.success("Exact BvP row found in the included baseline.")
                show_df("BvP match", match, max_rows=10)
            else:
                st.info("No exact historical BvP row found for this hitter/pitcher pair in the included starter data. This is normal for many matchups. The snapshot below still uses the included season logs and profiles.")
                render_matchup_snapshot(batter, pitcher, bc, pc, b_logs, p_logs)
        else:
            st.info("No BvP baseline file is included yet. To add true BvP rows, run build_bvp_baseline.py after installing pybaseball, or add data/mlb_bvp_baseline_master.csv to dashboard_publish/data.")
            render_matchup_snapshot(batter, pitcher, bc, pc, b_logs, p_logs)
        st.markdown("#### Matchup checklist")
        check = pd.DataFrame([
            {"factor": "Recent hitter form", "what_to_check": "Last 5/10 hits, total bases, HRR trend", "why_it_matters": "Models can lag sudden hot/cold streaks."},
            {"factor": "Pitcher strikeout profile", "what_to_check": "K trend, pitch count, innings trend", "why_it_matters": "K props depend on leash as much as skill."},
            {"factor": "Park/weather", "what_to_check": "Wind scalar, roof, elevation, park factor", "why_it_matters": "Run environment can shift hitter markets."},
            {"factor": "Line shopping", "what_to_check": "Compare 1-up/1-down lines", "why_it_matters": "A safer alternate can be better in a 2-leg or round robin."},
        ])
        show_df("Human review checklist", check, max_rows=10)


def render_model_builder() -> None:
    st.markdown("### Model builder")
    st.caption("Educational scoring sandbox. These are transparent starter weights buyers can edit as they learn. They are not guaranteed picks or financial advice.")
    st.markdown("#### Example feature weights")
    weights = pd.DataFrame([
        {"feature_group": "Baseline skill", "starter_weight": 0.25, "examples": "Season H rate, K rate, OPS, pitcher K rate"},
        {"feature_group": "Recent form", "starter_weight": 0.20, "examples": "Rolling 5/10/15 game metrics"},
        {"feature_group": "Matchup", "starter_weight": 0.20, "examples": "Opposing pitcher profile, BvP, handedness"},
        {"feature_group": "Market price", "starter_weight": 0.15, "examples": "Book implied probability, edge"},
        {"feature_group": "Context", "starter_weight": 0.10, "examples": "Park factor, elevation, roof, weather"},
        {"feature_group": "Risk controls", "starter_weight": 0.10, "examples": "Line type, alt distance, max odds, confidence tier"},
    ])
    show_df("Starter weight map", weights, max_rows=20)
    st.markdown("#### Simple scoring sandbox")
    c1, c2, c3 = st.columns(3)
    with c1:
        baseline = st.slider("Baseline skill", 0, 100, 60)
        recent = st.slider("Recent form", 0, 100, 55)
    with c2:
        matchup = st.slider("Matchup", 0, 100, 58)
        market = st.slider("Market value", 0, 100, 52)
    with c3:
        context = st.slider("Context", 0, 100, 50)
        risk = st.slider("Risk control", 0, 100, 65)
    score = 0.25*baseline + 0.20*recent + 0.20*matchup + 0.15*market + 0.10*context + 0.10*risk
    metric_card("Example model score", f"{score:.1f}/100")
    st.markdown("#### Suggested buyer workflow")
    show_df("Daily model workflow", pd.DataFrame([
        {"step": 1, "action": "Run daily pipeline", "goal": "Create predictions and pick boards."},
        {"step": 2, "action": "Review top markets", "goal": "Focus on HRR, K, and H before noisy markets."},
        {"step": 3, "action": "Use player/team/matchup pages", "goal": "Apply human review before betting."},
        {"step": 4, "action": "Track results", "goal": "Let accuracy decide what to adjust."},
        {"step": 5, "action": "Tune weights slowly", "goal": "Avoid overreacting to one bad day."},
    ]), max_rows=10)


st.markdown(
    """
    <div class="hero-card">
      <div class="hero-title">MLB Sports Model Dashboard</div>
      <div class="hero-sub">Professional Pro template — starter data, live schedule, player cards, team logos, headshots, results tracking, and accuracy views.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

run_dates = discover_run_dates()
selected_date = st.sidebar.selectbox("Run/results date", run_dates, index=len(run_dates) - 1)
selected_slate = st.sidebar.selectbox("Slate", SLATES, index=0)

st.sidebar.markdown("---")
st.sidebar.write("Common commands")
st.sidebar.code("python run_mlb_today.py --mode full --slate full", language="bash")
st.sidebar.code(f"python run_mlb_results.py --date {selected_date}", language="bash")

(tab_today, tab_season, tab_team, tab_matchup, tab_model, tab_picks, tab_parlays, tab_results, tab_accuracy, tab_files) = st.tabs([
    "Today", "Season Data", "Team Pages", "Matchups", "Model Builder", "Picks", "Parlays", "Results", "Accuracy", "Files"
])

with tab_today:
    preds = load_csv(DATA_DIR / "mlb_prop_predictions_today_with_ml.csv")
    schedule = load_schedule_for_today()
    context = load_csv(DATA_DIR / "mlb_game_context_today.csv")
    batter_master = load_csv(DATA_DIR / "mlb_batter_gamelogs_master.csv")
    history = load_csv(LOGS_DIR / "mlb_results_history.csv")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Today's prediction rows", f"{len(preds):,}" if not preds.empty else "0")
    with c2:
        metric_card("Today's games", f"{len(schedule):,}" if not schedule.empty else "0")
    with c3:
        metric_card("Starter batter logs", f"{len(batter_master):,}" if not batter_master.empty else "0")
    with c4:
        metric_card("Starter results rows", f"{len(history):,}" if not history.empty else "0")

    st.markdown("### Today's MLB Schedule")
    render_schedule_cards(schedule, max_cards=20)

    if preds.empty:
        st.info("No daily prediction file yet. That is normal immediately after setup. Run the daily pipeline to create data/mlb_prop_predictions_today_with_ml.csv. Until then, the demo player cards below use the latest included season game logs.")
        render_batter_cards("Latest included player results card board", latest_batter_sample(), max_cards=24)
    else:
        render_pick_cards("Today's scored prediction cards", preds.sort_values("MODEL_PROB_FINAL", ascending=False) if "MODEL_PROB_FINAL" in preds.columns else preds, max_cards=24)
        show_df("Today's scored predictions", preds)

    show_df("Game context", context)

with tab_season:
    st.markdown("### Starter season data included with Pro")
    st.caption("Season totals are recalculated from the included master game logs so demo totals reflect the actual included date range, not stale full-season/projection CSVs.")
    batter_master = load_csv(DATA_DIR / "mlb_batter_gamelogs_master.csv")
    pitcher_master = load_csv(DATA_DIR / "mlb_pitcher_gamelogs_master.csv")
    batter_season_file = load_csv(DATA_DIR / "mlb_batter_season_totals.csv")
    pitcher_season_file = load_csv(DATA_DIR / "mlb_pitcher_season_totals.csv")
    batter_season_calc = recompute_batter_season_totals_from_logs(batter_master)
    pitcher_season_calc = recompute_pitcher_season_totals_from_logs(pitcher_master)
    batter_season = batter_season_calc if not batter_season_calc.empty else batter_season_file
    pitcher_season = pitcher_season_calc if not pitcher_season_calc.empty else pitcher_season_file
    rolling_latest = load_csv(DATA_DIR / "mlb_rolling_latest.csv")
    history = load_csv(LOGS_DIR / "mlb_results_history.csv")
    result_summary = summarize_results_history(history)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Batter game logs", f"{len(batter_master):,}" if not batter_master.empty else "0")
    with c2:
        metric_card("Pitcher game logs", f"{len(pitcher_master):,}" if not pitcher_master.empty else "0")
    with c3:
        metric_card("Rolling rows", f"{len(rolling_latest):,}" if not rolling_latest.empty else "0")
    with c4:
        metric_card("Results history", f"{len(history):,}" if not history.empty else "0")

    render_batter_cards("Featured batter season cards", batter_season.sort_values(["HOME_RUNS", "TOTAL_BASES", "HITS"], ascending=False) if not batter_season.empty and "HOME_RUNS" in batter_season.columns else batter_season, max_cards=18)

    st.markdown("---")
    st.markdown("### Player lookup: season totals + trends")
    options = player_name_options(batter_season, pitcher_season, rolling_latest, batter_master, pitcher_master, history)
    col_search, col_select = st.columns([2, 2])
    with col_search:
        typed_player = st.text_input("Search player by name", placeholder="Example: Aaron Judge, Tarik Skubal, Shohei Ohtani")
    with col_select:
        selected_player_lookup = st.selectbox("Or choose from included players", options, index=0)
    player_query = typed_player.strip() or selected_player_lookup.strip()
    render_player_profile(player_query, batter_season, pitcher_season, rolling_latest, batter_master, pitcher_master, history)

    st.markdown("---")
    show_df("Results summary by market", result_summary, max_rows=50)
    show_df("Batter season totals", batter_season, max_rows=100)
    show_df("Pitcher season totals", pitcher_season, max_rows=100)
    show_df("Rolling latest", rolling_latest, max_rows=100)


with tab_team:
    batter_master = load_csv(DATA_DIR / "mlb_batter_gamelogs_master.csv")
    pitcher_master = load_csv(DATA_DIR / "mlb_pitcher_gamelogs_master.csv")
    batter_season = recompute_batter_season_totals_from_logs(batter_master)
    pitcher_season = recompute_pitcher_season_totals_from_logs(pitcher_master)
    history = load_csv(LOGS_DIR / "mlb_results_history.csv")
    schedule = load_schedule_for_today()
    render_team_page(batter_season, pitcher_season, batter_master, pitcher_master, history, schedule)

with tab_matchup:
    batter_master = load_csv(DATA_DIR / "mlb_batter_gamelogs_master.csv")
    pitcher_master = load_csv(DATA_DIR / "mlb_pitcher_gamelogs_master.csv")
    batter_season = recompute_batter_season_totals_from_logs(batter_master)
    pitcher_season = recompute_pitcher_season_totals_from_logs(pitcher_master)
    rolling_latest = load_csv(DATA_DIR / "mlb_rolling_latest.csv")
    render_matchup_lab(batter_season, pitcher_season, batter_master, pitcher_master, rolling_latest)

with tab_model:
    render_model_builder()

with tab_picks:
    best5_path = find_file("best5", selected_date, selected_slate)
    cand_path = find_file("daily_candidates", selected_date, selected_slate)
    best5_df = load_csv(best5_path) if best5_path else pd.DataFrame()
    cand_df = load_csv(cand_path) if cand_path else pd.DataFrame()

    render_pick_cards("Best 5 Pro card view", best5_df, max_cards=8)
    show_df("Best 5", best5_df)
    show_df("Daily candidates", cand_df)
    st.markdown("### Top boards")
    for market in TOP10_MARKETS:
        p = find_file("top10", selected_date, selected_slate, market)
        top_df = load_csv(p) if p else pd.DataFrame()
        render_pick_cards(f"Top cards - {market}", top_df, max_cards=6)
        show_df(f"Top 10 - {market}", top_df, max_rows=20)

with tab_parlays:
    parlay_files = {
        "Safe parlay": "mlb_safe_parlay",
        "Ladder parlay": "mlb_ladder_parlay",
        "Risky parlay": "mlb_risky_parlay",
        "Same-game parlays": "mlb_specific_game_parlays",
        "Best single per game": "mlb_best_single_per_game",
        "Best single parlay": "mlb_best_single_per_game_parlay",
        "Meta play card": "mlb_meta_play_card",
    }
    for label, prefix in parlay_files.items():
        p = find_file(prefix, selected_date, selected_slate)
        df_parlay = load_csv(p) if p else pd.DataFrame()
        render_parlay_card_board(label, df_parlay, label=label, max_cards=15 if "Same-game" in label or "Best single" in label else 6)
        with st.expander(f"Raw table: {label}", expanded=False):
            show_df(label, df_parlay, max_rows=100)

with tab_results:
    history = load_csv(LOGS_DIR / "mlb_results_history.csv")
    actuals = load_csv(DATA_DIR / f"mlb_results_actuals_{selected_date}.csv")
    if not history.empty and "DATE" in history.columns:
        day_history = history[history["DATE"].dt.strftime("%Y-%m-%d") == selected_date].copy()
    else:
        day_history = history
    render_results_cards("Selected date results cards", day_history.sort_values("HIT", ascending=False) if not day_history.empty and "HIT" in day_history.columns else day_history, max_cards=24)
    render_product_results_breakdown(selected_date, selected_slate)
    st.markdown("## Core results tables")
    show_df("Actual results", actuals)
    show_df("Results history for selected date", day_history, max_rows=300)
    show_df("All results history", history, max_rows=300)

with tab_accuracy:
    summary_path = ACCURACY_DIR / "mlb_accuracy_summary.txt"
    if summary_path.exists():
        st.subheader("Summary")
        st.code(summary_path.read_text(encoding="utf-8", errors="ignore"))
    accuracy_files = [
        "mlb_accuracy_by_market.csv",
        "mlb_accuracy_by_line_type.csv",
        "mlb_accuracy_by_confidence_tier.csv",
        "mlb_accuracy_by_edge_bucket.csv",
        "mlb_accuracy_by_model_prob_bucket.csv",
    ]
    for name in accuracy_files:
        show_df(name, load_csv(ACCURACY_DIR / name), max_rows=300)

with tab_files:
    st.subheader("Project file check")
    folders = [DATA_DIR, DAILY_OUT, RESULTS_DIR, LOGS_DIR, ACCURACY_DIR, Path("models"), Path("starter_data")]
    rows = []
    for folder in folders:
        if folder.exists():
            for path in folder.glob("*"):
                rows.append({
                    "folder": str(folder),
                    "file": path.name,
                    "size_kb": round(path.stat().st_size / 1024, 1),
                    "modified": pd.to_datetime(path.stat().st_mtime, unit="s"),
                })
    show_df("Available local files", pd.DataFrame(rows).sort_values("modified", ascending=False) if rows else pd.DataFrame(), max_rows=1000)
