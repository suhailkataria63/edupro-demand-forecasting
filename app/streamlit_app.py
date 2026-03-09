import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_data
from src.data_processing import create_targets, prepare_monthly_data
from src.feature_engineering import build_features
from src.model_utils import load_metadata, load_model

st.set_page_config(
    page_title="EduPro Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_DATA_PATH = os.getenv("EDUPRO_DATA_PATH", "data/raw/EduPro Online Platform.xlsx")
METADATA_FILE = "prediction_metadata.json"

LEVEL_MAP = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}


def apply_design_system():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

        :root {
            --bg: #070b12;
            --bg-soft: #0d1320;
            --panel: rgba(15, 21, 34, 0.82);
            --panel-strong: rgba(17, 25, 40, 0.94);
            --panel-2: rgba(21, 30, 48, 0.88);
            --text: #f3f7ff;
            --muted: #95a2b8;
            --line: rgba(255, 255, 255, 0.08);
            --line-strong: rgba(255, 255, 255, 0.14);
            --accent: #ff6a5c;
            --accent-2: #8b5cf6;
            --accent-3: #38bdf8;
            --success: #22c55e;
            --warning: #f59e0b;
            --shadow: 0 20px 50px rgba(0, 0, 0, 0.32);
            --radius-xl: 24px;
            --radius-lg: 18px;
            --radius-md: 14px;
            --radius-sm: 12px;
        }

        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
            background:
                radial-gradient(1200px 600px at 100% -10%, rgba(255, 106, 92, 0.18), transparent 60%),
                radial-gradient(900px 500px at -10% 0%, rgba(56, 189, 248, 0.10), transparent 58%),
                radial-gradient(700px 400px at 50% 100%, rgba(139, 92, 246, 0.08), transparent 60%),
                linear-gradient(180deg, #060910 0%, #0a0f19 100%);
            color: var(--text);
        }

        [data-testid="stHeader"] {
            background: rgba(0, 0, 0, 0);
        }

        [data-testid="stToolbar"] {
            right: 1rem;
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(7, 11, 18, 0.98) 0%, rgba(11, 16, 27, 0.98) 100%);
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] * {
            color: var(--text);
        }

        .block-container {
            max-width: 1380px;
            padding-top: 1.3rem;
            padding-bottom: 2.7rem;
        }

        h1, h2, h3, h4 {
            font-family: "Space Grotesk", sans-serif;
            color: var(--text);
            letter-spacing: -0.03em;
        }

        p, label, input, textarea, button {
            font-family: "Manrope", sans-serif;
        }

        span.material-symbols-rounded,
        span.material-symbols-outlined,
        i.material-icons,
        [class*="material-symbols"] {
            font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
            font-style: normal !important;
            font-weight: 400 !important;
            letter-spacing: normal !important;
            text-transform: none !important;
            white-space: nowrap !important;
            word-wrap: normal !important;
            -webkit-font-smoothing: antialiased !important;
        }

        .app-shell {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .hero-panel {
            position: relative;
            overflow: hidden;
            background:
                linear-gradient(135deg, rgba(18, 25, 39, 0.96) 0%, rgba(24, 18, 38, 0.92) 100%);
            border: 1px solid var(--line);
            border-radius: 26px;
            padding: 1.6rem 1.6rem 1.35rem 1.6rem;
            box-shadow: var(--shadow);
            margin-bottom: 1rem;
        }

        .hero-panel::before {
            content: "";
            position: absolute;
            inset: 0;
            background:
                radial-gradient(500px 220px at 85% 15%, rgba(255, 106, 92, 0.18), transparent 65%),
                radial-gradient(420px 220px at 15% 0%, rgba(56, 189, 248, 0.12), transparent 65%);
            pointer-events: none;
        }

        .hero-grid {
            position: relative;
            z-index: 1;
            display: grid;
            grid-template-columns: 1.45fr 1fr;
            gap: 1rem;
            align-items: stretch;
        }

        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.36rem 0.68rem;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.04);
            color: #ffb0a8;
            font-size: 0.72rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 0.8rem;
        }

        .hero-title {
            margin: 0;
            font-size: 2.55rem;
            line-height: 1.0;
            max-width: 700px;
        }

        .hero-subtitle {
            margin-top: 0.8rem;
            max-width: 720px;
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.65;
        }

        .hero-mini-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.75rem;
        }

        .hero-mini-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            min-height: 94px;
            backdrop-filter: blur(10px);
        }

        .hero-mini-label {
            color: var(--muted);
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
        }

        .hero-mini-value {
            margin-top: 0.38rem;
            color: #ffffff;
            font-family: "Space Grotesk", sans-serif;
            font-size: 1.38rem;
            font-weight: 700;
            line-height: 1.1;
        }

        .hero-mini-help {
            margin-top: 0.35rem;
            color: #aeb9ca;
            font-size: 0.78rem;
        }

        .section-title {
            margin-top: 0.55rem;
            margin-bottom: 0.3rem;
            font-size: 1.2rem;
        }

        .section-subtitle {
            color: var(--muted);
            margin-bottom: 0.9rem;
            font-size: 0.92rem;
        }

        .surface-card {
            background: linear-gradient(180deg, rgba(16, 22, 35, 0.90) 0%, rgba(11, 17, 27, 0.90) 100%);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1.15rem 1.1rem 0.95rem 1.1rem;
            box-shadow: var(--shadow);
            margin-bottom: 1rem;
        }

        .surface-card.slim {
            padding-top: 0.95rem;
        }

        .result-band {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.85rem;
            margin-bottom: 1rem;
        }

        .result-card {
            background:
                linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02)),
                linear-gradient(135deg, rgba(255, 106, 92, 0.06), rgba(139, 92, 246, 0.04));
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1rem 1rem 0.95rem 1rem;
        }

        .result-label {
            color: var(--muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 800;
        }

        .result-value {
            margin-top: 0.4rem;
            font-family: "Space Grotesk", sans-serif;
            font-size: 1.7rem;
            font-weight: 700;
            color: #ffffff;
        }

        .result-foot {
            margin-top: 0.28rem;
            color: #a8b4c7;
            font-size: 0.82rem;
        }

        [data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(17, 24, 39, 0.96), rgba(14, 19, 31, 0.96));
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 0.8rem 0.9rem;
        }

        [data-testid="stMetricLabel"] {
            color: var(--muted);
            font-weight: 600;
        }

        [data-testid="stMetricValue"] {
            color: #ffffff;
            font-size: 1.45rem;
            font-family: "Space Grotesk", sans-serif;
        }

        label[data-baseweb="radio"] {
            background: rgba(255,255,255,0.03);
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: 0.2rem 0.8rem 0.2rem 0.55rem;
        }

        [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        [data-testid="stMultiSelect"] div[data-baseweb="select"] > div,
        [data-testid="stNumberInput"] > div > div,
        [data-testid="stDateInput"] > div > div,
        [data-testid="stTextInput"] > div > div {
            background: rgba(20, 28, 43, 0.92);
            border: 1px solid var(--line);
            border-radius: 14px;
            color: var(--text);
        }

        [data-testid="stSelectbox"] div[data-baseweb="select"] svg,
        [data-testid="stMultiSelect"] div[data-baseweb="select"] svg,
        [data-testid="stDateInput"] svg {
            color: #b7c3d7;
        }

        input, textarea {
            color: var(--text) !important;
        }

        /* Hide Streamlit's occasional empty helper block shown above horizontal radios. */
        [data-testid="stRadio"] > div:first-child:empty {
            display: none !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.55rem;
            padding: 0.2rem;
            background: rgba(255,255,255,0.03);
            border: 1px solid var(--line);
            border-radius: 14px;
            margin-bottom: 0.95rem;
        }

        .stTabs [data-baseweb="tab"] {
            height: 2.55rem;
            border-radius: 10px;
            color: var(--muted);
            font-weight: 700;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(255,106,92,0.18), rgba(139,92,246,0.18));
            color: #ffffff;
        }

        div.stButton > button {
            width: 100%;
            border: 0;
            border-radius: 14px;
            min-height: 3rem;
            background: linear-gradient(135deg, #ff6a5c 0%, #ff4d6d 55%, #9b5cf6 100%);
            color: white;
            font-weight: 800;
            letter-spacing: 0.01em;
            box-shadow: 0 12px 28px rgba(255, 92, 107, 0.32);
            transition: transform 0.16s ease, filter 0.16s ease, box-shadow 0.16s ease;
        }

        div.stButton > button:hover {
            filter: brightness(1.04);
            transform: translateY(-1px);
            box-shadow: 0 16px 34px rgba(255, 92, 107, 0.38);
        }

        div.stButton > button:focus {
            outline: none;
            box-shadow: 0 0 0 0.2rem rgba(255, 106, 92, 0.20);
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 14px;
            overflow: hidden;
        }

        [data-testid="stPlotlyChart"] {
            border: 1px solid var(--line);
            border-radius: 16px;
            background: linear-gradient(180deg, rgba(18, 24, 38, 0.80), rgba(12, 18, 29, 0.82));
            padding: 0.2rem 0.2rem 0 0.2rem;
            overflow: visible !important;
        }

        [data-testid="stPlotlyChart"] > div {
            overflow: visible !important;
        }

        [data-testid="stExpander"] {
            border: 1px solid var(--line) !important;
            border-radius: 16px !important;
            background: rgba(255,255,255,0.02);
        }

        .sidebar-title {
            font-size: 1.05rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
            font-family: "Space Grotesk", sans-serif;
        }

        .sidebar-note {
            color: var(--muted);
            font-size: 0.85rem;
            line-height: 1.6;
            margin-bottom: 1rem;
        }

        .small-divider {
            height: 1px;
            width: 100%;
            background: var(--line);
            margin: 0.8rem 0 1rem 0;
        }

        @media (max-width: 1100px) {
            .hero-grid {
                grid-template-columns: 1fr;
            }

            .result-band {
                grid-template-columns: 1fr;
            }

            .hero-title {
                font-size: 2rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(ref: dict, data_path: str):
    st.markdown(
        f"""
        <div class="hero-panel">
            <div class="hero-grid">
                <div>
                    <div class="eyebrow">EduPro Forecasting Suite</div>
                    <h1 class="hero-title">Demand & Revenue Command Center</h1>
                    <div class="hero-subtitle">
                        A production-grade forecasting workspace for course demand, revenue projection,
                        and category performance benchmarking. Designed for analysts, operators,
                        and decision-makers who need clear inputs and high-signal outputs.
                    </div>
                </div>
                <div class="hero-mini-grid">
                    <div class="hero-mini-card">
                        <div class="hero-mini-label">Models Loaded</div>
                        <div class="hero-mini-value">3</div>
                        <div class="hero-mini-help">Enrollment, course revenue, category revenue</div>
                    </div>
                    <div class="hero-mini-card">
                        <div class="hero-mini-label">Known Courses</div>
                        <div class="hero-mini-value">{len(ref["course_ids"])}</div>
                        <div class="hero-mini-help">Reference profiles available for quick setup</div>
                    </div>
                    <div class="hero-mini-card">
                        <div class="hero-mini-label">Known Categories</div>
                        <div class="hero-mini-value">{len(ref["categories"])}</div>
                        <div class="hero-mini-help">Used for category encoding and benchmarking</div>
                    </div>
                    <div class="hero-mini-card">
                        <div class="hero-mini-label">Data Source</div>
                        <div class="hero-mini-value" style="font-size:1rem;">{Path(data_path).name}</div>
                        <div class="hero-mini-help">Reference dataset used for defaults and comparisons</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, subtitle: str):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def render_result_cards(predictions: dict):
    st.markdown(
        f"""
        <div class="result-band">
            <div class="result-card">
                <div class="result-label">Next-Month Enrollments</div>
                <div class="result-value">{predictions["pred_enrollment"]:.2f}</div>
                <div class="result-foot">Blended demand signal</div>
            </div>
            <div class="result-card">
                <div class="result-label">Course Revenue</div>
                <div class="result-value">${predictions["pred_course_revenue"]:.2f}</div>
                <div class="result-foot">Final blended revenue projection</div>
            </div>
            <div class="result-card">
                <div class="result-label">Category Revenue</div>
                <div class="result-value">${predictions["pred_category_revenue"]:.2f}</div>
                <div class="result-foot">Category-level model output</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


def _format_short_number(value: float) -> str:
    value = float(value)
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return f"{value:.2f}"


def _plotly_layout(title: str, height: int = 360) -> dict:
    return {
        "title": {"text": title, "x": 0.01, "xanchor": "left", "font": {"size": 17}},
        "height": height,
        "template": "plotly_dark",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "margin": {"l": 42, "r": 24, "t": 72, "b": 44},
        "font": {"family": "Manrope, sans-serif", "color": "#e7eefb", "size": 12},
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 11},
        },
        "xaxis": {
            "gridcolor": "rgba(255,255,255,0.08)",
            "zerolinecolor": "rgba(255,255,255,0.12)",
            "automargin": True,
        },
        "yaxis": {
            "gridcolor": "rgba(255,255,255,0.08)",
            "zerolinecolor": "rgba(255,255,255,0.12)",
            "automargin": True,
        },
    }


def _set_bar_chart_headroom(fig: go.Figure, values: List[float], y_title: str):
    safe_values = [float(_safe_float(v, 0.0)) for v in values]
    if not safe_values:
        fig.update_yaxes(title_text=y_title)
        return

    y_max = max(safe_values)
    y_min = min(safe_values)
    span = max(y_max - y_min, 1.0)

    upper_padding = max(span * 0.18, max(abs(y_max), 1.0) * 0.10)
    lower_padding = span * 0.04 if y_min < 0 else 0.0
    lower = min(0.0, y_min - lower_padding)
    upper = y_max + upper_padding

    fig.update_traces(cliponaxis=False, textposition="outside", selector={"type": "bar"})
    fig.update_yaxes(title_text=y_title, range=[lower, upper])


def build_enrollment_timeline_figure(feature_row: pd.DataFrame, predictions: dict) -> go.Figure:
    row = feature_row.iloc[0]
    labels = ["M-3", "M-2", "M-1", "M0", "M+1"]
    values = [
        _safe_float(row.get("Enrollment_lag3")),
        _safe_float(row.get("Enrollment_lag2")),
        _safe_float(row.get("Enrollment_lag1")),
        _safe_float(row.get("Enrollment_count")),
        _safe_float(predictions["pred_enrollment"]),
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=labels[:4],
            y=values[:4],
            mode="lines+markers",
            name="Observed",
            line={"color": "#38bdf8", "width": 3},
            marker={"size": 8, "color": "#38bdf8"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels[3:],
            y=values[3:],
            mode="lines+markers",
            name="Forecast Transition",
            line={"color": "#ff6a5c", "width": 3, "dash": "dot"},
            marker={"size": 9, "color": "#ff6a5c"},
        )
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            name="Level",
            marker={"color": "rgba(139, 92, 246, 0.28)", "line": {"width": 0}},
            opacity=0.45,
        )
    )
    fig.update_layout(**_plotly_layout("Enrollment Momentum to Forecast", height=360))
    fig.update_yaxes(title_text="Enrollments")
    fig.update_xaxes(title_text="Relative Month")
    return fig


def build_revenue_mix_figure(predictions: dict) -> go.Figure:
    chart_df = pd.DataFrame(
        {
            "Component": [
                "Direct Model",
                "Structural\n(Enrollments x Price)",
                "Final Blended",
            ],
            "Revenue": [
                _safe_float(predictions["pred_course_revenue_direct"]),
                _safe_float(predictions["pred_course_revenue_structural"]),
                _safe_float(predictions["pred_course_revenue"]),
            ],
            "Color": ["#8b5cf6", "#38bdf8", "#ff6a5c"],
        }
    )
    fig = go.Figure(
        go.Bar(
            x=chart_df["Component"],
            y=chart_df["Revenue"],
            marker={"color": chart_df["Color"], "line": {"color": "rgba(255,255,255,0.15)", "width": 1}},
            text=[f"${_format_short_number(v)}" for v in chart_df["Revenue"]],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_layout(**_plotly_layout("Revenue Composition", height=360))
    _set_bar_chart_headroom(fig, chart_df["Revenue"].tolist(), "Projected Revenue ($)")
    return fig


def build_feature_importance_figure(importance_df: pd.DataFrame, top_n: int) -> go.Figure:
    view = importance_df.head(top_n).copy()
    total = max(float(view["importance"].sum()), 1e-9)
    view["importance_pct"] = 100.0 * view["importance"] / total
    view = view.sort_values("importance_pct", ascending=True)

    fig = px.bar(
        view,
        x="importance_pct",
        y="feature",
        orientation="h",
        color="importance_pct",
        color_continuous_scale=[(0.0, "#38bdf8"), (0.5, "#8b5cf6"), (1.0, "#ff6a5c")],
        labels={"importance_pct": "Relative Importance (%)", "feature": ""},
    )
    fig.update_layout(**_plotly_layout("Top Feature Drivers", height=440))
    fig.update_coloraxes(showscale=False)
    fig.update_traces(
        marker_line_color="rgba(255,255,255,0.1)",
        marker_line_width=1,
        hovertemplate="%{y}<br>%{x:.2f}%<extra></extra>",
    )
    return fig


def build_category_benchmark_figure(
    predictions: dict, category_enrollment: float, median_course_demand: float
) -> go.Figure:
    benchmark_df = pd.DataFrame(
        {
            "Metric": [
                "Predicted Course Enrollment",
                "Category Current Enrollment",
                "Category Median Course Enrollment",
            ],
            "Value": [
                _safe_float(predictions["pred_enrollment"]),
                _safe_float(category_enrollment),
                _safe_float(median_course_demand),
            ],
            "Color": ["#ff6a5c", "#38bdf8", "#8b5cf6"],
        }
    )

    fig = go.Figure(
        go.Bar(
            x=benchmark_df["Metric"],
            y=benchmark_df["Value"],
            marker={"color": benchmark_df["Color"], "line": {"color": "rgba(255,255,255,0.15)", "width": 1}},
            text=[_format_short_number(v) for v in benchmark_df["Value"]],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_layout(**_plotly_layout("Category Demand Benchmark", height=360))
    _set_bar_chart_headroom(fig, benchmark_df["Value"].tolist(), "Enrollments")
    return fig


def build_share_gauge_figure(predicted_share_pct: float) -> go.Figure:
    clamped_share = max(0.0, min(100.0, float(predicted_share_pct)))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=clamped_share,
            number={"suffix": "%", "font": {"size": 38, "color": "#ffffff"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#9aa8bd"},
                "bar": {"color": "#ff6a5c", "thickness": 0.35},
                "bgcolor": "rgba(255,255,255,0.03)",
                "borderwidth": 1,
                "bordercolor": "rgba(255,255,255,0.16)",
                "steps": [
                    {"range": [0, 35], "color": "rgba(56, 189, 248, 0.22)"},
                    {"range": [35, 70], "color": "rgba(139, 92, 246, 0.22)"},
                    {"range": [70, 100], "color": "rgba(255, 106, 92, 0.22)"},
                ],
            },
            title={"text": "Predicted Share of Category", "font": {"size": 14}},
        )
    )
    fig.update_layout(**_plotly_layout("Category Share Gauge", height=360))
    fig.update_layout(margin={"l": 24, "r": 24, "t": 62, "b": 16})
    return fig


def _safe_float(value, default=0.0):
    try:
        numeric = float(value)
        if not np.isfinite(numeric):
            return float(default)
        return numeric
    except (TypeError, ValueError):
        return float(default)


def _clamp(value, min_value=None, max_value=None, default=0.0):
    numeric = _safe_float(value, default=default)
    if min_value is not None:
        numeric = max(float(min_value), numeric)
    if max_value is not None:
        numeric = min(float(max_value), numeric)
    return float(numeric)


def _fallback_metadata(models: Dict[str, object]) -> dict:
    return {
        "category_map": {},
        "blending": {
            "enrollment_blend_alpha": 1.0,
            "course_revenue_blend_alpha": 0.5,
        },
        "models": {
            name: {
                "feature_columns": list(getattr(model, "feature_names_in_", [])),
                "target_transform": "log1p",
            }
            for name, model in models.items()
        },
    }


@st.cache_resource
def load_artifacts():
    models = {
        "enrollment": load_model("xgboost_enrollment_model.pkl"),
        "course_revenue": load_model("xgboost_course_revenue_model.pkl"),
        "category_revenue": load_model("xgboost_category_revenue_model.pkl"),
    }

    metadata_error = None
    try:
        metadata = load_metadata(METADATA_FILE)
    except Exception as exc:
        metadata = _fallback_metadata(models)
        metadata_error = str(exc)

    return models, metadata, metadata_error


@st.cache_data
def load_reference_data(data_path: str):
    courses, teachers, transactions = load_data(data_path)
    monthly = prepare_monthly_data(courses, teachers, transactions)
    dataset = create_targets(monthly)
    features = build_features(dataset).dropna().copy()

    if features.empty:
        raise ValueError("Feature dataset is empty after preprocessing. Train data cannot be used for inference.")

    features = features.sort_values(["YearMonth", "CourseID"]).copy()
    course_latest = features.groupby("CourseID", as_index=False).last()
    category_latest = features.groupby("CourseCategory", as_index=False).last()
    numeric_defaults = features.select_dtypes(include=[np.number]).median().to_dict()

    categories = sorted(features["CourseCategory"].dropna().astype(str).unique().tolist())
    course_ids = sorted(features["CourseID"].dropna().astype(str).unique().tolist())

    return {
        "dataset": features,
        "course_latest": course_latest,
        "category_latest": category_latest,
        "numeric_defaults": numeric_defaults,
        "categories": categories,
        "course_ids": course_ids,
    }


def validate_artifact_contract(models: Dict[str, object], metadata: dict) -> List[str]:
    errors = []
    metadata_models = metadata.get("models", {})

    for model_name in ["enrollment", "course_revenue", "category_revenue"]:
        model = models.get(model_name)
        if model is None:
            errors.append(f"Missing model artifact: {model_name}")
            continue

        model_cols = list(getattr(model, "feature_names_in_", []))
        meta_cols = metadata_models.get(model_name, {}).get("feature_columns", [])

        if not model_cols:
            errors.append(f"Model `{model_name}` does not expose `feature_names_in_`.")
            continue

        if meta_cols and list(meta_cols) != list(model_cols):
            errors.append(
                f"Feature contract mismatch for `{model_name}` between model artifact and metadata."
            )

    return errors


def get_category_map(metadata: dict, categories: List[str]) -> Dict[str, int]:
    raw_map = metadata.get("category_map", {})
    cleaned = {}

    for key, value in raw_map.items():
        try:
            cleaned[str(key)] = int(value)
        except (TypeError, ValueError):
            continue

    if cleaned:
        return cleaned

    return {category: idx for idx, category in enumerate(sorted(categories))}


def build_feature_row(inputs: dict, category_map: Dict[str, int]) -> pd.DataFrame:
    target_period = inputs["target_period"]
    target_ts = target_period.to_timestamp()

    enrollment_count = _safe_float(inputs["enrollment_count"])
    revenue = _safe_float(inputs["revenue"])
    course_price = _safe_float(inputs["course_price"])
    course_duration = _safe_float(inputs["course_duration"])
    course_rating = _safe_float(inputs["course_rating"])
    category_enrollment = _safe_float(inputs["category_enrollment"])

    enrollment_lag1 = _safe_float(inputs["enrollment_lag1"])
    enrollment_lag2 = _safe_float(inputs["enrollment_lag2"])
    enrollment_lag3 = _safe_float(inputs["enrollment_lag3"])

    revenue_lag1 = _safe_float(inputs["revenue_lag1"])
    revenue_lag2 = _safe_float(inputs["revenue_lag2"])
    revenue_lag3 = _safe_float(inputs["revenue_lag3"])

    cat_enroll_lag1 = _safe_float(inputs["cat_enroll_lag1"])
    cat_enroll_lag2 = _safe_float(inputs["cat_enroll_lag2"])

    enroll_mean_3 = float(np.mean([enrollment_lag1, enrollment_lag2, enrollment_lag3]))
    rev_mean_3 = float(np.mean([revenue_lag1, revenue_lag2, revenue_lag3]))

    course_share = enrollment_count / category_enrollment if category_enrollment > 0 else 0.0
    course_share_lag1 = enrollment_lag1 / cat_enroll_lag1 if cat_enroll_lag1 > 0 else 0.0

    revenue_per_enroll_lag1 = revenue_lag1 / enrollment_lag1 if enrollment_lag1 > 0 else 0.0
    rev_per_enroll_mean_3 = rev_mean_3 / enroll_mean_3 if enroll_mean_3 > 0 else 0.0

    month = int(target_ts.month)

    row = {
        "Enrollment_count": enrollment_count,
        "Revenue": revenue,
        "CoursePrice": course_price,
        "CourseDuration": course_duration,
        "CourseRating": course_rating,
        "Category_Enrollment": category_enrollment,
        "Course_Share": course_share,
        "Course_Share_lag1": course_share_lag1,
        "course_level_enc": LEVEL_MAP.get(inputs["course_level"], -1),
        "instr_experience_bucket": _safe_float(inputs["instr_experience_bucket"]),
        "instr_rating_score": _safe_float(inputs["instr_rating_score"]),
        "instr_expertise_match": _safe_float(inputs["instr_expertise_match"]),
        "year": int(target_ts.year),
        "month": month,
        "quarter": int(target_ts.quarter),
        "is_q_start": int(month in [1, 4, 7, 10]),
        "Enrollment_lag1": enrollment_lag1,
        "Revenue_lag1": revenue_lag1,
        "Enrollment_lag2": enrollment_lag2,
        "Revenue_lag2": revenue_lag2,
        "Enrollment_lag3": enrollment_lag3,
        "Revenue_lag3": revenue_lag3,
        "Revenue_per_enrollment_lag1": revenue_per_enroll_lag1,
        "Enroll_mean_3": enroll_mean_3,
        "Rev_mean_3": rev_mean_3,
        "RevperEnroll_mean_3": rev_per_enroll_mean_3,
        "Enrollment_roll3": enroll_mean_3,
        "Revenue_roll3": rev_mean_3,
        "Enrollment_trend": enrollment_lag1 - enrollment_lag2,
        "Revenue_trend": revenue_lag1 - revenue_lag2,
        "cat_enroll_lag1": cat_enroll_lag1,
        "cat_enroll_lag2": cat_enroll_lag2,
        "CourseCategory_enc": float(category_map.get(inputs["course_category"], -1)),
    }

    return pd.DataFrame([row])


def validate_inputs(inputs: dict) -> Tuple[List[str], List[str]]:
    errors = []
    warnings = []

    non_negative_fields = [
        "course_price",
        "course_duration",
        "course_rating",
        "enrollment_count",
        "revenue",
        "enrollment_lag1",
        "enrollment_lag2",
        "enrollment_lag3",
        "revenue_lag1",
        "revenue_lag2",
        "revenue_lag3",
        "category_enrollment",
        "cat_enroll_lag1",
        "cat_enroll_lag2",
        "instr_experience_bucket",
        "instr_rating_score",
        "instr_expertise_match",
    ]

    for field in non_negative_fields:
        if _safe_float(inputs.get(field, 0.0), 0.0) < 0:
            errors.append(f"`{field}` cannot be negative.")

    if _safe_float(inputs["course_price"]) <= 0:
        errors.append("`course_price` must be greater than 0.")

    if not (0 <= _safe_float(inputs["course_rating"]) <= 5):
        errors.append("`course_rating` must be in [0, 5].")
    if not (0 <= _safe_float(inputs["instr_rating_score"]) <= 5):
        errors.append("`instr_rating_score` must be in [0, 5].")
    if not (0 <= _safe_float(inputs["instr_expertise_match"]) <= 1):
        errors.append("`instr_expertise_match` must be in [0, 1].")

    if _safe_float(inputs["category_enrollment"]) < _safe_float(inputs["enrollment_count"]):
        warnings.append("Category enrollment is lower than course enrollment for the same month.")

    if _safe_float(inputs["cat_enroll_lag1"]) < _safe_float(inputs["enrollment_lag1"]):
        warnings.append("Category lag1 enrollment is lower than course lag1 enrollment.")

    expected_revenue = _safe_float(inputs["enrollment_count"]) * _safe_float(inputs["course_price"])
    if expected_revenue > 0:
        observed_revenue = _safe_float(inputs["revenue"])
        relative_error = abs(observed_revenue - expected_revenue) / expected_revenue
        if relative_error > 0.5:
            warnings.append(
                "Current revenue differs significantly from enrollments x course price. Verify input units."
            )

    return errors, warnings


def predict_all(models: Dict[str, object], metadata: dict, feature_row: pd.DataFrame, inputs: dict):
    metadata_models = metadata.get("models", {})

    def _model_cols(model_name: str):
        cols = metadata_models.get(model_name, {}).get("feature_columns", [])
        if cols:
            return list(cols)
        return list(getattr(models[model_name], "feature_names_in_", []))

    enrollment_cols = _model_cols("enrollment")
    course_revenue_cols = _model_cols("course_revenue")
    category_revenue_cols = _model_cols("category_revenue")

    X_enroll = feature_row.reindex(columns=enrollment_cols, fill_value=0.0)
    X_course_rev = feature_row.reindex(columns=course_revenue_cols, fill_value=0.0)
    X_cat_rev = feature_row.reindex(columns=category_revenue_cols, fill_value=0.0)

    pred_enrollment_direct = float(np.expm1(models["enrollment"].predict(X_enroll)[0]))
    pred_course_revenue_direct = float(np.expm1(models["course_revenue"].predict(X_course_rev)[0]))
    pred_category_revenue = float(np.expm1(models["category_revenue"].predict(X_cat_rev)[0]))

    blending = metadata.get("blending", {})
    enrollment_blend_alpha = float(blending.get("enrollment_blend_alpha", 1.0))
    course_revenue_blend_alpha = float(blending.get("course_revenue_blend_alpha", 0.5))

    enrollment_lag1 = _safe_float(inputs["enrollment_lag1"])
    course_price = _safe_float(inputs["course_price"])

    pred_enrollment = enrollment_blend_alpha * pred_enrollment_direct + (1 - enrollment_blend_alpha) * enrollment_lag1
    pred_enrollment = float(max(0.0, pred_enrollment))

    pred_course_revenue_structural = pred_enrollment * course_price
    pred_course_revenue = (
        course_revenue_blend_alpha * pred_course_revenue_direct
        + (1 - course_revenue_blend_alpha) * pred_course_revenue_structural
    )
    pred_course_revenue = float(max(0.0, pred_course_revenue))
    pred_category_revenue = float(max(0.0, pred_category_revenue))

    return {
        "pred_enrollment": pred_enrollment,
        "pred_enrollment_direct": float(max(0.0, pred_enrollment_direct)),
        "pred_course_revenue": pred_course_revenue,
        "pred_course_revenue_direct": float(max(0.0, pred_course_revenue_direct)),
        "pred_course_revenue_structural": float(max(0.0, pred_course_revenue_structural)),
        "pred_category_revenue": pred_category_revenue,
        "enrollment_blend_alpha": enrollment_blend_alpha,
        "course_revenue_blend_alpha": course_revenue_blend_alpha,
    }


def resolve_index(options: List[str], value: str) -> int:
    if value in options:
        return options.index(value)
    return 0


def get_feature_importance_df(model, feature_columns: List[str]) -> pd.DataFrame:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return pd.DataFrame(columns=["feature", "importance"])
    df = pd.DataFrame(
        {"feature": feature_columns, "importance": np.array(importances, dtype=float)}
    ).sort_values("importance", ascending=False)
    return df


def main():
    apply_design_system()

    try:
        models, metadata, metadata_error = load_artifacts()
    except Exception as exc:
        st.error(f"Failed to load model artifacts: {exc}")
        st.stop()

    try:
        ref = load_reference_data(DEFAULT_DATA_PATH)
    except Exception as exc:
        st.error(f"Failed to load reference dataset from `{DEFAULT_DATA_PATH}`: {exc}")
        st.stop()

    contract_errors = validate_artifact_contract(models, metadata)
    if contract_errors:
        for err in contract_errors:
            st.error(err)
        st.stop()

    if metadata_error:
        st.warning(
            "Model metadata file was not found or unreadable. Using fallback metadata inferred from model files."
        )

    category_map = get_category_map(metadata, ref["categories"])

    with st.sidebar:
        st.markdown('<div class="sidebar-title">Control Center</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-note">Production workspace for course demand and revenue forecasting. Configure the profile, submit the historical signals, and generate a blended forecast.</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="small-divider"></div>', unsafe_allow_html=True)
        st.write(f"**Data path**  \n`{DEFAULT_DATA_PATH}`")
        st.write("**Models**  \nEnrollment, course revenue, category revenue")
        st.write(f"**Known categories**  \n{len(ref['categories'])}")
        st.write(f"**Known courses**  \n{len(ref['course_ids'])}")

    render_hero(ref, DEFAULT_DATA_PATH)

    course_latest_df = ref["course_latest"].copy()
    category_latest_df = ref["category_latest"].copy()

    course_profiles = {str(row["CourseID"]): row for _, row in course_latest_df.iterrows()}
    category_profiles = {str(row["CourseCategory"]): row for _, row in category_latest_df.iterrows()}

    render_section_header(
        "1 · Course Profile",
        "Define the commercial and instructional characteristics of the course to forecast.",
    )

    source_mode = st.segmented_control(
        "Course source",
        options=["Existing course", "New course"],
        default="Existing course",
        selection_mode="single",
    )
    if source_mode is None:
        source_mode = "Existing course"

    if source_mode == "Existing course":
        course_id = st.selectbox("Course ID", ref["course_ids"])
        course_profile = course_profiles.get(course_id)

        default_category = str(course_profile["CourseCategory"])
        default_price = _safe_float(course_profile.get("CoursePrice"), ref["numeric_defaults"].get("CoursePrice", 100.0))
        default_duration = _safe_float(
            course_profile.get("CourseDuration"), ref["numeric_defaults"].get("CourseDuration", 10.0)
        )
        default_rating = _safe_float(course_profile.get("CourseRating"), ref["numeric_defaults"].get("CourseRating", 4.0))
        default_level = str(course_profile.get("CourseLevel", "Intermediate"))
    else:
        course_id = st.text_input("Course ID", value="NEW_COURSE")
        default_category = ref["categories"][0]
        default_price = _safe_float(ref["numeric_defaults"].get("CoursePrice"), 100.0)
        default_duration = _safe_float(ref["numeric_defaults"].get("CourseDuration"), 10.0)
        default_rating = _safe_float(ref["numeric_defaults"].get("CourseRating"), 4.0)
        default_level = "Intermediate"

    default_price = _clamp(default_price, min_value=0.01, default=100.0)
    default_duration = _clamp(default_duration, min_value=0.1, default=10.0)
    default_rating = _clamp(default_rating, min_value=0.0, max_value=5.0, default=4.0)

    categories = ref["categories"]

    col_a, col_b = st.columns([1, 1])
    with col_a:
        course_category = st.selectbox("Course Category", categories, index=resolve_index(categories, default_category))
    with col_b:
        course_level = st.selectbox("Course Level", list(LEVEL_MAP.keys()), index=resolve_index(list(LEVEL_MAP.keys()), default_level))

    col_c, col_d, col_e = st.columns(3)
    with col_c:
        course_price = st.number_input("Course Price", min_value=0.01, value=float(default_price), step=1.0)
    with col_d:
        course_duration = st.number_input("Course Duration (hours)", min_value=0.1, value=float(default_duration), step=0.5)
    with col_e:
        course_rating = st.number_input("Course Rating", min_value=0.0, max_value=5.0, value=float(default_rating), step=0.1)

    col_f, col_g, col_h = st.columns(3)
    with col_f:
        instr_experience_bucket = st.number_input(
            "Instructor Experience Bucket",
            min_value=0.0,
            max_value=5.0,
            value=_clamp(ref["numeric_defaults"].get("instr_experience_bucket", 2.0), min_value=0.0, max_value=5.0, default=2.0),
            step=1.0,
        )
    with col_g:
        instr_rating_score = st.number_input(
            "Instructor Rating",
            min_value=0.0,
            max_value=5.0,
            value=_clamp(ref["numeric_defaults"].get("instr_rating_score", 4.0), min_value=0.0, max_value=5.0, default=4.0),
            step=0.1,
        )
    with col_h:
        instr_expertise_match = st.number_input(
            "Expertise Match",
            min_value=0.0,
            max_value=1.0,
            value=_clamp(ref["numeric_defaults"].get("instr_expertise_match", 0.8), min_value=0.0, max_value=1.0, default=0.8),
            step=0.05,
        )

    render_section_header(
        "2 · Historical Signals",
        "Provide recent enrollment, revenue, and category-level signals used to build the feature row.",
    )

    category_profile = category_profiles.get(course_category)
    course_profile = course_profiles.get(course_id)

    default_enrollment = _safe_float(
        (course_profile.get("Enrollment_count") if course_profile is not None else None),
        ref["numeric_defaults"].get("Enrollment_count", 10.0),
    )
    default_revenue = _safe_float(
        (course_profile.get("Revenue") if course_profile is not None else None),
        ref["numeric_defaults"].get("Revenue", default_enrollment * course_price),
    )

    default_enrollment_lag1 = _safe_float(
        (course_profile.get("Enrollment_lag1") if course_profile is not None else None),
        ref["numeric_defaults"].get("Enrollment_lag1", default_enrollment),
    )
    default_enrollment_lag2 = _safe_float(
        (course_profile.get("Enrollment_lag2") if course_profile is not None else None),
        ref["numeric_defaults"].get("Enrollment_lag2", default_enrollment_lag1),
    )
    default_enrollment_lag3 = _safe_float(
        (course_profile.get("Enrollment_lag3") if course_profile is not None else None),
        ref["numeric_defaults"].get("Enrollment_lag3", default_enrollment_lag2),
    )

    default_revenue_lag1 = _safe_float(
        (course_profile.get("Revenue_lag1") if course_profile is not None else None),
        ref["numeric_defaults"].get("Revenue_lag1", default_revenue),
    )
    default_revenue_lag2 = _safe_float(
        (course_profile.get("Revenue_lag2") if course_profile is not None else None),
        ref["numeric_defaults"].get("Revenue_lag2", default_revenue_lag1),
    )
    default_revenue_lag3 = _safe_float(
        (course_profile.get("Revenue_lag3") if course_profile is not None else None),
        ref["numeric_defaults"].get("Revenue_lag3", default_revenue_lag2),
    )

    default_category_enrollment = _safe_float(
        (category_profile.get("Category_Enrollment") if category_profile is not None else None),
        ref["numeric_defaults"].get("Category_Enrollment", max(default_enrollment, 100.0)),
    )
    default_cat_lag1 = _safe_float(
        (category_profile.get("cat_enroll_lag1") if category_profile is not None else None),
        ref["numeric_defaults"].get("cat_enroll_lag1", default_category_enrollment),
    )
    default_cat_lag2 = _safe_float(
        (category_profile.get("cat_enroll_lag2") if category_profile is not None else None),
        ref["numeric_defaults"].get("cat_enroll_lag2", default_cat_lag1),
    )

    top_left, top_right = st.columns([1.1, 1.9])
    with top_left:
        default_target_month = (pd.Timestamp.today() + pd.offsets.MonthBegin(1)).date()
        target_date = st.date_input("Forecast Month", value=default_target_month)
        target_period = pd.Period(pd.Timestamp(target_date), freq="M")
    with top_right:
        st.markdown("")
        st.info("Use values that reflect your latest operational reality. Large mismatches between price, enrollments, and revenue will trigger validation warnings.")

    col_1, col_2 = st.columns(2)
    with col_1:
        st.markdown("#### Enrollment Inputs")
        enrollment_count = st.number_input("Current Month Enrollments", min_value=0.0, value=_clamp(default_enrollment, min_value=0.0, default=10.0), step=1.0)
        enrollment_lag1 = st.number_input("Enrollments 1 Month Ago", min_value=0.0, value=_clamp(default_enrollment_lag1, min_value=0.0, default=10.0), step=1.0)
        enrollment_lag2 = st.number_input("Enrollments 2 Months Ago", min_value=0.0, value=_clamp(default_enrollment_lag2, min_value=0.0, default=10.0), step=1.0)
        enrollment_lag3 = st.number_input("Enrollments 3 Months Ago", min_value=0.0, value=_clamp(default_enrollment_lag3, min_value=0.0, default=10.0), step=1.0)

    with col_2:
        st.markdown("#### Revenue Inputs")
        revenue = st.number_input("Current Month Revenue", min_value=0.0, value=_clamp(default_revenue, min_value=0.0, default=1000.0), step=10.0)
        revenue_lag1 = st.number_input("Revenue 1 Month Ago", min_value=0.0, value=_clamp(default_revenue_lag1, min_value=0.0, default=1000.0), step=10.0)
        revenue_lag2 = st.number_input("Revenue 2 Months Ago", min_value=0.0, value=_clamp(default_revenue_lag2, min_value=0.0, default=1000.0), step=10.0)
        revenue_lag3 = st.number_input("Revenue 3 Months Ago", min_value=0.0, value=_clamp(default_revenue_lag3, min_value=0.0, default=1000.0), step=10.0)

    col_3, col_4 = st.columns(2)
    with col_3:
        st.markdown("#### Category Context")
        category_enrollment = st.number_input(
            "Category Total Enrollments (Current Month)",
            min_value=0.0,
            value=_clamp(default_category_enrollment, min_value=0.0, default=100.0),
            step=1.0,
        )
    with col_4:
        st.markdown("#### Category Momentum")
        cat_enroll_lag1 = st.number_input("Category Enrollments 1 Month Ago", min_value=0.0, value=_clamp(default_cat_lag1, min_value=0.0, default=100.0), step=1.0)
        cat_enroll_lag2 = st.number_input("Category Enrollments 2 Months Ago", min_value=0.0, value=_clamp(default_cat_lag2, min_value=0.0, default=100.0), step=1.0)

    inputs = {
        "course_id": course_id,
        "course_category": course_category,
        "course_level": course_level,
        "course_price": course_price,
        "course_duration": course_duration,
        "course_rating": course_rating,
        "instr_experience_bucket": instr_experience_bucket,
        "instr_rating_score": instr_rating_score,
        "instr_expertise_match": instr_expertise_match,
        "target_period": target_period,
        "enrollment_count": enrollment_count,
        "revenue": revenue,
        "enrollment_lag1": enrollment_lag1,
        "enrollment_lag2": enrollment_lag2,
        "enrollment_lag3": enrollment_lag3,
        "revenue_lag1": revenue_lag1,
        "revenue_lag2": revenue_lag2,
        "revenue_lag3": revenue_lag3,
        "category_enrollment": category_enrollment,
        "cat_enroll_lag1": cat_enroll_lag1,
        "cat_enroll_lag2": cat_enroll_lag2,
    }

    action_col_1, action_col_2, action_col_3 = st.columns([1.1, 1.5, 1.2])
    with action_col_2:
        if st.button("Generate Forecast", type="primary"):
            errors, warnings = validate_inputs(inputs)
            if errors:
                st.session_state.pop("forecast_payload", None)
                for err in errors:
                    st.error(err)
                st.stop()

            if course_category not in category_map:
                warnings.append(
                    "Selected category is missing in model category map. Encoding as -1 may reduce prediction quality."
                )

            try:
                feature_row = build_feature_row(inputs, category_map)
                predictions = predict_all(models, metadata, feature_row, inputs)
            except Exception as exc:
                st.session_state.pop("forecast_payload", None)
                st.error(f"Prediction failed: {exc}")
                st.stop()

            st.session_state["forecast_payload"] = {
                "predictions": predictions,
                "feature_row": feature_row.to_dict(orient="records"),
                "course_category": course_category,
                "category_enrollment": float(_safe_float(category_enrollment, 0.0)),
                "warnings": warnings,
            }

    forecast_payload = st.session_state.get("forecast_payload")
    if forecast_payload:
        predictions = forecast_payload["predictions"]
        feature_row = pd.DataFrame(forecast_payload["feature_row"])
        result_course_category = forecast_payload["course_category"]
        result_category_enrollment = float(_safe_float(forecast_payload["category_enrollment"], 0.0))

        for warn in forecast_payload.get("warnings", []):
            st.warning(warn)

        render_section_header(
            "3 · Forecast Results",
            "Review the predicted demand and revenue outputs, then inspect model composition and category context.",
        )
        render_result_cards(predictions)

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "Executive Summary",
                "Revenue View",
                "Feature Importance",
                "Category Benchmark",
            ]
        )

        with tab1:
            row = feature_row.iloc[0]
            last_enrollment = _safe_float(row.get("Enrollment_lag1"))
            current_enrollment = _safe_float(row.get("Enrollment_count"))
            forecast_enrollment = _safe_float(predictions["pred_enrollment"])

            change_vs_last = 100.0 * (forecast_enrollment - last_enrollment) / max(last_enrollment, 1.0)
            change_vs_current = 100.0 * (forecast_enrollment - current_enrollment) / max(current_enrollment, 1.0)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Direct Enrollment Model", f"{predictions['pred_enrollment_direct']:.2f}")
            m2.metric("Blend Alpha", f"{predictions['enrollment_blend_alpha']:.2f}")
            m3.metric("Forecast vs Last Month", f"{change_vs_last:.1f}%")
            m4.metric("Forecast vs Current", f"{change_vs_current:.1f}%")

            trend_col, compare_col = st.columns([1.65, 1.0])
            with trend_col:
                st.plotly_chart(
                    build_enrollment_timeline_figure(feature_row, predictions),
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                )
            with compare_col:
                comp_df = pd.DataFrame(
                    {
                        "Period": ["Current Month", "Next-Month Forecast"],
                        "Enrollments": [current_enrollment, forecast_enrollment],
                        "Color": ["#38bdf8", "#ff6a5c"],
                    }
                )
                comp_fig = go.Figure(
                    go.Bar(
                        x=comp_df["Period"],
                        y=comp_df["Enrollments"],
                        marker={"color": comp_df["Color"], "line": {"color": "rgba(255,255,255,0.15)", "width": 1}},
                        text=[_format_short_number(v) for v in comp_df["Enrollments"]],
                        textposition="outside",
                        cliponaxis=False,
                    )
                )
                comp_fig.update_layout(**_plotly_layout("Current vs Forecast", height=360))
                _set_bar_chart_headroom(comp_fig, comp_df["Enrollments"].tolist(), "Enrollments")
                st.plotly_chart(comp_fig, use_container_width=True, config=PLOTLY_CONFIG)

            st.caption("Final demand forecast blends direct model output with lag-driven baseline behavior.")

        with tab2:
            row = feature_row.iloc[0]
            rev_labels = ["M-3", "M-2", "M-1", "M0", "M+1"]
            rev_values = [
                _safe_float(row.get("Revenue_lag3")),
                _safe_float(row.get("Revenue_lag2")),
                _safe_float(row.get("Revenue_lag1")),
                _safe_float(row.get("Revenue")),
                _safe_float(predictions["pred_course_revenue"]),
            ]

            left_col, right_col = st.columns([1.35, 1.2])
            with left_col:
                st.plotly_chart(
                    build_revenue_mix_figure(predictions),
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                )
            with right_col:
                revenue_trend_fig = go.Figure()
                revenue_trend_fig.add_trace(
                    go.Scatter(
                        x=rev_labels[:4],
                        y=rev_values[:4],
                        mode="lines+markers",
                        name="Observed Revenue",
                        line={"color": "#38bdf8", "width": 3},
                        marker={"size": 8, "color": "#38bdf8"},
                    )
                )
                revenue_trend_fig.add_trace(
                    go.Scatter(
                        x=rev_labels[3:],
                        y=rev_values[3:],
                        mode="lines+markers",
                        name="Forecast Transition",
                        line={"color": "#ff6a5c", "width": 3, "dash": "dot"},
                        marker={"size": 9, "color": "#ff6a5c"},
                    )
                )
                revenue_trend_fig.update_layout(**_plotly_layout("Revenue Momentum to Forecast", height=360))
                revenue_trend_fig.update_yaxes(title_text="Revenue ($)")
                revenue_trend_fig.update_xaxes(title_text="Relative Month")
                st.plotly_chart(revenue_trend_fig, use_container_width=True, config=PLOTLY_CONFIG)

            alpha = _safe_float(predictions["course_revenue_blend_alpha"])
            st.metric("Revenue Blend Alpha", f"{alpha:.2f}", help="Weight assigned to direct revenue model.")

        with tab3:
            model_label_map = {
                "Enrollment Model": "enrollment",
                "Course Revenue Model": "course_revenue",
                "Category Revenue Model": "category_revenue",
            }
            selected_label = st.selectbox(
                "Model",
                list(model_label_map.keys()),
                index=0,
                key="feature_importance_model_select",
            )
            selected_model_key = model_label_map[selected_label]
            selected_model = models[selected_model_key]
            selected_features = metadata.get("models", {}).get(selected_model_key, {}).get(
                "feature_columns",
                list(getattr(selected_model, "feature_names_in_", [])),
            )
            importance_df = get_feature_importance_df(selected_model, selected_features)

            if importance_df.empty:
                st.info("Feature importance is unavailable for the selected model.")
            else:
                top_n = st.slider(
                    "Top features",
                    min_value=5,
                    max_value=min(30, len(importance_df)),
                    value=min(15, len(importance_df)),
                    step=1,
                    key="feature_importance_top_n",
                )
                top_imp = importance_df.head(top_n).copy()
                importance_sum = max(float(top_imp["importance"].sum()), 1e-9)
                top_imp["importance_pct"] = 100.0 * top_imp["importance"] / importance_sum
                top_imp["cumulative_pct"] = top_imp["importance_pct"].cumsum()

                fi_col_1, fi_col_2 = st.columns([1.4, 1.0])
                with fi_col_1:
                    st.plotly_chart(
                        build_feature_importance_figure(importance_df, top_n),
                        use_container_width=True,
                        config=PLOTLY_CONFIG,
                    )
                with fi_col_2:
                    cum_fig = go.Figure()
                    cum_fig.add_trace(
                        go.Scatter(
                            x=top_imp["feature"],
                            y=top_imp["cumulative_pct"],
                            mode="lines+markers",
                            line={"color": "#ff6a5c", "width": 3},
                            marker={"size": 7, "color": "#ff6a5c"},
                            fill="tozeroy",
                            fillcolor="rgba(255,106,92,0.16)",
                            name="Cumulative Coverage",
                        )
                    )
                    cum_fig.update_layout(**_plotly_layout("Cumulative Importance Coverage", height=440))
                    cum_fig.update_yaxes(title_text="Coverage (%)", range=[0, 100])
                    cum_fig.update_xaxes(title_text="Feature Rank", tickangle=-35)
                    st.plotly_chart(cum_fig, use_container_width=True, config=PLOTLY_CONFIG)

                table_view = top_imp[["feature", "importance", "importance_pct", "cumulative_pct"]].copy()
                table_view["importance"] = table_view["importance"].map(lambda x: f"{x:.6f}")
                table_view["importance_pct"] = table_view["importance_pct"].map(lambda x: f"{x:.2f}%")
                table_view["cumulative_pct"] = table_view["cumulative_pct"].map(lambda x: f"{x:.2f}%")
                st.dataframe(table_view, use_container_width=True, hide_index=True)

        with tab4:
            category_slice = ref["dataset"][ref["dataset"]["CourseCategory"] == result_course_category]
            median_course_demand = _safe_float(category_slice["Enrollment_count"].median(), 0.0)
            predicted_share_pct = (
                100 * predictions["pred_enrollment"] / max(_safe_float(result_category_enrollment, 0.0), 1.0)
            )

            bm_col_1, bm_col_2 = st.columns([1.35, 1.0])
            with bm_col_1:
                st.plotly_chart(
                    build_category_benchmark_figure(
                        predictions,
                        category_enrollment=result_category_enrollment,
                        median_course_demand=median_course_demand,
                    ),
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                )
            with bm_col_2:
                st.plotly_chart(
                    build_share_gauge_figure(predicted_share_pct),
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                )

            st.caption("Benchmark combines current category volume, historical median course demand, and forecasted course output.")

        with st.expander("Prediction Details"):
            st.write(
                {
                    "enrollment_direct_model": round(predictions["pred_enrollment_direct"], 4),
                    "enrollment_blend_alpha": predictions["enrollment_blend_alpha"],
                    "enrollment_final": round(predictions["pred_enrollment"], 4),
                    "course_revenue_direct_model": round(predictions["pred_course_revenue_direct"], 4),
                    "course_revenue_structural": round(predictions["pred_course_revenue_structural"], 4),
                    "course_revenue_blend_alpha": predictions["course_revenue_blend_alpha"],
                    "course_revenue_final": round(predictions["pred_course_revenue"], 4),
                    "category_revenue_predicted": round(predictions["pred_category_revenue"], 4),
                }
            )
            st.write("Feature row used for inference")
            st.dataframe(feature_row, use_container_width=True)
    else:
        st.info("Complete the profile and historical inputs, then click Generate Forecast to render the dashboard.")


if __name__ == "__main__":
    main()
