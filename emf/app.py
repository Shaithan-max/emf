import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from supabase import create_client
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
 
# --- 1. CONFIGURATION ---
st.set_page_config(page_title="EMF Risk Mapper", layout="wide", page_icon="📡")
 
# --- ROSE GOLD & WHITE THEME ---
custom_style = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');
 
/* BASE BACKGROUND */
.stApp {
    background: linear-gradient(145deg, #fff5f5 0%, #fdf0f5 40%, #fef9f0 100%);
    background-attachment: fixed;
    font-family: 'DM Sans', sans-serif;
}
 
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
.main, .block-container {
    background: transparent !important;
}
 
html, body {
    background: transparent !important;
}
 
/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2c1a1a 0%, #3d2525 100%) !important;
    border-right: 1px solid rgba(188, 110, 110, 0.3);
}
[data-testid="stSidebar"] * {
    color: #f5d5d5 !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    color: #e8b8b8 !important;
    font-size: 13px;
    letter-spacing: 0.5px;
}
 
/* TITLE BLOCK */
.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 52px;
    font-weight: 700;
    color: #8b3a52;
    letter-spacing: 1px;
    line-height: 1.1;
}
 
.hero-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 15px;
    font-weight: 300;
    color: #b07070;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 4px;
}
 
.hero-divider {
    width: 80px;
    height: 2px;
    background: linear-gradient(90deg, #c9748a, #e8b4a0);
    margin: 16px 0 28px 0;
    border-radius: 2px;
}
 
/* METRIC CARDS */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.85) !important;
    border: 1px solid rgba(201, 116, 138, 0.25) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    box-shadow: 0 4px 24px rgba(139, 58, 82, 0.08) !important;
    backdrop-filter: blur(8px);
}
 
[data-testid="stMetricLabel"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: #b07070 !important;
    font-weight: 500 !important;
}
 
[data-testid="stMetricValue"] {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 32px !important;
    font-weight: 700 !important;
    color: #5a2030 !important;
}
 
[data-testid="stMetricDelta"] {
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
}
 
/* TABS */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255, 255, 255, 0.6);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid rgba(201, 116, 138, 0.2);
    gap: 4px;
}
 
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    letter-spacing: 0.5px;
    color: #b07070;
    border-radius: 8px;
    padding: 8px 20px;
}
 
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #c9748a, #e8b4a0) !important;
    color: white !important;
}
 
/* EXPANDER */
.streamlit-expanderHeader {
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #8b3a52 !important;
    background: rgba(255, 255, 255, 0.7) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(201, 116, 138, 0.25) !important;
}
 
/* ALERT BOXES */
.alert-high {
    background: linear-gradient(135deg, #fff0f0, #ffe0e0);
    border-left: 4px solid #c94a4a;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin: 12px 0;
}
 
.alert-moderate {
    background: linear-gradient(135deg, #fff8ee, #ffefd8);
    border-left: 4px solid #d4882a;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin: 12px 0;
}
 
.alert-safe {
    background: linear-gradient(135deg, #f0fff5, #d8f5e5);
    border-left: 4px solid #2a9d5c;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin: 12px 0;
}
 
.alert-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 20px;
    font-weight: 700;
    margin: 0 0 4px 0;
}
 
.alert-body {
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    color: #555;
    margin: 0;
    line-height: 1.6;
}
 
/* SECTION HEADERS */
.section-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #c9748a;
    margin-bottom: 12px;
    font-weight: 500;
}
 
/* INFO / WARNING overrides */
[data-testid="stInfo"] {
    background: rgba(232, 180, 160, 0.15) !important;
    border: 1px solid rgba(201, 116, 138, 0.3) !important;
    color: #7a3040 !important;
    border-radius: 12px !important;
}
 
[data-testid="stWarning"] {
    background: rgba(212, 136, 42, 0.1) !important;
    border: 1px solid rgba(212, 136, 42, 0.3) !important;
    border-radius: 12px !important;
}
 
/* BUTTONS */
.stButton > button {
    background: linear-gradient(135deg, #c9748a, #e8b4a0) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    letter-spacing: 0.5px !important;
    padding: 10px 24px !important;
    box-shadow: 0 4px 14px rgba(201, 116, 138, 0.35) !important;
    transition: all 0.2s !important;
}
 
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(201, 116, 138, 0.45) !important;
}
 
/* SELECTBOX & INPUTS */
.stSelectbox > div > div {
    background: rgba(255, 255, 255, 0.85) !important;
    border: 1px solid rgba(201, 116, 138, 0.3) !important;
    border-radius: 10px !important;
}
 
/* PLOTLY CHART CONTAINER */
.stPlotlyChart {
    background: rgba(255, 255, 255, 0.7) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(201, 116, 138, 0.15) !important;
    padding: 8px !important;
    box-shadow: 0 4px 24px rgba(139, 58, 82, 0.06) !important;
}
 
/* DATAFRAME */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    border: 1px solid rgba(201, 116, 138, 0.2) !important;
    overflow: hidden !important;
}
</style>
"""
 
st.markdown(custom_style, unsafe_allow_html=True)
 
# --- PLOTLY THEME ---
PLOTLY_COLORS = {
    "bg": "rgba(255,250,252,0)",
    "paper": "rgba(255,250,252,0)",
    "grid": "rgba(201,116,138,0.12)",
    "text": "#8b3a52",
    "accent1": "#c9748a",
    "accent2": "#e8b4a0",
    "accent3": "#d4a0b0",
}
 
def style_figure(fig):
    fig.update_layout(
        font=dict(family="DM Sans", color=PLOTLY_COLORS["text"]),
        plot_bgcolor=PLOTLY_COLORS["bg"],
        paper_bgcolor=PLOTLY_COLORS["paper"],
        xaxis=dict(gridcolor=PLOTLY_COLORS["grid"], linecolor="rgba(201,116,138,0.2)", tickfont=dict(color="#b07070")),
        yaxis=dict(gridcolor=PLOTLY_COLORS["grid"], linecolor="rgba(201,116,138,0.2)", tickfont=dict(color="#b07070")),
        legend=dict(bgcolor="rgba(255,255,255,0.7)", bordercolor="rgba(201,116,138,0.2)", borderwidth=1),
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig
 
 
# --- SUPABASE CONNECTION ---
try:
    SUB_URL = st.secrets["SUPABASE_URL"]
    SUB_KEY = st.secrets["SUPABASE_KEY"]
    supabase = create_client(SUB_URL, SUB_KEY)
except Exception as e:
    st.error("Missing Secrets! Add SUPABASE_URL and SUPABASE_KEY in Streamlit settings.")
    supabase = None
 
 
# --- DATA FETCH ---
def fetch_data():
    if not supabase:
        return pd.DataFrame()
    try:
        response = supabase.table("emf_readings").select("*").order("id", desc=True).limit(100).execute()
        return pd.DataFrame(response.data)
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()
 
 
# --- RISK HELPERS ---
def get_risk_level(val):
    if val > 5.0:
        return "HIGH RISK", "#c94a4a", "🔴"
    if val > 2.0:
        return "MODERATE", "#d4882a", "🟡"
    return "SAFE", "#2a9d5c", "🟢"
 
 
def render_alert(val, location_label="Current Location"):
    label, color, icon = get_risk_level(val)
    who_limit = 100  # µT WHO public limit
    icnirp_ref = 200  # µT ICNIRP reference
    percent_of_limit = round((val / who_limit) * 100, 1)
 
    if label == "HIGH RISK":
        css_cls = "alert-high"
        message = f"EMF intensity of {val} µT exceeds safe thresholds. Prolonged exposure not recommended. Maintain distance from the source."
    elif label == "MODERATE":
        css_cls = "alert-moderate"
        message = f"EMF intensity of {val} µT is elevated. Below WHO limits, but consider limiting continuous exposure near this zone."
    else:
        css_cls = "alert-safe"
        message = f"EMF intensity of {val} µT is within safe limits. No health precautions required at {location_label}."
 
    st.markdown(f"""
    <div class="{css_cls}">
        <p class="alert-title">{icon} {label} — {location_label}</p>
        <p class="alert-body">{message}<br>
        <strong>Intensity:</strong> {val} µT &nbsp;|&nbsp;
        <strong>WHO limit:</strong> {who_limit} µT &nbsp;|&nbsp;
        <strong>% of limit:</strong> {percent_of_limit}%
        </p>
    </div>
    """, unsafe_allow_html=True)
 
 
# ===================== SIDEBAR =====================
with st.sidebar:
    st.markdown("### 📡 EMF Risk Mapper")
    st.markdown("---")
    st.markdown("**Filters & Settings**")
 
    risk_filter = st.selectbox(
        "Risk Level Filter",
        ["All", "HIGH RISK", "MODERATE", "SAFE"],
    )
 
    alert_threshold = st.slider(
        "Alert Threshold (µT)",
        min_value=0.5, max_value=10.0, value=2.0, step=0.5,
    )
 
    auto_refresh = st.checkbox("Auto-refresh data", value=False)
    if auto_refresh:
        st.caption("Refreshing every 30s")
        st.session_state["refresh"] = True
 
    st.markdown("---")
    st.markdown("**WHO EMF Guidelines**")
    st.markdown("""
    - 🟢 Safe: < 2 µT  
    - 🟡 Moderate: 2–5 µT  
    - 🔴 High Risk: > 5 µT  
    - Limit: 100 µT (public)
    """)
    st.markdown("---")
    st.caption("AI EMF Risk Mapper v2.0  \n Rose Gold Edition")
 
 
# ===================== HEADER =====================
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.markdown("""
    <div class="hero-title">EMF Risk Mapper</div>
    <div class="hero-subtitle">AI-Powered Electromagnetic Field Analysis</div>
    <div class="hero-divider"></div>
    """, unsafe_allow_html=True)
with col_badge:
    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("🔄  Refresh Data"):
        st.cache_data.clear()
        st.rerun()
 
 
# ===================== FETCH =====================
df = fetch_data()
 
# Apply risk level column
if not df.empty:
    df["risk_level"] = df["intensity"].apply(lambda x: get_risk_level(x)[0])
 
    # Apply sidebar filter
    if risk_filter != "All":
        df_filtered = df[df["risk_level"] == risk_filter]
    else:
        df_filtered = df
 
 
# ===================== MAIN CONTENT =====================
if not df.empty:
 
    latest = df.iloc[0]
    label, color, icon = get_risk_level(latest["intensity"])
 
    # --- METRICS ROW ---
    st.markdown('<p class="section-label">Live Readings</p>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Intensity", f"{latest['intensity']} µT", delta=f"{icon} {label}", delta_color="off")
    col2.metric("Distance from Source", f"{latest['distance']} m")
    col3.metric("Total Readings", len(df))
    high_count = len(df[df["risk_level"] == "HIGH RISK"])
    col4.metric("High Risk Zones", high_count, delta="⚠️ Active" if high_count > 0 else "✅ None", delta_color="inverse" if high_count > 0 else "normal")
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # --- RISK ALERT NOTIFICATION ---
    st.markdown('<p class="section-label">Risk Alert</p>', unsafe_allow_html=True)
    render_alert(latest["intensity"], location_label="Latest Reading")
 
    # Alert for any readings exceeding user threshold
    threshold_breaches = df[df["intensity"] > alert_threshold]
    if len(threshold_breaches) > 0 and alert_threshold != 2.0:
        st.markdown(f"""
        <div class="alert-moderate">
            <p class="alert-title">⚠️ Custom Threshold Breach</p>
            <p class="alert-body">
                {len(threshold_breaches)} reading(s) exceed your custom alert threshold of <strong>{alert_threshold} µT</strong>.
                Review the data below for affected zones.
            </p>
        </div>
        """, unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["📈  Predictive Curve", "🌡️  Heatmap", "🗺️  Risk Zone Map", "📋  Data"])
 
    # ---- TAB 1: PREDICTIVE CURVE ----
    with tab1:
        if len(df) >= 3:
            X = df[["distance"]].values
            y = df["intensity"].values
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)
            model = LinearRegression().fit(X_poly, y)
 
            dist_range = np.linspace(0.1, 10, 200).reshape(-1, 1)
            preds = model.predict(poly.transform(dist_range))
 
            fig = go.Figure()
 
            # Risk bands
            fig.add_hrect(y0=0, y1=2, fillcolor="rgba(42,157,92,0.07)", line_width=0, annotation_text="Safe Zone", annotation_position="right")
            fig.add_hrect(y0=2, y1=5, fillcolor="rgba(212,136,42,0.07)", line_width=0, annotation_text="Moderate Zone", annotation_position="right")
            fig.add_hrect(y0=5, y1=max(max(preds), 6) + 1, fillcolor="rgba(201,74,74,0.07)", line_width=0, annotation_text="High Risk Zone", annotation_position="right")
 
            fig.add_trace(go.Scatter(
                x=df["distance"], y=df["intensity"],
                mode="markers",
                name="Recorded Readings",
                marker=dict(color=PLOTLY_COLORS["accent1"], size=10, line=dict(color="white", width=1.5)),
            ))
 
            fig.add_trace(go.Scatter(
                x=dist_range.flatten(), y=preds,
                name="AI Prediction",
                line=dict(color=PLOTLY_COLORS["accent2"], width=2.5, dash="solid"),
            ))
 
            fig.update_layout(
                xaxis_title="Distance from Source (m)",
                yaxis_title="EMF Intensity (µT)",
                height=420,
            )
            style_figure(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 3 readings to generate the AI prediction curve.")
 
    # ---- TAB 2: HEATMAP ----
    with tab2:
        if len(df) >= 3:
            x_grid = np.linspace(0.1, 5, 60)
            y_grid = np.linspace(0, 2, 15)
            grid_intensities = model.predict(poly.transform(x_grid.reshape(-1, 1)))
            z_data = np.tile(grid_intensities, (len(y_grid), 1))
 
            fig_heat = px.imshow(
                z_data, x=np.round(x_grid, 2), y=np.round(y_grid, 2),
                labels=dict(x="Distance (m)", y="Lateral Spread", color="µT"),
                color_continuous_scale=[
                    [0.0,  "#d4f5e5"],
                    [0.3,  "#f5e8b0"],
                    [0.6,  "#f0b8a0"],
                    [1.0,  "#c94a4a"],
                ],
                aspect="auto",
            )
            fig_heat.update_layout(
                height=380,
                coloraxis_colorbar=dict(
                    title="µT",
                    tickfont=dict(color=PLOTLY_COLORS["text"]),
                    titlefont=dict(color=PLOTLY_COLORS["text"]),
                ),
            )
            style_figure(fig_heat)
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("Need at least 3 readings to render the heatmap.")
 
    # ---- TAB 3: GEOGRAPHIC RISK ZONE MAP ----
    with tab3:
        st.markdown('<p class="section-label">Geographic EMF Risk Zones</p>', unsafe_allow_html=True)
 
        has_geo = "latitude" in df.columns and "longitude" in df.columns
 
        if has_geo and df[["latitude", "longitude"]].notna().all().all():
            map_df = df_filtered.copy()
            map_df["risk_color"] = map_df["risk_level"].map({
                "HIGH RISK": "#c94a4a",
                "MODERATE":  "#d4882a",
                "SAFE":       "#2a9d5c",
            })
            map_df["size"] = map_df["intensity"].clip(1, 10) * 3
            map_df["hover_text"] = map_df.apply(
                lambda r: f"<b>{r['risk_level']}</b><br>Intensity: {r['intensity']} µT<br>Distance: {r['distance']} m",
                axis=1
            )
 
            fig_map = go.Figure()
 
            for level, color in [("HIGH RISK", "#c94a4a"), ("MODERATE", "#d4882a"), ("SAFE", "#2a9d5c")]:
                sub = map_df[map_df["risk_level"] == level]
                if len(sub):
                    fig_map.add_trace(go.Scattermapbox(
                        lat=sub["latitude"],
                        lon=sub["longitude"],
                        mode="markers",
                        marker=dict(
                            size=sub["size"],
                            color=color,
                            opacity=0.82,
                        ),
                        text=sub["hover_text"],
                        hoverinfo="text",
                        name=level,
                    ))
 
            fig_map.update_layout(
                mapbox=dict(
                    style="carto-positron",
                    zoom=10,
                    center=dict(lat=df["latitude"].mean(), lon=df["longitude"].mean()),
                ),
                legend=dict(
                    bgcolor="rgba(255,250,252,0.9)",
                    bordercolor="rgba(201,116,138,0.3)",
                    borderwidth=1,
                    font=dict(color="#8b3a52"),
                ),
                margin=dict(l=0, r=0, t=0, b=0),
                height=480,
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_map, use_container_width=True)
 
            # Risk zone summary
            st.markdown('<p class="section-label" style="margin-top:16px;">Zone Summary</p>', unsafe_allow_html=True)
            zone_cols = st.columns(3)
            for i, (level, color, icon_str) in enumerate([
                ("HIGH RISK", "#c94a4a", "🔴"),
                ("MODERATE", "#d4882a", "🟡"),
                ("SAFE", "#2a9d5c", "🟢"),
            ]):
                count = len(map_df[map_df["risk_level"] == level])
                zone_cols[i].metric(f"{icon_str} {level}", f"{count} site(s)")
 
        else:
            # Demo map with simulated risk zones (no lat/lon in DB yet)
            st.info("Your readings don't have GPS coordinates yet. Showing a demo risk zone map.")
            st.caption("Add `latitude` and `longitude` columns to your Supabase `emf_readings` table to show real locations.")
 
            # Simulate sample coordinates around a city center
            np.random.seed(42)
            demo_lat = 28.6139 + np.random.uniform(-0.05, 0.05, 15)
            demo_lon = 77.2090 + np.random.uniform(-0.05, 0.05, 15)
            demo_intensity = np.random.choice([1.0, 1.5, 2.5, 3.5, 4.0, 6.5, 7.2], 15)
            demo_df = pd.DataFrame({
                "latitude": demo_lat,
                "longitude": demo_lon,
                "intensity": demo_intensity,
                "risk_level": [get_risk_level(v)[0] for v in demo_intensity],
            })
            demo_df["size"] = demo_df["intensity"].clip(1, 10) * 3
 
            fig_demo = go.Figure()
            for level, color in [("HIGH RISK", "#c94a4a"), ("MODERATE", "#d4882a"), ("SAFE", "#2a9d5c")]:
                sub = demo_df[demo_df["risk_level"] == level]
                if len(sub):
                    fig_demo.add_trace(go.Scattermapbox(
                        lat=sub["latitude"], lon=sub["longitude"],
                        mode="markers",
                        marker=dict(size=sub["size"], color=color, opacity=0.8),
                        name=f"{level} (demo)",
                        hovertemplate=f"<b>{level}</b><br>Intensity: %{{text}} µT<extra></extra>",
                        text=sub["intensity"].round(1).astype(str),
                    ))
 
            fig_demo.update_layout(
                mapbox=dict(style="carto-positron", zoom=11, center=dict(lat=28.6139, lon=77.2090)),
                legend=dict(bgcolor="rgba(255,250,252,0.9)", bordercolor="rgba(201,116,138,0.3)", borderwidth=1, font=dict(color="#8b3a52")),
                margin=dict(l=0, r=0, t=0, b=0),
                height=480,
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_demo, use_container_width=True)
 
    # ---- TAB 4: RAW DATA ----
    with tab4:
        st.markdown('<p class="section-label">Raw Readings from Database</p>', unsafe_allow_html=True)
 
        col_dl, col_flt = st.columns([2, 1])
        with col_dl:
            csv = df_filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️  Download CSV",
                data=csv,
                file_name="emf_readings.csv",
                mime="text/csv",
            )
        with col_flt:
            st.caption(f"Showing {len(df_filtered)} of {len(df)} records")
 
        st.dataframe(
            df_filtered.style.applymap(
                lambda v: f"color: #c94a4a; font-weight:600" if v == "HIGH RISK"
                else (f"color: #d4882a; font-weight:600" if v == "MODERATE" else f"color: #2a9d5c; font-weight:600"),
                subset=["risk_level"] if "risk_level" in df_filtered.columns else []
            ),
            use_container_width=True,
            height=360,
        )
 
else:
    # Empty state
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px;">
        <div style="font-size:48px; margin-bottom:16px;">📡</div>
        <div style="font-family:'Cormorant Garamond',serif; font-size:28px; color:#8b3a52; font-weight:700;">No Data Found</div>
        <div style="font-family:'DM Sans',sans-serif; font-size:14px; color:#b07070; margin-top:8px; letter-spacing:1px;">
            The database appears empty or unreachable.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.info("Check: Is Row Level Security (RLS) disabled in Supabase → Authentication → Policies?")
    st.warning("Also verify SUPABASE_URL and SUPABASE_KEY are correctly set in Streamlit Secrets.")
