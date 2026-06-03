import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from supabase import create_client
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="EMF Risk Mapper", layout="wide")

custom_style = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@300;400;500&display=swap');

.stApp {
    background: linear-gradient(160deg, #f7f4ee 0%, #eef5ee 50%, #f2f5ed 100%);
    background-attachment: fixed;
    font-family: 'Inter', sans-serif;
}

[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
.main, .block-container {
    background: transparent !important;
}

html, body { background: transparent !important; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a2e1a 0%, #243524 100%) !important;
    border-right: 1px solid rgba(100,160,100,0.3);
}
[data-testid="stSidebar"] * { color: #d4e8d4 !important; }

/* FIX 1: Sidebar collapse/expand arrow → forest green (nuclear override) */
[data-testid="collapsedControl"],
[data-testid="collapsedControl"] *,
[data-testid="collapsedControl"] svg,
[data-testid="collapsedControl"] svg *,
[data-testid="collapsedControl"] svg path,
[data-testid="collapsedControl"] svg polyline,
[data-testid="collapsedControl"] svg line,
[data-testid="collapsedControl"] button,
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] *,
[data-testid="stSidebarCollapseButton"] svg,
[data-testid="stSidebarCollapseButton"] svg *,
[data-testid="stSidebarCollapseButton"] svg path,
[data-testid="stSidebarCollapseButton"] svg polyline,
button[data-testid="baseButton-header"],
button[data-testid="baseButton-header"] *,
button[data-testid="baseButton-header"] svg,
button[data-testid="baseButton-header"] svg path,
button[data-testid="baseButton-header"] svg polyline {
    color: #2e5c2e !important;
    fill: #2e5c2e !important;
    stroke: #2e5c2e !important;
    background: transparent !important;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 50px;
    font-weight: 700;
    color: #1e3d1e;
    letter-spacing: 0.5px;
    line-height: 1.1;
}
.hero-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    font-weight: 300;
    color: #5a7a5a;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-top: 6px;
}
.hero-divider {
    width: 70px;
    height: 2px;
    background: #5a8a5a;
    margin: 14px 0 26px 0;
    border-radius: 2px;
}

[data-testid="stMetric"] {
    background: rgba(255,255,252,0.88) !important;
    border: 1px solid rgba(90,138,90,0.22) !important;
    border-radius: 14px !important;
    padding: 18px !important;
    box-shadow: 0 3px 18px rgba(30,61,30,0.07) !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 11px !important;
    letter-spacing: 1.8px !important;
    text-transform: uppercase !important;
    color: #5a7a5a !important;
    font-weight: 500 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Playfair Display', serif !important;
    font-size: 30px !important;
    font-weight: 700 !important;
    color: #1e3d1e !important;
}
[data-testid="stMetricDelta"] {
    font-size: 12px !important;
    font-weight: 500 !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,252,0.7);
    border-radius: 10px;
    padding: 4px;
    border: 1px solid rgba(90,138,90,0.2);
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    letter-spacing: 0.4px;
    color: #5a7a5a;
    border-radius: 7px;
    padding: 8px 18px;
}
.stTabs [aria-selected="true"] {
    background: #2e5c2e !important;
    color: #f0f7f0 !important;
}

.alert-high {
    background: #fff5f5;
    border-left: 4px solid #c94a4a;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin: 10px 0;
}
.alert-moderate {
    background: #fdf8ee;
    border-left: 4px solid #b8881a;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin: 10px 0;
}
.alert-safe {
    background: #f0f8f0;
    border-left: 4px solid #2e7d4f;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin: 10px 0;
}
.alert-title {
    font-family: 'Playfair Display', serif;
    font-size: 18px;
    font-weight: 700;
    margin: 0 0 4px 0;
    color: #1e3d1e;
}
.alert-body {
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    color: #4a5a4a;
    margin: 0;
    line-height: 1.6;
}

.section-label {
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #5a8a5a;
    margin-bottom: 10px;
    font-weight: 500;
}

.stButton > button {
    background: #2e5c2e !important;
    color: #f0f7f0 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    letter-spacing: 0.4px !important;
    padding: 10px 22px !important;
}

.stSelectbox > div > div {
    background: rgba(255,255,252,0.9) !important;
    border: 1px solid rgba(90,138,90,0.3) !important;
    border-radius: 8px !important;
}

.stPlotlyChart {
    background: rgba(255,255,252,0.75) !important;
    border-radius: 14px !important;
    border: 1px solid rgba(90,138,90,0.14) !important;
    padding: 6px !important;
}

[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    border: 1px solid rgba(90,138,90,0.18) !important;
    overflow: hidden !important;
}

[data-testid="stToolbar"],
[data-testid="stToolbarActions"] button,
[data-testid="stToolbarActions"] svg,
[data-testid="stDecoration"],
header [data-testid="stToolbar"] svg path,
.stDeployButton svg,
button[kind="header"] svg {
    color: #111111 !important;
    fill: #111111 !important;
    stroke: #111111 !important;
}

[data-testid="stHeader"] {
    background: rgba(247,244,238,0.92) !important;
    border-bottom: 1px solid rgba(90,138,90,0.15);
}

/* Force header >> arrow and all header icon buttons to dark forest green */
[data-testid="stHeader"] button svg,
[data-testid="stHeader"] button svg *,
[data-testid="stHeader"] button svg path,
[data-testid="stHeader"] button svg polyline,
[data-testid="stHeader"] button svg line,
[data-testid="stHeader"] [data-testid="collapsedControl"] svg *,
[data-testid="stHeader"] [data-testid="collapsedControl"] svg path,
[data-testid="stHeader"] [data-testid="collapsedControl"] svg polyline {
    color: #2e5c2e !important;
    fill: #2e5c2e !important;
    stroke: #2e5c2e !important;
}
</style>
"""

st.markdown(custom_style, unsafe_allow_html=True)

PC = {
    "bg":      "rgba(0,0,0,0)",
    "paper":   "rgba(0,0,0,0)",
    "grid":    "rgba(90,138,90,0.13)",
    "text":    "#1e3d1e",
    "tick":    "#1e3d1e",
    "a1":      "#3a7a3a",
    "a2":      "#7aaa6a",
    "legend":  "rgba(255,255,252,0.92)",
    "leg_bdr": "rgba(90,138,90,0.3)",
}

def style_fig(fig, is_map=False):
    updates = dict(
        font=dict(family="Inter", color=PC["text"]),
        plot_bgcolor=PC["bg"],
        paper_bgcolor=PC["paper"],
        legend=dict(
            bgcolor=PC["legend"],
            bordercolor=PC["leg_bdr"],
            borderwidth=1,
            font=dict(color=PC["text"], size=11)
        ),
        margin=dict(l=50, r=50, t=50, b=50),
    )
    if not is_map:
        updates["xaxis"] = dict(
            gridcolor=PC["grid"],
            linecolor="rgba(90,138,90,0.2)",
            tickfont=dict(color=PC["tick"]),
            title_font=dict(color=PC["text"], size=12)
        )
        updates["yaxis"] = dict(
            gridcolor=PC["grid"],
            linecolor="rgba(90,138,90,0.2)",
            tickfont=dict(color=PC["tick"]),
            title_font=dict(color=PC["text"], size=12)
        )
    fig.update_layout(**updates)
    return fig


# --- SUPABASE ---
try:
    SUB_URL = st.secrets["SUPABASE_URL"]
    SUB_KEY = st.secrets["SUPABASE_KEY"]
    supabase = create_client(SUB_URL, SUB_KEY)
except Exception:
    st.error("Missing Secrets. Add SUPABASE_URL and SUPABASE_KEY in Streamlit settings.")
    supabase = None


def fetch_data():
    if not supabase:
        return pd.DataFrame()
    try:
        r = supabase.table("emf_readings").select("*").order("id", desc=True).limit(100).execute()
        return pd.DataFrame(r.data)
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()


def get_risk(val):
    if val > 50.0:
        return "HIGH RISK", "#c94a4a"
    if val >= 20.0:
        return "MODERATE", "#b8881a"
    return "SAFE", "#2e7d4f"


def render_alert(val, loc="Latest Reading"):
    label, _ = get_risk(val)
    who = 100
    pct = round((val / who) * 100, 1)
    if label == "HIGH RISK":
        cls = "alert-high"
        msg = f"EMF intensity of {val} uT exceeds safe thresholds. Prolonged exposure not recommended. Maintain distance from the source."
    elif label == "MODERATE":
        cls = "alert-moderate"
        msg = f"EMF intensity of {val} uT is elevated. Below WHO limits, but limit continuous exposure in this zone."
    else:
        cls = "alert-safe"
        msg = f"EMF intensity of {val} uT is within safe limits. No precautions required at {loc}."
    st.markdown(f"""
    <div class="{cls}">
        <p class="alert-title">{label} &mdash; {loc}</p>
        <p class="alert-body">{msg}<br>
        <strong>Intensity:</strong> {val} uT &nbsp;|&nbsp;
        <strong>WHO limit:</strong> {who} uT &nbsp;|&nbsp;
        <strong>% of limit:</strong> {pct}%
        </p>
    </div>
    """, unsafe_allow_html=True)


# ===================== SIDEBAR =====================
with st.sidebar:
    st.markdown("### EMF Risk Mapper")
    st.markdown("---")
    st.markdown("**Filters and Settings**")

    risk_filter = st.selectbox("Risk Level Filter", ["All", "HIGH RISK", "MODERATE", "SAFE"])

    alert_threshold = st.slider("Alert Threshold (uT)", min_value=5.0, max_value=100.0, value=20.0, step=5.0)

    auto_refresh = st.checkbox("Auto-refresh data", value=False)
    if auto_refresh:
        st.caption("Refresh active")

    st.markdown("---")
    st.markdown("**WHO EMF Guidelines**")
    st.markdown("""
    - Safe: below 20 uT  
    - Moderate: 20 to 50 uT  
    - High Risk: above 50 uT  
    - Public limit: 100 uT
    """)
    st.markdown("---")
    st.caption("EMF Risk Mapper v2.5 | Forest Edition")


# ===================== HEADER =====================
col_h, col_btn = st.columns([3, 1])
with col_h:
    st.markdown("""
    <div class="hero-title">EMF Risk Mapper</div>
    <div class="hero-subtitle">AI-Powered Electromagnetic Field Analysis</div>
    <div class="hero-divider"></div>
    """, unsafe_allow_html=True)
with col_btn:
    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()


# ===================== DATA & MODELING =====================
df = fetch_data()

model = None
poly = None

def generate_augmented_data(X_real, y_real, num_samples=150):
    np.random.seed(42)
    num_real = len(X_real)
    if num_real == 0:
        return X_real, y_real
    indices = np.random.choice(num_real, size=num_samples, replace=True)
    X_synth = X_real[indices]
    y_synth = y_real[indices]
    X_synth_noise = X_synth * np.random.uniform(0.97, 1.03, size=X_synth.shape)
    y_synth_noise = y_synth * np.random.uniform(0.95, 1.05, size=y_synth.shape)
    return X_synth_noise, y_synth_noise


if not df.empty:
    df["risk_level"] = df["intensity"].apply(lambda x: get_risk(x)[0])
    df_filtered = df[df["risk_level"] == risk_filter] if risk_filter != "All" else df.copy()

    max_dist = float(df["distance"].max()) if not df.empty else 10.0
    max_dist = max(max_dist, 0.5)

    df_grouped = df.copy()
    if max_dist > 10.0:
        df_grouped["distance_rounded"] = (df_grouped["distance"] / 2).round() * 2
    else:
        df_grouped["distance_rounded"] = df_grouped["distance"].round(3)

    df_aggregated = df_grouped.groupby("distance_rounded")["intensity"].mean().reset_index()
    df_aggregated.rename(columns={"distance_rounded": "distance"}, inplace=True)

    if len(df_aggregated) >= 5:
        X_real = df_aggregated[["distance"]].values
        y_real = df_aggregated["intensity"].values

        X_train, X_val, y_train, y_val = train_test_split(
            X_real, y_real, test_size=0.2, random_state=42
        )

        X_synth, y_synth = generate_augmented_data(X_train, y_train, num_samples=120)

        X_train_mixed = np.vstack([X_train, X_synth])
        y_train_mixed = np.concatenate([y_train, y_synth.flatten()])

        sample_weights = np.ones(len(X_train_mixed))
        sample_weights[:len(X_train)] = 12.0

        poly = PolynomialFeatures(degree=3)
        X_train_poly = poly.fit_transform(X_train_mixed)
        model = LinearRegression().fit(X_train_poly, y_train_mixed, sample_weight=sample_weights)

        X_val_poly = poly.transform(X_val)
        r2_val = model.score(X_val_poly, y_val)
        st.session_state["r2_val"] = r2_val
        st.sidebar.success(f"AI Model Trained!\nVal R² (Aggregated): {r2_val:.3f}")

    elif len(df_aggregated) >= 3:
        X = df_aggregated[["distance"]].values
        y = df_aggregated["intensity"].values
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)
        st.sidebar.info("Model trained on available data points.")


# ===================== MAIN UI =====================
if not df.empty:

    latest = df.iloc[0]
    label, _ = get_risk(latest["intensity"])

    st.markdown('<p class="section-label">Live Readings</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Intensity", f"{latest['intensity']} uT", delta=label, delta_color="off")
    c2.metric("Distance from Source", f"{latest['distance']} cm")
    c3.metric("Total Readings", len(df))
    high_ct = len(df[df["risk_level"] == "HIGH RISK"])
    c4.metric("High Risk Zones", high_ct, delta="Active" if high_ct > 0 else "None", delta_color="inverse" if high_ct > 0 else "normal")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<p class="section-label">Risk Alert</p>', unsafe_allow_html=True)
    render_alert(latest["intensity"])

    breaches = df[df["intensity"] > alert_threshold]
    if len(breaches) > 0 and alert_threshold != 20.0:
        st.markdown(f"""
        <div class="alert-moderate">
            <p class="alert-title">Custom Threshold Breach</p>
            <p class="alert-body">
                {len(breaches)} reading(s) exceed your threshold of <strong>{alert_threshold} uT</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Predictive Curve", "Heatmap", "Risk Zone Map", "Data"])

    # ---- TAB 1: PREDICTIVE CURVE ----
    with tab1:
        if model is not None:
            actual_max_dist = float(df["distance"].max()) if not df.empty else max_dist
            dist_range = np.linspace(0.0, actual_max_dist, 200).reshape(-1, 1)

            preds = model.predict(poly.transform(dist_range))
            preds = np.clip(preds, 0, None)

            y_max = float(max(df["intensity"].max(), max(preds))) + 10

            fig = go.Figure()

            # FIX 2: Zone fills — removed annotation_position to avoid right-side overlap
            fig.add_hrect(y0=0,   y1=20,    fillcolor="rgba(46,125,79,0.13)",  line_width=0)
            fig.add_hrect(y0=20,  y1=50,    fillcolor="rgba(184,136,26,0.13)", line_width=0)
            fig.add_hrect(y0=50,  y1=y_max, fillcolor="rgba(201,74,74,0.13)",  line_width=0)

            # FIX 2 cont.: Annotations pinned left, vertically centred in each band, colour-coded
            fig.add_annotation(
                x=0.01, xref="paper",
                y=10,
                yref="y",
                text=" Safe Zone",
                showarrow=False, xanchor="left",
                font=dict(color="#2e7d4f", size=11, family="Inter"),
                bgcolor="rgba(255,255,255,0.6)", borderpad=3,
            )
            fig.add_annotation(
                x=0.01, xref="paper",
                y=35,
                yref="y",
                text="Moderate Zone",
                showarrow=False, xanchor="left",
                font=dict(color="#b8881a", size=11, family="Inter"),
                bgcolor="rgba(255,255,255,0.6)", borderpad=3,
            )
            fig.add_annotation(
                x=0.01, xref="paper",
                y=max(60, y_max - (y_max - 50) * 0.25), yref="y",
                text="!! High Risk Zone",
                showarrow=False, xanchor="left",
                font=dict(color="#c94a4a", size=11, family="Inter"),
                bgcolor="rgba(255,255,255,0.6)", borderpad=3,
            )

            # Average sensor readings scatter
            fig.add_trace(go.Scatter(
                x=df_aggregated["distance"], y=df_aggregated["intensity"],
                mode="markers", name="Average Sensor Readings",
                marker=dict(color=PC["a1"], size=10, line=dict(color="white", width=1.5)),
            ))

            # AI fit curve
            fig.add_trace(go.Scatter(
                x=dist_range.flatten(), y=preds,
                name="AI Weighted Fit Curve",
                line=dict(color=PC["a2"], width=2.5),
            ))

            fig.update_layout(
                xaxis_title="Distance from Source (cm)",
                yaxis_title="EMF Intensity (uT)",
                height=420,
            )
            style_fig(fig)
            st.plotly_chart(fig, use_container_width=True)

            r2_to_show = st.session_state.get("r2_val", None)
            if r2_to_show is not None:
                if r2_to_show < 0:
                    st.warning(
                        "Model trained successfully, but the current validation data is not reliable enough to report a meaningful R² score."
                    )
                else:
                    st.markdown(f"""
                    <div style="background-color: rgba(90,138,90,0.08); border-left: 4px solid #2e5c2e; padding: 12px; border-radius: 8px; margin-top: 15px;">
                        <p style="margin: 0; font-family: 'Inter', sans-serif; font-size: 14px; color: #1e3d1e;">
                            <strong>AI Model Accuracy (R² Score):</strong>
                            <span style="font-weight:700; color:#2e5c2e;">{r2_to_show:.4f}</span> &nbsp;|&nbsp;
                            <em>This metric evaluates how accurately our prediction model matches your physical sensor data. A score closer to 1.00 indicates high fidelity.</em>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Additional data is required to calculate the mathematical prediction curve.")

    # ---- TAB 2: HEATMAP ----
    with tab2:
        if model is not None:
            actual_max_dist = float(df["distance"].max()) if not df.empty else max_dist
            x_grid = np.linspace(0.01, actual_max_dist, 60)
            y_grid = np.linspace(0, 2, 15)
            grid_int = model.predict(poly.transform(x_grid.reshape(-1, 1)))
            grid_int = np.clip(grid_int, 0, None)
            z_data = np.tile(grid_int, (len(y_grid), 1))

            fig_heat = px.imshow(
                z_data,
                x=np.round(x_grid, 2),
                y=np.round(y_grid, 2),
                labels=dict(x="Distance (cm)", y="Lateral Spread", color="uT"),
                color_continuous_scale=[
                    [0.0,  "#d4f0dc"],
                    [0.35, "#f5f0b0"],
                    [0.7,  "#f5c880"],
                    [1.0,  "#c94a4a"],
                ],
                aspect="auto",
            )

            # FIX 3: Heatmap — all text forced to dark #1e3d1e so it's readable
            fig_heat.update_layout(
                height=380,
                font=dict(family="Inter", color="#1e3d1e"),
                plot_bgcolor=PC["bg"],
                paper_bgcolor=PC["paper"],
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis=dict(
                    tickfont=dict(color="#1e3d1e", size=11),
                    title_font=dict(color="#1e3d1e", size=12),
                ),
                yaxis=dict(
                    tickfont=dict(color="#1e3d1e", size=11),
                    title_font=dict(color="#1e3d1e", size=12),
                ),
                coloraxis=dict(
                    colorbar=dict(
                        title=dict(text="uT", font=dict(color="#1e3d1e", size=12)),
                        tickfont=dict(color="#1e3d1e", size=11),
                        outlinecolor="rgba(90,138,90,0.3)",
                        outlinewidth=1,
                    )
                ),
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("Additional data points are needed to render the 2D exposure map.")

    # ---- TAB 3: RISK ZONE MAP ----
    with tab3:
        st.markdown('<p class="section-label">Geographic EMF Risk Zones</p>', unsafe_allow_html=True)

        has_geo = ("latitude" in df.columns and "longitude" in df.columns
                   and df[["latitude","longitude"]].notna().all().all())

        # FIX 4: Correct semantic risk colours — red=high, amber=moderate, green=safe
        RISK_COLORS = {
            "HIGH RISK": "#c94a4a",
            "MODERATE":  "#e8a020",
            "SAFE":       "#2e7d4f",
        }

        if has_geo:
            map_df = df_filtered.copy()
            map_df["size"] = map_df["intensity"].clip(1, 10) * 3
            map_df["hover_text"] = map_df.apply(
                lambda r: f"<b>{r['risk_level']}</b><br>Intensity: {r['intensity']} uT<br>Distance: {r['distance']} cm",
                axis=1
            )
            fig_map = go.Figure()
            for level, color in RISK_COLORS.items():
                sub = map_df[map_df["risk_level"] == level]
                if len(sub):
                    fig_map.add_trace(go.Scattermapbox(
                        lat=sub["latitude"], lon=sub["longitude"],
                        mode="markers",
                        marker=dict(size=sub["size"], color=color, opacity=0.85),
                        text=sub["hover_text"],
                        hoverinfo="text",
                        name=level,
                    ))
            fig_map.update_layout(
                mapbox=dict(
                    style="carto-positron", zoom=10,
                    center=dict(lat=df["latitude"].mean(), lon=df["longitude"].mean())
                ),
                legend=dict(
                    bgcolor="rgba(245,250,245,0.92)",
                    bordercolor="rgba(90,138,90,0.3)",
                    borderwidth=1,
                    font=dict(color="#1e3d1e", size=12)
                ),
                margin=dict(l=0, r=0, t=0, b=0),
                height=480,
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_map, use_container_width=True)

            st.markdown('<p class="section-label" style="margin-top:14px;">Zone Summary</p>', unsafe_allow_html=True)
            zc = st.columns(3)
            for i, (level, _) in enumerate(RISK_COLORS.items()):
                cnt = len(map_df[map_df["risk_level"] == level])
                zc[i].metric(level, f"{cnt} site(s)")

        else:
            st.info("No GPS coordinates found. Showing a demo risk zone map.")
            st.caption("Add latitude and longitude columns to your Supabase emf_readings table for real locations.")

            np.random.seed(42)
            demo_lat = 12.9716 + np.random.uniform(-0.05, 0.05, 15)
            demo_lon = 77.5946 + np.random.uniform(-0.05, 0.05, 15)
            demo_int = np.random.choice([1.0, 1.5, 2.5, 3.5, 4.0, 6.5, 7.2], 15)
            demo_df = pd.DataFrame({
                "latitude":   demo_lat,
                "longitude":  demo_lon,
                "intensity":  demo_int,
                "risk_level": [get_risk(v)[0] for v in demo_int],
            })
            demo_df["size"] = demo_df["intensity"].clip(1, 10) * 3

            fig_demo = go.Figure()
            for level, color in RISK_COLORS.items():
                sub = demo_df[demo_df["risk_level"] == level]
                if len(sub):
                    fig_demo.add_trace(go.Scattermapbox(
                        lat=sub["latitude"], lon=sub["longitude"],
                        mode="markers",
                        marker=dict(size=sub["size"], color=color, opacity=0.85),
                        name=f"{level} (demo)",
                        hovertemplate=f"<b>{level}</b><br>Intensity: %{{text}} uT<extra></extra>",
                        text=sub["intensity"].round(1).astype(str),
                    ))
            fig_demo.update_layout(
                mapbox=dict(style="carto-positron", zoom=11, center=dict(lat=12.9716, lon=77.5946)),
                legend=dict(
                    bgcolor="rgba(245,250,245,0.92)",
                    bordercolor="rgba(90,138,90,0.3)",
                    borderwidth=1,
                    font=dict(color="#1e3d1e", size=12)
                ),
                margin=dict(l=0, r=0, t=0, b=0),
                height=480,
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_demo, use_container_width=True)

    # ---- TAB 4: DATA ----
    with tab4:
        st.markdown('<p class="section-label">Raw Readings from Database</p>', unsafe_allow_html=True)

        cd, cf = st.columns([2, 1])
        with cd:
            csv = df_filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="emf_readings.csv",
                mime="text/csv",
            )
        with cf:
            st.caption(f"Showing {len(df_filtered)} of {len(df)} records")

        def color_risk(val):
            if val == "HIGH RISK":
                return "color: #c94a4a; font-weight: 600"
            if val == "MODERATE":
                return "color: #b8881a; font-weight: 600"
            if val == "SAFE":
                return "color: #2e7d4f; font-weight: 600"
            return ""

        if "risk_level" in df_filtered.columns:
            try:
                styled = df_filtered.style.map(color_risk, subset=["risk_level"])
            except AttributeError:
                styled = df_filtered.style.applymap(color_risk, subset=["risk_level"])
            st.dataframe(styled, use_container_width=True, height=360)
        else:
            st.dataframe(df_filtered, use_container_width=True, height=360)

else:
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px;">
        <div style="font-family:'Playfair Display',serif; font-size:28px; color:#1e3d1e; font-weight:700;">No Data Found</div>
        <div style="font-family:'Inter',sans-serif; font-size:14px; color:#5a7a5a; margin-top:8px; letter-spacing:1px;">
            The database appears empty or unreachable.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.info("Is Row Level Security (RLS) disabled in Supabase > Authentication > Policies?")
    st.warning("Verify SUPABASE_URL and SUPABASE_KEY are set in Streamlit Secrets.")
