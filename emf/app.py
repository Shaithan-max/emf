import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from supabase import create_client
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="AI EMF Risk Mapper", layout="wide")

# Fetch credentials from Streamlit Secrets
# Ensure these are set in Streamlit Cloud > Settings > Secrets
try:
    SUB_URL = st.secrets["SUPABASE_URL"]
    SUB_KEY = st.secrets["SUPABASE_KEY"]
    supabase = create_client(SUB_URL, SUB_KEY)
except Exception as e:
    st.error("Missing Secrets! Make sure SUPABASE_URL and SUPABASE_KEY are in Streamlit settings.")

# --- 2. DATA FETCHING ---
def fetch_data():
    try:
        # UPDATED: Table name is now emf_readings
        response = supabase.table("emf_readings").select("*").order("id", desc=True).limit(100).execute()
        return pd.DataFrame(response.data)
    except Exception as e:
        # This will show us if the error is RLS, Table Name, or Connection
        st.error(f"Internal Database Error: {e}")
        return pd.DataFrame()

# --- 3. UI LAYOUT ---
custom_style = """
<style>
    /* Glowing Title */
    .glowing-title {
        font-size: 40px;
        font-weight: 900;
        color: #00FFAA;
        text-align: center;
        text-shadow: 0px 0px 15px rgba(0, 255, 170, 0.6);
        margin-bottom: 30px;
    }
    
    /* Styling the 3 Data Boxes */
    div[data-testid="metric-container"] {
        background-color: #0B1426;
        border: 2px solid #00FFAA;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0px 0px 15px rgba(0, 255, 170, 0.2);
    }
</style>
"""
st.markdown(custom_style, unsafe_allow_html=True)
st.markdown("<div class='glowing-title'>⚡ AI-Based EMF Risk Mapper</div>", unsafe_allow_html=True)

df = fetch_data()

# --- 4. LOGIC ---
if not df.empty:
    # Show the raw data table for a second to verify it's working
    with st.expander("View Raw Data from Supabase"):
        st.write(df)

    latest = df.iloc[0]
    col1, col2, col3 = st.columns(3)
    
    # Risk Logic
    def get_risk_label(val):
        if val > 5.0: return "HIGH RISK", "inverse"
        if val > 2.0: return "MODERATE", "off"
        return "SAFE", "normal"

    label, color = get_risk_label(latest['intensity'])
    
    col1.metric("Current Intensity", f"{latest['intensity']} µT", delta=label, delta_color=color)
    col2.metric("Distance from Line", f"{latest['distance']} m")
    col3.metric("Total Samples", len(df))

    # AI Training (Requires at least 3 points to draw a curve)
    if len(df) >= 3:
        X = df[['distance']].values
        y = df['intensity'].values
        
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)

        tab1, tab2 = st.tabs(["Predictive Curve", "🌡️ Heatmap"])

        with tab1:
            dist_range = np.linspace(0.1, 10, 100).reshape(-1, 1)
            preds = model.predict(poly.transform(dist_range))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['distance'], y=df['intensity'], mode='markers', name='Actual Data'))
            fig.add_trace(go.Scatter(x=dist_range.flatten(), y=preds, name='AI Prediction', line=dict(color='red')))
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            x_grid = np.linspace(0.1, 5, 50)
            y_grid = np.linspace(0, 2, 10)
            grid_intensities = model.predict(poly.transform(x_grid.reshape(-1, 1)))
            z_data = np.tile(grid_intensities, (len(y_grid), 1))
            
            fig_heat = px.imshow(
                z_data, x=x_grid, y=y_grid,
                labels=dict(x="Distance (m)", y="Lateral", color="µT"),
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Need at least 3 data points to start AI mapping. Keep adding data!")

else:
    st.warning("Database appears empty to the app.")
    st.info("Check if Row Level Security (RLS) is DISABLED in Supabase > Authentication > Policies.")
