import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from supabase import create_client
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 1. Connection to Supabase
# Make sure these match the keys in your Streamlit "Secrets"
SUB_URL = st.secrets["SUPABASE_URL"]
SUB_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUB_URL, SUB_KEY)

st.set_page_config(page_title="AI EMF Risk Mapper", layout="wide")
st.title("⚡ AI-Based EMF Monitoring & Risk Mapping")

# 2. Function to fetch data from Supabase
def fetch_data():
    try:
        # Change "emf_readings" to whatever your table name is in Supabase
        response = supabase.table("emf_readings").select("*").order("id", desc=True).limit(100).execute()
        return pd.DataFrame(response.data)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

df = fetch_data()

if not df.empty:
    latest = df.iloc[0]
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Latest Intensity", f"{latest['intensity']:.2f} µT")
        st.subheader("Field Intensity Over Time")
        st.line_chart(df.set_index('id')['intensity'])

    # 3. AI Prediction Logic
    # We need at least a few points to train the model
    if len(df) > 5:
        X = df[['distance']].values
        y = df['intensity'].values
        poly = PolynomialFeatures(degree=2)
        model = LinearRegression().fit(poly.fit_transform(X), y)

        with col2:
            st.subheader("2D AI Exposure Map")
            x_grid = np.linspace(0.1, 5, 50)
            preds = model.predict(poly.transform(x_grid.reshape(-1, 1)))
            z_data = np.tile(preds, (20, 1))
            
            fig_heat = px.imshow(z_data, x=x_grid, color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig_heat)
    else:
        st.info("Collecting more data points for AI training...")
else:
    st.warning("Database is currently empty. Waiting for ESP32 data...")
