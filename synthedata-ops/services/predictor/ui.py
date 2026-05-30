import streamlit as st
import requests

# 1. Setup the Page
st.set_page_config(page_title="MedPredict AI", page_icon="🧬", layout="centered")
st.title("🧬 MedPredict: Clinical Diagnostic Engine")
st.markdown("Enter patient demographics and history to generate a real-time risk assessment.")

st.divider()

# 2. Build the Input Form
st.subheader("Patient Profile")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Patient Age", min_value=18, max_value=100, value=55)
    surgeries = st.selectbox("Past Surgeries?", ["Yes", "No"])

with col2:
    lifestyle = st.selectbox("Lifestyle Habit", ["Healthy", "Smoking", "Sedentary"])
    chemicals = st.selectbox("Chemical Exposure?", ["Yes", "No"])

st.divider()

# 3. The Prediction Logic
if st.button("Generate Diagnostic Prediction", type="primary", use_container_width=True):
    # Package the data exactly how our FastAPI backend expects it
    payload = {
        "Age": age,
        "Surgeries_past": surgeries,
        "Lifestyle_habit": lifestyle,
        "Exposure_chemicals": chemicals
    }

    try:
        # Send the data to your Docker container
        with st.spinner("Analyzing patient data with LightGBM..."):
            response = requests.post("http://localhost:8000/predict", json=payload)
            response.raise_for_status()
            result = response.json()

            # 4. Display the Results Beautifully
            st.success("Analysis Complete")

            st.metric(label="Primary Prognosis", value=result["predicted_prognosis"])

            st.write("**Confidence Distribution:**")
            for condition, prob in result["diagnostic_distribution"].items():
                st.progress(prob, text=f"{condition} ({prob*100:.1f}%)")

    except requests.exceptions.ConnectionError:
        st.error("⚠️ Could not connect to the Backend API. Make sure your Docker container is running!")
