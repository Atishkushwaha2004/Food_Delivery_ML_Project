import streamlit as st
import pickle
import pandas as pd
import os

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Smart Delivery Time Predictor",
    page_icon="🚚",
    layout="wide"
)

# =====================================================
# PREMIUM UI CSS
# =====================================================
st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #0E1117;
}

/* Title */
.title {
    font-size:48px;
    font-weight:700;
    text-align:center;
    color:#4FA3FF;
}

/* Subtitle */
.subtitle {
    text-align:center;
    color:#9CA3AF;
    margin-bottom:30px;
}

/* Card */
.card {
    padding:25px;
    border-radius:18px;
    background: linear-gradient(145deg, #111827, #1F2937);
    box-shadow: 0 8px 25px rgba(0,0,0,0.6);
}

/* Button */
.stButton>button {
    width:100%;
    border-radius:12px;
    height:50px;
    font-size:18px;
    font-weight:600;
}

/* Result */
.result-box {
    font-size:28px;
    font-weight:bold;
    text-align:center;
    padding:18px;
    border-radius:12px;
    background: linear-gradient(90deg, #1E8449, #27AE60);
    color:white;
    margin-top:20px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# TITLE
# =====================================================
st.markdown('<div class="title">🚚 Smart Delivery Time Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter delivery details to predict estimated delivery time</div>', unsafe_allow_html=True)

# =====================================================
# LOAD MODEL (FIXED)
# =====================================================
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, "delivery_model.pkl")

    if not os.path.exists(file_path):
        st.error("❌ Model file not found")
        st.write("Available files:", os.listdir(BASE_DIR))
        return None, None

    with open(file_path, "rb") as f:
        model = pickle.load(f)

    return model, "delivery_model.pkl"


model, model_name = load_model()

if model is None:
    st.stop()

st.success(f"✅ Model Loaded Successfully: {model_name}")

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.header("📊 Model Information")
    st.write("Model File:", model_name)

    if hasattr(model, "feature_names_in_"):
        st.write("Number of Features:", len(model.feature_names_in_))

    st.markdown("---")
    st.info("Fill details and click Predict")

# =====================================================
# INPUT FORM
# =====================================================
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        Distance_km = st.number_input("📏 Distance (km)", 0.0, value=10.0)
        Weather = st.selectbox("🌦 Weather", ["Clear", "Foggy", "Rainy", "Windy"])

    with col2:
        Traffic_Level = st.selectbox("🚦 Traffic", ["Low", "Medium", "High"])
        Time_of_Day = st.selectbox("🕒 Time", ["Morning", "Afternoon", "Evening", "Night"])

    with col3:
        Vehicle_Type = st.selectbox("🚗 Vehicle", ["Bike", "Scooter", "Car"])
        Preparation_Time_min = st.number_input("🍳 Prep Time", 0, value=10)
        Courier_Experience_yrs = st.number_input("👨‍💼 Experience", 0, value=2)

    st.markdown("<br>", unsafe_allow_html=True)

    predict_button = st.button("🚀 Predict Delivery Time")

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# PREDICTION
# =====================================================
if predict_button:
    try:
        input_data = pd.DataFrame({
            "Distance_km": [Distance_km],
            "Weather": [Weather],
            "Traffic_Level": [Traffic_Level],
            "Time_of_Day": [Time_of_Day],
            "Vehicle_Type": [Vehicle_Type],
            "Preparation_Time_min": [Preparation_Time_min],
            "Courier_Experience_yrs": [Courier_Experience_yrs]
        })

        input_encoded = pd.get_dummies(input_data)

        if hasattr(model, "feature_names_in_"):
            input_encoded = input_encoded.reindex(
                columns=model.feature_names_in_,
                fill_value=0
            )

        prediction = model.predict(input_encoded)
        delivery_time = float(prediction[0])

        # RESULT UI
        st.markdown(
            f'<div class="result-box">🚚 Estimated Delivery Time: {delivery_time:.2f} minutes</div>',
            unsafe_allow_html=True
        )

        colA, colB, colC = st.columns(3)

        with colA:
            st.metric("Distance", f"{Distance_km} km")

        with colB:
            st.metric("Traffic", Traffic_Level)

        with colC:
            st.metric("Vehicle", Vehicle_Type)

    except Exception as e:
        st.error("❌ Prediction failed")
        st.write(e)

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.caption("🚀 Machine Learning Project | Delivery Time Prediction")
