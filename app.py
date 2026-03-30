import streamlit as st
import pickle
import pandas as pd

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Smart Delivery Time Predictor",
    page_icon="🚚",
    layout="wide"
)

# =====================================================
# CUSTOM CSS (ATTRACTIVE UI)
# =====================================================
st.markdown(
    """
    <style>

    .title {
        font-size:42px;
        font-weight:700;
        text-align:center;
        color:#2E86C1;
    }

    .subtitle {
        font-size:18px;
        text-align:center;
        color:gray;
        margin-bottom:25px;
    }

    .card {
        padding:20px;
        border-radius:15px;
        background-color:#111827;
        box-shadow:0 4px 15px rgba(0,0,0,0.4);
    }

    .result-box {
        font-size:26px;
        font-weight:bold;
        text-align:center;
        padding:15px;
        border-radius:12px;
        background-color:#1E8449;
        color:white;
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">🚚 Smart Delivery Time Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Enter delivery details to predict estimated delivery time (minutes)</div>',
    unsafe_allow_html=True
)

# =====================================================
# LOAD MODEL (AUTO DETECT)
# =====================================================
@st.cache_resource
def load_model():
    model_files = [
        "delivery_model.pkl",
        "lasso_model.pkl",
        "ridge_model.pkl",
        "delivery_model.pkl",
        "model.pkl"
    ]

    for file in model_files:
        try:
            with open(file, "rb") as f:
                model = pickle.load(f)
            return model, file
        except Exception:
            continue

    return None, None


model, model_name = load_model()

if model is None:
    st.error("❌ No model file found. Please keep your .pkl file in the same folder.")
    st.stop()

st.success(f"✅ Model Loaded Successfully: {model_name}")

# =====================================================
# SIDEBAR INFO
# =====================================================
with st.sidebar:
    st.header("📊 Model Information")

    st.write("Model File:", model_name)

    if hasattr(model, "feature_names_in_"):
        st.write("Number of Features:", len(model.feature_names_in_))

    st.markdown("---")

    st.info(
        "Fill all delivery details and click Predict to estimate delivery time."
    )

# =====================================================
# INPUT FORM (MATCHES YOUR DATASET EXACTLY)
# =====================================================
with st.container():

    st.markdown('<div class="card">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    # Column 1
    with col1:
        Distance_km = st.number_input(
            "📏 Distance (km)",
            min_value=0.0,
            step=0.5,
            value=10.0
        )

        Weather = st.selectbox(
            "🌦 Weather",
            [
                "Clear",
                "Foggy",
                "Rainy",
                "Windy"
            ]
        )

    # Column 2
    with col2:
        Traffic_Level = st.selectbox(
            "🚦 Traffic Level",
            [
                "Low",
                "Medium",
                "High"
            ]
        )

        Time_of_Day = st.selectbox(
            "🕒 Time of Day",
            [
                "Morning",
                "Afternoon",
                "Evening",
                "Night"
            ]
        )

    # Column 3
    with col3:
        Vehicle_Type = st.selectbox(
            "🚗 Vehicle Type",
            [
                "Bike",
                "Scooter",
                "Car"
            ]
        )

        Preparation_Time_min = st.number_input(
            "🍳 Preparation Time (minutes)",
            min_value=0,
            step=1,
            value=10
        )

        Courier_Experience_yrs = st.number_input(
            "👨‍💼 Courier Experience (years)",
            min_value=0,
            step=1,
            value=2
        )

    st.markdown("---")

    predict_button = st.button(
        "🚀 Predict Delivery Time",
        use_container_width=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# PREDICTION LOGIC (ROBUST & PRODUCTION SAFE)
# =====================================================
if predict_button:

    try:

        # Create DataFrame (Order_ID excluded)
        input_data = pd.DataFrame({
            "Distance_km": [Distance_km],
            "Weather": [Weather],
            "Traffic_Level": [Traffic_Level],
            "Time_of_Day": [Time_of_Day],
            "Vehicle_Type": [Vehicle_Type],
            "Preparation_Time_min": [Preparation_Time_min],
            "Courier_Experience_yrs": [Courier_Experience_yrs]
        })

        # One-hot encoding
        input_encoded = pd.get_dummies(input_data)

        # Align columns with model
        if hasattr(model, "feature_names_in_"):
            input_encoded = input_encoded.reindex(
                columns=model.feature_names_in_,
                fill_value=0
            )

        # Prediction
        prediction = model.predict(input_encoded)

        delivery_time = float(prediction[0])

        st.balloons()

        st.markdown(
            f'<div class="result-box">🚚 Estimated Delivery Time: {delivery_time:.2f} minutes</div>',
            unsafe_allow_html=True
        )

        # Metrics summary
        colA, colB, colC = st.columns(3)

        with colA:
            st.metric("Distance", f"{Distance_km} km")

        with colB:
            st.metric("Traffic", Traffic_Level)

        with colC:
            st.metric("Vehicle", Vehicle_Type)

    except Exception as e:

        st.error("❌ Prediction failed. Check model training columns.")

        st.write("Error details:")
        st.write(e)

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")

st.caption(
    "Machine Learning Project | Delivery Time Prediction | Production-Ready Streamlit UI"
)
