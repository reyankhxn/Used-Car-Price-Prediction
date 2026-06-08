import streamlit as st
import pandas as pd
import joblib

# Load model and columns
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Used Car Price Predictor")

st.title("🚗 Used Car Price Prediction System")

year = st.number_input(
    "Manufacturing Year",
    min_value=2000,
    max_value=2025,
    value=2018
)

present_price = st.number_input(
    "Present Price (Lakhs)",
    min_value=0.0,
    value=5.0
)

kms_driven = st.number_input(
    "Kilometers Driven",
    min_value=0,
    value=30000
)

owner = st.selectbox(
    "Number of Previous Owners",
    [0, 1, 2, 3]
)

fuel_type = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel"]
)

seller_type = st.selectbox(
    "Seller Type",
    ["Dealer", "Individual"]
)

transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic"]
)

if st.button("Predict Price"):

    data = {
        "Year": year,
        "Present_Price": present_price,
        "Kms_Driven": kms_driven,
        "Owner": owner,
        "Fuel_Type_Diesel": 1 if fuel_type == "Diesel" else 0,
        "Seller_Type_Individual": 1 if seller_type == "Individual" else 0,
        "Transmission_Manual": 1 if transmission == "Manual" else 0
    }

    input_df = pd.DataFrame([data])

    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[columns]

    prediction = model.predict(input_df)

    st.success(
        f"Estimated Selling Price: ₹ {prediction[0]:.2f} Lakhs"
    )