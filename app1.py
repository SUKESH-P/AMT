import streamlit as st
import pandas as pd
import numpy as np
import pickle  # For loading the trained model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor

# Load trained model & feature names
@st.cache_data
def load_model():
    with open("gbr_model.pkl", "rb") as file:
        model_data = pickle.load(file)
    return model_data["model"], model_data["feature_names"]

gbr, expected_features = load_model()

# Streamlit UI
st.title("🚀 Delivery Time Prediction App")
st.write("Enter the details below to predict the estimated delivery time.")

# Feature Inputs
Category = st.selectbox("Category", [ 2,  4, 14,  3, 15, 13, 11,  0,  7,  9,  5,  1,  8,  6, 10, 12])
Agent_Rating = st.slider("Agent Rating", 1.0, 5.0, step=0.1)
Weather = st.selectbox("Weather Condition", [4, 3, 2, 0, 1, 5])
Traffic = st.selectbox("Traffic Condition",[0, 1, 2, 3])
Agent_Age = st.number_input("Agent Age", min_value=18, max_value=70, value=30)
Order_Time_hr = st.slider("Order Hour", 0, 23, value=12)
Order_Date_day = st.slider("Order Day", 1, 31, value=15)
Drop_Longitude = st.number_input("Drop Longitude", value=77.5)
Drop_Latitude = st.number_input("Drop Latitude", value=12.9)
Store_Longitude = st.number_input("Store Longitude", value=77.6)
Store_Latitude = st.number_input("Store Latitude", value=13.0)
Vehicle = st.selectbox("Vehicle Type", [0, 1, 2])
Pickup_Time_min = st.slider("Pickup Time (Minutes)", 0, 59, value=10)
Order_Time_min = st.slider("Order Time (Minutes)", 0, 59, value=30)
Area = st.selectbox("Area Type", [3, 0, 2, 1])
Order_Date_month = st.slider("Order Month", 1, 12, value=6)


# Convert inputs to DataFrame
input_data = pd.DataFrame({
    "Category": [Category],
    "Agent_Rating": [Agent_Rating],
    "Weather": [Weather],
    "Traffic": [Traffic],
    "Agent_Age": [Agent_Age],
    "Order_Time_hr": [Order_Time_hr],
    "Order_Date_day": [Order_Date_day],
    "Drop_Longitude": [Drop_Longitude],
    "Drop_Latitude": [Drop_Latitude],
    "Store_Longitude": [Store_Longitude],
    "Store_Latitude": [Store_Latitude],
    "Vehicle": [Vehicle],
    "Pickup_Time_min": [Pickup_Time_min],
    "Order_Time_min": [Order_Time_min],
    "Area": [Area],
    "Order_Date_month": [Order_Date_month]
})

# Ensure feature order matches training data
input_data = input_data[expected_features]

# Predict Button
if st.button("Predict Delivery Time"):
    predicted_time = gbr.predict(input_data)[0]
    st.success(f"⏳ Estimated Delivery Time: {predicted_time:.2f} minutes")

    