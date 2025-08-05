import streamlit as st
import numpy as np
import pickle as pkl


with open("Model.pkl", "rb") as f:
    model=pkl.load(f)
with open("scaler.pkl", "rb") as f:
    scaler=pkl.load(f)
posted_by_map = {"Owner": 1, "Dealer": 0, "Builder": 2}
bhk_or_rk_map = {"BHK": 0, "RK": 1}

st.title("House Price Prediction")

under_construction = st.selectbox("Under Construction (0/1)", [0, 1])
rera = st.selectbox("RERA Approved (0/1)", [0, 1])
bhk_or_rk = st.selectbox("Type", list(bhk_or_rk_map.keys()))
no_of_rooms = st.number_input("Total Rooms", 1, 20)
resale = st.selectbox("Is Resale (0/1)", [0, 1])
posted_by = st.selectbox("Posted By", list(posted_by_map.keys()))
bhk = st.number_input("Number of BHK", 1, 10)
square_ft = st.number_input("Square Feet", 200, 10000)
ready_to_move = st.selectbox("Ready to Move (0/1)", [0, 1])
bathroom = st.number_input("Number of Bathrooms", 1, 10)

if st.button("Predict"):
    input_data = np.array([[under_construction, rera, bhk_or_rk_map[bhk_or_rk],
                            no_of_rooms, resale, posted_by_map[posted_by],
                            bhk, square_ft, ready_to_move, bathroom]])

    input_scaled = scaler.transform(input_data)
    prediction_log = float(model.predict(input_scaled)[0])
    prediction_log = max(0, min(prediction_log, 20))
    predicted_price = np.expm1(prediction_log)

    if not np.isfinite(predicted_price):
        st.write("Prediction error. Please try again.")
    else:
        st.write(f"Estimated Price: ₹ {predicted_price:.2f} Lacs")



