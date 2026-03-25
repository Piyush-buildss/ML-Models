# app.py
import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression

# Training data
X = np.array([
    [2, 6, 40],
    [3, 7, 50],
    [4, 6, 60],
    [5, 8, 79],
    [6, 7, 85]
])
y = np.array([50, 60, 65, 80, 85])

# Train model
model = LinearRegression()
model.fit(X, y)

# Streamlit UI
st.title("📊 Marks Prediction AI")

study = st.number_input("Enter study hours:", min_value=0)
sleep = st.number_input("Enter sleep hours:", min_value=0)
attendance = st.number_input("Enter attendance:", min_value=0)

if st.button("Predict Marks"):
    marks = model.predict([[study, sleep, attendance]])[0]
    st.subheader(f"Predicted Marks: {marks:.2f}")

    # Logic layer
    if marks < 50:
        st.warning("⚠️ High fail risk. Study more.")
    elif marks < 70:
        st.info("⚡ Average. Improve consistency.")
    else:
        st.success("🔥 Good performance!")

    # Feature impact
    weights = model.coef_
    st.write("---")
    st.write("📌 Feature Contributions:")
    st.write(f"Study contribution: {weights[0]*study:.2f}")
    st.write(f"Sleep contribution: {weights[1]*sleep:.2f}")
    st.write(f"Attendance contribution: {weights[2]*attendance:.2f}")