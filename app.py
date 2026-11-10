import pickle

import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="centered")


@st.cache_resource
def load_artifacts():
    model = load_model("model.h5", compile=False)
    with open("lb_gender.pk1", "rb") as f:
        lb_gender = pickle.load(f)
    with open("oh_geo.pk2", "rb") as f:
        oh_geo = pickle.load(f)
    with open("scalar.pk1", "rb") as f:
        scalar = pickle.load(f)
    return model, lb_gender, oh_geo, scalar


model, lb_gender, oh_geo, scalar = load_artifacts()

st.title("📉 Customer Churn Prediction")
st.caption("An Artificial Neural Network (ANN) that estimates whether a bank customer is likely to leave.")

with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        geography = st.selectbox("Geography", list(oh_geo.categories_[0]))
        gender = st.selectbox("Gender", list(lb_gender.classes_))
        age = st.slider("Age", 18, 92, 40)
        tenure = st.slider("Tenure (years)", 0, 10, 3)
        num_products = st.slider("Number of Products", 1, 4, 2)

    with col2:
        credit_score = st.number_input("Credit Score", 300, 900, 600)
        balance = st.number_input("Balance", 0.0, 300000.0, 6000.0, step=500.0)
        salary = st.number_input("Estimated Salary", 0.0, 300000.0, 50000.0, step=1000.0)
        has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
        is_active = st.selectbox("Is Active Member?", ["Yes", "No"])

    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    input_df = pd.DataFrame(
        [
            {
                "CreditScore": credit_score,
                "Geography": geography,
                "Gender": gender,
                "Age": age,
                "Tenure": tenure,
                "Balance": balance,
                "NumOfProducts": num_products,
                "HasCrCard": 1 if has_cr_card == "Yes" else 0,
                "IsActiveMember": 1 if is_active == "Yes" else 0,
                "EstimatedSalary": salary,
            }
        ]
    )

    # Encode gender
    input_df["Gender"] = lb_gender.transform(input_df["Gender"])

    # One-hot encode geography
    geo_encoded = oh_geo.transform(input_df[["Geography"]])
    if hasattr(geo_encoded, "toarray"):
        geo_encoded = geo_encoded.toarray()
    geo_df = pd.DataFrame(geo_encoded, columns=oh_geo.get_feature_names_out(["Geography"]))

    input_df = pd.concat([input_df.drop("Geography", axis=1), geo_df], axis=1)

    # Scale and predict
    input_scaled = scalar.transform(input_df)
    prob = float(model.predict(input_scaled, verbose=0)[0][0])

    st.divider()
    st.subheader("Result")
    st.metric("Churn Probability", f"{prob:.2%}")
    st.progress(min(max(prob, 0.0), 1.0))

    if prob > 0.5:
        st.error("⚠️ The customer is **likely to churn**.")
    else:
        st.success("✅ The customer is **not likely to churn**.")
