# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib


# Load models and encoders
reg_model = joblib.load("price_predictor.pkl")
clf_model = joblib.load("rating_classifier.pkl")
encoders = joblib.load("label_encoders.pkl")

# Load source data to get dynamic dropdown options
data_df = pd.read_csv("enhanced_zomato_dataset_clean.csv")  # Ensure this matches your dataset file name

# Extract unique dropdown values
cities = sorted(data_df["City"].dropna().unique())
cuisines = sorted(data_df["Cuisine"].dropna().unique())
items = sorted(data_df["Item_Name"].dropna().unique())
places = sorted(data_df["Place_Name"].dropna().unique())
restaurants = sorted(data_df["Restaurant_Name"].dropna().unique())
bestseller_choices = sorted(data_df["Best_Seller"].dropna().unique())


# Add this early in your code
if "predict" not in st.session_state:
    st.session_state.predict = False

def trigger_predict():
    st.session_state.predict = True

st.set_page_config(page_title="🍽️ Restaurant Intelligence App", layout="centered")
st.title("🍽️ Restaurant Intelligence: Price & Rating Predictor")
st.write("Enter restaurant and item details below:")


with st.form("input_form"):
    restaurant_name = st.selectbox("Restaurant Name", restaurants)
    cuisine = st.selectbox("Cuisine", cuisines)
    place = st.selectbox("Place Name", places)
    city = st.selectbox("City", cities)
    item_name = st.selectbox("Item Name", items)
    best_seller = st.selectbox("Is Best Seller?", bestseller_choices)

    dining_rating = st.slider("Dining Rating", 0.0, 5.0, 4.0)
    delivery_rating = st.slider("Delivery Rating", 0.0, 5.0, 4.0)
    dining_votes = st.number_input("Dining Votes", min_value=0, value=30)
    delivery_votes = st.number_input("Delivery Votes", min_value=0, value=10)
    total_votes = st.number_input("Total Votes", min_value=0, value=40)
    avg_rating = st.slider("Average Rating", 0.0, 5.0, 4.2)
    restaurant_popularity = st.number_input("Restaurant Popularity Score", min_value=0, value=50)
    avg_rating_rest = st.slider("Avg Rating for Restaurant", 0.0, 5.0, 4.0)
    avg_price_rest = st.number_input("Avg Price for Restaurant", min_value=0.0, value=250.0)

    submitted = st.form_submit_button("Predict", on_click=trigger_predict)


# Run prediction only if user clicks Predict
if st.session_state.predict:

    # Derived features
    rating_gap = dining_rating - delivery_rating
    votes = dining_votes + delivery_votes
    value_score = avg_rating / avg_price_rest if avg_price_rest > 0 else 0
    price_per_vote = avg_price_rest / (votes + 1)  # avoid zero
    log_price = np.log1p(avg_price_rest)

    # Encode input
    def encode(col, value):
        if value in encoders[col].classes_:
            return encoders[col].transform([value])[0]
        else:
            return 0  # fallback to 0 if unseen category

    input_data = pd.DataFrame({
        "Restaurant_Name": [encode("Restaurant_Name", restaurant_name)],
        "Dining_Rating": [dining_rating],
        "Delivery_Rating": [delivery_rating],
        "Dining_Votes": [dining_votes],
        "Delivery_Votes": [delivery_votes],
        "Cuisine": [encode("Cuisine", cuisine)],
        "Place_Name": [encode("Place_Name", place)],
        "City": [encode("City", city)],
        "Item_Name": [encode("Item_Name", item_name)],
        "Best_Seller": [encode("Best_Seller", best_seller)],
        "Votes": [votes],
        "Average_Rating": [avg_rating],
        "Total_Votes": [total_votes],
        "Price_per_Vote": [price_per_vote],
        "Log_Price": [log_price],
        "Is_Bestseller": [1 if best_seller == "BESTSELLER" else 0],
        "Restaurant_Popularity": [restaurant_popularity],
        "Avg_Rating_Restaurant": [avg_rating_rest],
        "Avg_Price_Restaurant": [avg_price_rest],
        "Rating_Gap": [rating_gap],
        "Value_Score": [value_score]
    })


    price_pred = reg_model.predict(input_data)[0]
    is_highly_rated = clf_model.predict(input_data)[0]

    st.success(f"\n💰 **Predicted Price:** ₹{round(price_pred, 2)}")
    st.info(f"\n⭐ **Highly Rated:** {'Yes' if is_highly_rated else 'No'}")
    # Reset flag after prediction (optional)
    st.session_state.predict = False
