import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load model
model = pickle.load(open('/Users/sharunrajk/house_price_app/model.pkl', 'rb'))

# Load dataset 
df = pd.read_csv("Bengaluru_House_Data.csv")

st.title("Bengaluru House Price Prediction")

st.write("Enter property details:")

# Location dropdown
locations = sorted(df['location'].dropna().unique())
location = st.selectbox("Select Location", locations)

# Inputs
bhk = st.number_input("BHK", min_value=1)
sqft = st.number_input("Total Sqft", min_value=1.0)
bath = st.number_input("Bathrooms", min_value=1)
balcony = st.number_input("Balcony", min_value=0)

#  Input validation
if sqft <= 0:
    st.error("Total sqft must be greater than 0")

#  Prediction
if st.button("Predict"):
    try:
        # Create a dummy input with correct shape
        input_data = np.zeros((1, model.n_features_in_))
        
        # Fill only important features (adjust positions if needed)
        input_data[0][0] = sqft
        input_data[0][1] = bath
        input_data[0][2] = balcony
        input_data[0][3] = bhk
        
        prediction = model.predict(input_data)
        st.success(f"Estimated Price: {prediction[0]:.2f} Lakhs")
    
    except Exception as e:
        st.error(f"Error: {e}")

#  Top 5 expensive locations for selected BHK
st.subheader(f"Top 5 Expensive Locations for {bhk} BHK")

df_bhk = df[df['size'].str.contains(str(bhk), na=False)]

top_locations = (
    df_bhk.groupby('location')['price']
    .mean()
    .sort_values(ascending=False)
    .head(5)
)

st.bar_chart(top_locations)