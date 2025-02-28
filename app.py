import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("crime_prediction_model.pkl")

# Define city and crime mappings (ensure these match your trained model's mappings)
city_mapping = {
    'Agra': 0, 'Ahmedabad': 1, 'Bangalore': 2, 'Bhopal': 3, 'Chennai': 4, 'Delhi': 5,
    'Faridabad': 6, 'Ghaziabad': 7, 'Hyderabad': 8, 'Indore': 9, 'Jaipur': 10, 'Kalyan': 11,
    'Kanpur': 12, 'Kolkata': 13, 'Lucknow': 14, 'Ludhiana': 15, 'Meerut': 16, 'Mumbai': 17,
    'Nagpur': 18, 'Nashik': 19, 'Patna': 20, 'Pune': 21, 'Rajkot': 22, 'Srinagar': 23,
    'Surat': 24, 'Thane': 25, 'Varanasi': 26, 'Vasai': 27, 'Visakhapatnam': 28
}

crime_mapping = {
    'Arson': 0, 'Assault': 1, 'Burglary': 2, 'Counterfeiting': 3, 'Cybercrime': 4,
    'Domestic Violence': 5, 'Drug Offense': 6, 'Extortion': 7, 'Firearm Offense': 8,
    'Fraud': 9, 'Homicide': 10, 'Identity Theft': 11, 'Illegal Possession': 12,
    'Kidnapping': 13, 'Public Intoxication': 14, 'Robbery': 15, 'Sexual Assault': 16,
    'Shoplifting': 17, 'Traffic Violation': 18, 'Vandalism': 19, 'Vehicle - Stolen': 20
}

def predict_crime(year, city_label, crime_label):
    future_data = pd.DataFrame({"Year": [year], "City_Label": [city_label], "Crime_Label": [crime_label]})
    prediction = model.predict(future_data)
    return round(prediction[0])

# Streamlit UI
st.title("Crime Count Prediction App")

# Dropdown selection for year, city, and crime type
year = st.selectbox("Select Year", list(range(2020, 2026)))
city_name = st.selectbox("Select City", list(city_mapping.keys()))
crime_type = st.selectbox("Select Crime Type", list(crime_mapping.keys()))

# Get encoded values
city_label = city_mapping[city_name]
crime_label = crime_mapping[crime_type]

if st.button("Predict Crime Count"):
    predicted_count = predict_crime(year, city_label, crime_label)
    st.success(f"Predicted Crime Count for {year} in {city_name} ({crime_type}): {predicted_count}")
