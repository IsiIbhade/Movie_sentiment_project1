import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model and vectorizer
model = joblib.load('logistic_regression_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')  # Ensure you save the vectorizer

# Streamlit interface
st.title("Movie Sentiment Analysis")
st.write("Enter your movie review below:")

# User input
review = st.text_area("Review")

if st.button("Predict"):
    # Vectorize the input review
    review_vectorized = vectorizer.transform([review])
    # Make prediction
    prediction = model.predict(review_vectorized)

    # Display the result
    sentiment = "Positive" if prediction[0] == 1 else "Negative"
    st.write(f"The sentiment of the review is: **{sentiment}**")
