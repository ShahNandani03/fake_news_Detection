
import streamlit as st
import joblib

# Load the trained model
model = joblib.load("fake_news_model.pkl")

# Page settings
st.set_page_config(page_title="Fake News Detection", page_icon="📰")

st.title("📰 Fake News Detection")
st.write("Enter a news article to predict whether it is Fake or Real.")

# News input
news_text = st.text_area("Enter News Text:")

if st.button("Check News"):
    if news_text.strip():
        prediction = model.predict([news_text])[0]

        if prediction == 0:
            st.error("Prediction: Fake News")
        else:
            st.success("Prediction: Real News")
    else:
        st.warning("Please enter some news text.")
