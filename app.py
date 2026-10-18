
import streamlit as st
import joblib

model = joblib.load("fake_news_model.pkl")

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰"
)

st.title("📰 Fake News Detection")
st.write("Enter a news title and article to get a model prediction.")

title = st.text_input("News Title")
article = st.text_area("Full News Article", height=250)

if st.button("Check News"):
    if title.strip() and article.strip():
        news = title.strip() + " " + article.strip()
        prediction = model.predict([news])[0]

        if prediction == 0:
            st.error("Prediction: Fake News")
        else:
            st.success("Prediction: Real News")

        st.caption(
            "This is an ML prediction, not a verification of the news."
        )
    else:
        st.warning("Please enter both the title and full article.")
