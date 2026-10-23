
import streamlit as st
import joblib

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("fake_news_model.pkl")

model = load_model()

st.title("📰 Fake News Detection")
st.write(
    "A Machine Learning project using TF-IDF "
    "and Logistic Regression."
)

st.info(
    "Enter a news headline and article to get a model prediction."
)

title = st.text_input("📝 News Title")
article = st.text_area(
    "📄 Full News Article",
    height=220,
    placeholder="Paste the full news article here..."
)

if st.button("🔍 Check News", use_container_width=True):
    if title.strip() and article.strip():
        news = title.strip() + " " + article.strip()

        with st.spinner("Analyzing news..."):
            prediction = model.predict([news])[0]
            probabilities = model.predict_proba([news])[0]

        if prediction == 0:
            st.error("🚨 Model Prediction: Fake News")
        else:
            st.success("✅ Model Prediction: Real News")

        st.caption(
            f"Fake probability: {probabilities[0] * 100:.2f}%"
        )
        st.caption(
            f"Real probability: {probabilities[1] * 100:.2f}%"
        )
    else:
        st.warning("Please enter both the title and full article.")

st.divider()
st.caption(
    "Disclaimer: This model predicts based on learned patterns. "
    "It does not verify facts, and predictions may be incorrect."
)

