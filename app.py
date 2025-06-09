import streamlit as st

st.set_page_config(page_title="Sentilytics", layout="centered")

from transformers import pipeline
import matplotlib.pyplot as plt

# rest of your code...

@st.cache(allow_output_mutation=True)
def load_pipeline():
    return pipeline("sentiment-analysis")

sentiment_pipeline = load_pipeline()

st.set_page_config(page_title="Sentilytics", layout="centered")
st.title("📊 Sentilytics: Real-time Sentiment Analyzer")

user_input = st.text_area("📝 Enter your text below:")

if st.button("Analyze"):
    if user_input.strip():
        result = sentiment_pipeline(user_input)[0]
        label = result["label"]
        score = result["score"]

        st.markdown(f"### ✅ Sentiment: **{label}**")
        st.markdown(f"### 🎯 Confidence: **{score:.2f}**")

        # Pie Chart
        fig, ax = plt.subplots()
        ax.pie([score, 1 - score], labels=[label, "Other"], colors=["green", "lightgray"], autopct="%1.1f%%")
        ax.axis("equal")
        st.pyplot(fig)
    else:
        st.warning("Please enter some text before analyzing.")
