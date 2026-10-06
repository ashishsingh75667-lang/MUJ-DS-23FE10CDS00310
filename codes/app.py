import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import plotly.express as px

# ----------------- Page Configuration -----------------
st.set_page_config(
    page_title="Sentiment & Emotion Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- Helper Functions & Dictionaries -----------------
EMOTION_WORDS = {
    "joy": [
        "happy", "happiness", "joy", "joyful", "love", "loved", "lovely", 
        "amazing", "awesome", "excellent", "fantastic", "wonderful", 
        "brilliant", "great", "good", "fun", "enjoy", "enjoyed", 
        "enjoyable", "excited", "exciting", "perfect", "beautiful", "best", "pleasure"
    ],
    "anger": [
        "angry", "anger", "hate", "hated", "horrible", "terrible", "furious", 
        "annoyed", "annoying", "rage", "rude", "worst", "awful", "disgusting", 
        "irritating", "irritated"
    ],
    "sadness": [
        "sad", "sadness", "unhappy", "depressed", "cry", "crying", "tears", 
        "lonely", "loneliness", "disappointed", "disappointing", 
        "disappointment", "pain", "painful", "tragic", "heartbroken", "sorry"
    ],
    "fear": [
        "fear", "fearful", "scared", "scary", "frightened", "terrified", 
        "terror", "horror", "horrifying", "danger", "dangerous", "threat", 
        "threatening", "panic", "panicked"
    ],
    "surprise": [
        "surprised", "surprise", "shocked", "shock", "unexpected", 
        "unbelievable", "suddenly", "astonishing", "astonished", "wow", "incredible"
    ]
}

EMOTION_ICONS = {
    "joy": "😊 Joy",
    "anger": "😡 Anger",
    "sadness": "😢 Sadness",
    "fear": "😨 Fear",
    "surprise": "😲 Surprise",
    "neutral": "😐 Neutral"
}

def clean_text(text: str) -> str:
    text = re.sub(r"<.*?>", "", str(text))
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text.lower()

def handle_negation(text: str) -> str:
    return re.sub(r"not\s+(\w+)", r"not_\1", text)

def detect_emotions(review: str):
    words = re.findall(r'\b[a-zA-Z]+\b', review.lower())
    scores = {}
    found_keywords = []

    for emotion, keywords in EMOTION_WORDS.items():
        count = 0
        for w in words:
            if w in keywords:
                count += 1
                found_keywords.append(f"{w} ({emotion})")
        scores[emotion] = count

    total = sum(scores.values())
    if total == 0:
        percentages = {k: 0.0 for k in scores}
        percentages["neutral"] = 100.0
        dominant = "neutral"
    else:
        percentages = {k: (v / total) * 100 for k, v in scores.items()}
        percentages["neutral"] = 0.0
        dominant = max(percentages, key=percentages.get)

    return percentages, dominant, found_keywords

# ----------------- Load Artifacts -----------------
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("svm_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        return model, vectorizer
    except FileNotFoundError:
        return None, None

model, tfidf = load_artifacts()

# ----------------- Sidebar -----------------
st.sidebar.title("NLP Control Hub")
st.sidebar.markdown("**Model:** LinearSVC")
st.sidebar.markdown("**Vectorizer:** TF-IDF (10k features, 1-3 n-grams)")
st.sidebar.markdown("---")
view_mode = st.sidebar.radio("Select Workflow:", ["Single Text Prediction", "Batch CSV Processing", "Model Insights"])

if model is None or tfidf is None:
    st.error("Model artifacts (`svm_model.pkl` and `tfidf_vectorizer.pkl`) not found! Please run the export script from your notebook first.")
    st.stop()

# ----------------- App Views -----------------
if view_mode == "Single Text Prediction":
    st.title("Sentiment & Emotion Analyzer")
    st.caption("Real-time sentiment classification with confidence scoring and lexical emotion distribution.")

    default_text = "A wonderful little production with brilliant acting and great storytelling!"
    user_input = st.text_area("Enter your review or text here:", value=default_text, height=130)

    if st.button("Analyze Sentiment", type="primary"):
        if not user_input.strip():
            st.warning("Please enter some text before analyzing.")
        else:
            # Preprocessing & Prediction
            cleaned = clean_text(user_input)
            negation_handled = handle_negation(cleaned)
            vec = tfidf.transform([negation_handled])
            
            prediction = model.predict(vec)[0]
            decision = model.decision_function(vec)[0]
            confidence = (1 / (1 + np.exp(-abs(decision)))) * 100
            
            emotion_pct, dominant_emotion, keywords_found = detect_emotions(user_input)

            st.markdown("---")
            col1, col2, col3 = st.columns(3)

            with col1:
                if prediction.lower() == "positive":
                    st.metric("Predicted Sentiment", "POSITIVE", delta="Favorable")
                else:
                    st.metric("Predicted Sentiment", "NEGATIVE", delta="-Critical", delta_color="inverse")

            with col2:
                st.metric("Confidence Score", f"{confidence:.2f}%")

            with col3:
                st.metric("Dominant Emotion", EMOTION_ICONS.get(dominant_emotion, dominant_emotion).upper())

            # Detailed Visuals
            st.markdown("### Emotion Breakdown")
            chart_df = pd.DataFrame({
                "Emotion": [EMOTION_ICONS[k] for k in emotion_pct.keys()],
                "Percentage (%)": list(emotion_pct.values())
            })
            fig = px.bar(chart_df, x="Percentage (%)", y="Emotion", orientation="h", color="Percentage (%)",
                         color_continuous_scale="Viridis", text_auto=".1f")
            fig.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False, height=320)
            st.plotly_chart(fig, use_container_width=True)

            # Explainability / Token Inspection
            with st.expander("Show Preprocessed Pipeline & Detected Keywords"):
                st.write("**Cleaned & Negation Form:**", negation_handled)
                st.write("**Detected Emotion Keywords:**", ", ".join(keywords_found) if keywords_found else "None")

elif view_mode == "Batch CSV Processing":
    st.title("Batch Review Analysis")
    st.caption("Upload a CSV file containing movie/product reviews to analyze sentiments in bulk.")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data:", df_uploaded.head())

        text_column = st.selectbox("Select the column containing review text:", df_uploaded.columns)

        if st.button("Run Batch Prediction", type="primary"):
            with st.spinner("Processing reviews..."):
                processed_texts = df_uploaded[text_column].astype(str).apply(clean_text).apply(handle_negation)
                batch_vec = tfidf.transform(processed_texts)
                predictions = model.predict(batch_vec)
                
                decision_scores = model.decision_function(batch_vec)
                confidences = (1 / (1 + np.exp(-np.abs(decision_scores)))) * 100

                df_uploaded["predicted_sentiment"] = predictions
                df_uploaded["confidence_score"] = np.round(confidences, 2)

                st.success("Batch processing complete!")

                # Distribution chart
                c1, c2 = st.columns([1, 2])
                with c1:
                    counts = df_uploaded["predicted_sentiment"].value_counts().reset_index()
                    counts.columns = ["Sentiment", "Count"]
                    fig_pie = px.pie(counts, names="Sentiment", values="Count", color="Sentiment",
                                     color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c"})
                    st.plotly_chart(fig_pie, use_container_width=True)

                with c2:
                    st.dataframe(df_uploaded[[text_column, "predicted_sentiment", "confidence_score"]].head(15), height=350)

                # Export CSV
                csv_data = df_uploaded.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Results as CSV",
                    data=csv_data,
                    file_name="sentiment_predictions.csv",
                    mime="text/csv"
                )

elif view_mode == "Model Insights":
    st.title("Project Benchmarks & Insights")
    st.markdown("""
    Here is a summary of the model evaluation obtained during training:
    - **Linear Support Vector Machine (TF-IDF):** ~88.6% Accuracy *(Chosen for deployment)*
    - **Multinomial Naive Bayes (TF-IDF):** ~85.9% Accuracy
    - **Linear SVM (Bag of Words):** ~84.7% Accuracy
    
    ### Why TF-IDF + LinearSVC?
    1. **N-gram Representation (1 to 3):** Captures multi-word phrases such as negations.
    2. **Linear Hyperplane:** High-dimensional sparse text vectors are separated effectively by linear SVM.
    3. **Low Latency:** Fast inference time suitable for real-time applications.
    """)