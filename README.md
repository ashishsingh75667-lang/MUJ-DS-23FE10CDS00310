# NLP Capstone Project: Movie Review Sentiment & Emotion Analysis

## 📌 Student & Batch Details
- **Name:** Ashish Singh
- **Registration Number:** 23FE10CDS00310
- **Branch:** Data Science and Engineering
- **Batch:** Batch E
- **GitHub Username:** ashishsingh75667-lang
- **Project Title:** Multimodal Sentiment & Emotion Classification on Movie Reviews
- **Training Program:** NLP Capstone Project Training Program

---

## 📖 Project Overview
This project delivers an end-to-end Natural Language Processing (NLP) system designed to analyze and classify sentiments in movie reviews alongside fine-grained lexical emotion detection:
- **Dataset:** IMDB 50,000 Movie Reviews Dataset (Balanced 25k positive, 25k negative).
- **Text Preprocessing:** HTML tag removal, alphanumeric filtering, lowercasing, and custom n-gram negation handling (`not_good`).
- **Feature Extraction:** TF-IDF Vectorizer with unigrams to trigrams (10,000 max features) compared against Bag-of-Words (CountVectorizer).
- **Classification Models:** Linear Support Vector Classifier (LinearSVC) and Multinomial Naïve Bayes.
- **Emotion Recognition:** Lexical distribution mapping into 6 core states (Joy, Anger, Sadness, Fear, Surprise, Neutral).
- **Web Interface:** Interactive Streamlit dashboard supporting single-text predictions, confidence calibration, and bulk CSV batch processing.

---

## 📊 Model Performance Benchmarks
| Model / Pipeline | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Linear SVM (TF-IDF)** *(Selected)* | **0.89** | **0.88** | **0.89** | **0.89** |
| **Multinomial Naïve Bayes (TF-IDF)** | 0.86 | 0.83 | 0.88 | 0.86 |
| **Linear SVM (Bag of Words)** | 0.85 | 0.84 | 0.85 | 0.84 |
| **Multinomial Naïve Bayes (BoW)** | 0.86 | 0.83 | 0.88 | 0.86 |

---

## 🗂️ Repository Structure
Following the Capstone Project Guidelines:
\`\`\`text
.
├── README.md                   # Complete project documentation and submission metadata
├── requirements.txt            # Python dependencies
├── .gitignore                  # Excluded files (large CSV datasets, virtualenvs)
├── capstone/
│   ├── app.py                  # Production Streamlit web application
│   ├── svm_model.pkl           # Serialized LinearSVC model artifact
│   └── tfidf_vectorizer.pkl    # Serialized TF-IDF vectorizer artifact
├── notebooks/
│   └── sentiment_analysis_project.ipynb # EDA, preprocessing, training, and benchmarking
├── code/
│   └── app.py                  # Pipeline execution and model source scripts
\`\`\`

---

## 🚀 Installation & Setup Guide

### 1. Clone the Repository
\`\`\`bash
git clone https://github.com/ashishsingh75667-lang/MUJ-DS-23FE10CDS00310
.git
cd MUJ-DS-23FE10CDS00310

\`\`\`

### 2. Install Dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 3. Launch the Streamlit Frontend
\`\`\`bash
cd capstone
streamlit run app.py
\`\`\`
Access the application locally at: \`http://localhost:8501\`

---

## ✨ Key Features
1. **Real-time Sentiment Inference:** Instant positive/negative classification paired with a calibrated confidence meter.
2. **Lexical Emotion Radar:** Identifies underlying affective tones (e.g., Joy, Sadness, Anger, Surprise) from review tokens.
3. **Bulk CSV Processing:** Upload bulk review files, run vector inference in seconds, and export predictions as a downloadable CSV.
4. **Negation-Aware Tokenization:** Converts inverted constructs such as \`not good\` into \`not_good\` to retain semantic weight through sparse vectors.
"@ | Out-File -Encoding utf8 README.md
