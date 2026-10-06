## 1. Project Overview & Objective
Construct and deploy a full stack Natural Language Processing (NLP) system that:
Learns binary sentiment (Positive or Negative) in the IMDB 50,000 Movie Reviews dataset.
- Preserves semantic context by dealing with negations in text preprocessing (e.g., not_good is changed to not good).
- Uses sparse TF-IDF, Bag-of-Words (CountVectorizer) 1-2 gram to extract text features.
Compare the performance of Benchmarks Linear Support Vector Classifier (LinearSVC) with the Multinomial Naive Bayes (MultinomialNB).
- Accurately identifies the emotion of words at the micro level (Joy, Anger, Sadness, Fear, Surprise, Neutral).
- Supplies an interactive production ready streamlit web application that enables both inference with an individual review and inference in batches from a CSV file.

---

## 2. Dataset & Preprocessing Pipeline
- **Dataset:** `IMDB Dataset.csv` (50,000 balanced movie reviews: 25,000 positive, 25,000 negative).
- **Text Cleaning:**
  - Strip HTML tags (`<.*?>`).
  Remove the special characters, digits and punctuation (regex: [^a-zA-Z ]).
  - Change text to lower case.
- **Negation Handling:**
  - Preserve sentiment polarity in the bag-of-words/n-gram representation of text by converting pattern matching `not <word>` into `not_<word>` (note: this is a feature that is not provided by many other NLP tools).
- **Split Ratio:** 80% Training (40,000 reviews), 20% Testing (10,000 reviews), stratified by sentiment.

---

## 3. Feature Extraction & Modeling
- **Feature Extraction Setup:**
  - TF-IDF Vectorizer: `stop_words='english'`, `ngram_range=(1, 3)`, `max_features=10000`.
  - CountVectorizer (BoW): `stop_words='english'`, `ngram_range=(1, 3)`, `max_features=10000`.
- **Model Evaluation:**
  - LinearSVC (TF-IDF): ~88.6% Accuracy (Production Choice).
  - **MultinomialNB (TF-IDF):** ~85.9% Accuracy.
  - **LinearSVC (BoW):** ~84.7% Accuracy.
  - **MultinomialNB (BoW):** ~85.9% Accuracy.
- **Confidence Calibration:**
  For LinearSVC: calculate prediction confidence by taking the sigmoid function of the absolute decision margin:
    $$\text{Confidence} = \frac{1}{1 + e^{-\vert{}\text{decision\_function}(x)\vert{}}} \times 100$$


## 4. Lexical Emotion Detection Engine
Use an emotion parser based on the dictionary for 5 active states:
Joy: happy, loved, amazing, awesome, wonderful, brilliant, great, good, . . .
Anger: angry, hate, horrible, terrible, furious, annoying, worst, awful, ...
Sadness: sad, unhappy, depressed, cry, tears, disappointed, painful, tragic, ...
fear, scared, scary, frightened, terrified, horror, danger, panic, ... (Fear):
Surprise: surprised, shocked, unexpected, unbelievable, astonishing, wow, ...
Neutral: Standard emotional state if no emotion keywords are found.


## 5. The following is a list of requirements for the Web Interface (Streamlit):
- **Single-Text Prediction:**
  - Text box to paste or type movie reviews.
  - Metrics cards showing Predicted Sentiment, Calibrated Confidence (%), Dominant Emotion.
  An emotion bar graph displaying percentages of different emotions detected.
  - Expanded drawer with emotion keywords extracted from text tokens and cleaned.
- **Batch CSV Analysis:**
  Drag-and-drop File uploader for review data sets in format: `.csv`
  - Dropdown selector to identify the review text column.
  - Batch vectorization and generation of prediction and confidence score columns.
  - Interactive sentiment distribution pie chart and output CSV for download.


## 6. Project Structure & Deliverables
```text
NLP Project
├── README.md                   # Complete project documentation and submission metadata
├── requirements.txt            # Python dependencies
├── prompt.md                   # Serves as a blueprint that store system inst, project specs, etc.
├── IMDB Dataset.csv            # Main dataset used for the NLP pipeline
├── capstone/
│   ├── app.py                  # Production Streamlit web application
│   ├── svm_model.pkl           # Serialized LinearSVC model artifact
│   └── tfidf_vectorizer.pkl    # Serialized TF-IDF vectorizer artifact
├── notebooks/
│   └── sentiment_analysis_project.ipynb # EDA, preprocessing, training, and benchmarking
├── code/
│   └── app.py                  # Pipeline execution and model source scripts
