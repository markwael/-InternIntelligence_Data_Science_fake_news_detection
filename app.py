import streamlit as st
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import string

# Load vectorizer and ML models
vectorizer = joblib.load("C:\\AMIT\\PROJ\\intern_task2\\vectorizer.pkl")
models = {
    "Logistic Regression": joblib.load("C:\\AMIT\\PROJ\\intern_task2\\logistic_regression.pkl"),
    "Decision Tree": joblib.load("C:\\AMIT\\PROJ\\intern_task2\\decision_tree.pkl"),
    "Gradient Boosting": joblib.load("C:\\AMIT\\PROJ\\intern_task2\\gradient_boosting.pkl"),
    "Random Forest": joblib.load("C:\\AMIT\\PROJ\\intern_task2\\random_forest.pkl"),
}

# Load LSTM model & tokenizer
lstm_tokenizer = joblib.load("C:\\AMIT\\PROJ\\intern_task2\\lstm_tokenizer.pkl")
lstm_model = tf.keras.models.load_model("C:\\AMIT\\PROJ\\intern_task2\\lstm_model.h5")

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Function to predict with traditional ML models
def predict_with_ml(news, model):
    cleaned_news = clean_text(news)
    news_vectorized = vectorizer.transform([cleaned_news])
    prediction = model.predict(news_vectorized)[0]
    return "🟢 Real News" if prediction == 1 else "🔴 Fake News"

# Function to predict with LSTM
def predict_with_lstm(news):
    cleaned_news = clean_text(news)
    sequence = lstm_tokenizer.texts_to_sequences([cleaned_news])
    padded_seq = pad_sequences(sequence, maxlen=200)
    pred = lstm_model.predict(padded_seq)
    return "🟢 Real News" if pred >= 0.5 else "🔴 Fake News"

# Streamlit UI
st.title("📰 Fake News Detection App")
st.write("This application detects whether a news article is **Fake or Real** using different models.")

# Sidebar for model selection
st.sidebar.header("⚙️ Select Model")
selected_model_name = st.sidebar.radio("Choose a Model", list(models.keys()) + ["LSTM"])
selected_model = models.get(selected_model_name, None)

# User Input
st.subheader("📝 Enter News Article")
news_input = st.text_area("Paste your news article below:", height=200)

if st.button("🔍 Analyze News"):
    if news_input.strip() == "":
        st.warning("⚠️ Please enter some text for analysis.")
    else:
        if selected_model_name == "LSTM":
            prediction = predict_with_lstm(news_input)
            st.subheader(f"🤖 LSTM Prediction: {prediction}")
        else:
            prediction = predict_with_ml(news_input, selected_model)
            st.subheader(f"📊 {selected_model_name} Prediction: {prediction}")

# Additional Information
st.sidebar.markdown("🔍 **How it Works**")
st.sidebar.info(
    "This app uses **TF-IDF Vectorization** for traditional models. "
    "Choose a model from the sidebar and test different news articles!"
)

st.write("🚀 **Developed by:** Mark Wael ❤️")
