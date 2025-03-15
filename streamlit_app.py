# streamlit_app.py
#pickle.dump(tokenizer, open('tokenizer.pickle', 'wb'))
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import nltk
import emoji
import re
import pickle

# Load the saved models
cnn_model = load_model('/Users/saiteja/cnn_model.h5')
lstm_model = load_model('/Users/saiteja/lstm_model.h5')


# Load the tokenizer
tokenizer = pickle.load(open('tokenizer.pickle', 'rb'))

# Preprocessing function
stop_words = set(nltk.corpus.stopwords.words('english'))

def preprocess_text(text):
    if isinstance(text, float):
        return ""
    text = text.lower()
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Define a function to predict sentiment
def predict_sentiment(text, model_choice):
    processed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=200)

    if model_choice == "CNN":
        prediction = cnn_model.predict(padded_sequence)[0]
    else:
        prediction = lstm_model.predict(padded_sequence)[0]

    sentiment_class = np.argmax(prediction)
    sentiment_labels = {0: "negative", 1: "neutral", 2: "positive"}
    return sentiment_labels[sentiment_class]

# Streamlit app layout
st.title("Zomato Review Sentiment Analysis")
text_input = st.text_area("Enter a review:")

# Allow the user to choose between CNN and LSTM models
model_choice = st.radio("Choose a model:", ("LSTM", "CNN"))

if st.button("Predict"):
    if text_input:
        sentiment = predict_sentiment(text_input, model_choice)
        st.write(f"Prediction: {sentiment}")
    else:
        st.write("Please enter a review to analyze.")
