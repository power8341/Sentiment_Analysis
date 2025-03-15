import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Embedding, GlobalMaxPooling1D, Dropout, Bidirectional, SpatialDropout1D, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
import re
from fastapi import FastAPI
from nltk.tokenize import word_tokenize
from tensorflow.keras.optimizers import Adam
import emoji
import uvicorn
import streamlit as st
import pickle
# from pyngrok import ngrok

file_path = "/Users/saiteja/Downloads/zomato_reviews.csv"  # Update with actual file path
df = pd.read_csv(file_path)



stop_words = set(stopwords.words('english'))
# Preprocessing function
def preprocess_text(text):
    if isinstance(text, float):  
        return ""
    text = text.lower()
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    #tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)


# Apply preprocessing
df['cleaned_text'] = df['review'].apply(preprocess_text)

# Filter out empty reviews after preprocessing
df = df[df['cleaned_text'].str.strip().str.len() > 0]

max_words = 20000
max_len = 200
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(df['cleaned_text'])
sequences = tokenizer.texts_to_sequences(df['cleaned_text'])
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

pickle.dump(tokenizer, open('tokenizer.pickle', 'wb'))

def categorize_rating(rating):
    if rating <= 2:
        return 0  # Negative
    elif rating == 3:
        return 1  # Neutral
    else:  # rating >= 4
        return 2  # Positive
    

df['category'] = df['rating'].apply(categorize_rating).astype(int)

from tensorflow.keras.utils import to_categorical
y = to_categorical(df['category'], num_classes=3)

#X_train, X_temp, y_train, y_temp = train_test_split(padded_sequences, y, test_size=0.2, random_state=42, stratify=df['category'])
X_train, X_temp, y_train, y_temp = train_test_split(padded_sequences, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


def create_cnn_model():
    model = Sequential([
        Embedding(max_words, 200, input_length=max_len),
        Conv1D(256, 5, activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(0.4),
        Conv1D(128, 5, activation='relu', padding='same'),
        GlobalMaxPooling1D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0005, decay=1e-6), metrics=['accuracy'])
    return model

# LSTM Model
def create_lstm_model():
    model = Sequential([
        Embedding(max_words, 200, input_length=max_len),
        SpatialDropout1D(0.4),
        Bidirectional(LSTM(256, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(128)),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(3, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0007, decay=1e-6), metrics=['accuracy'])
    return model


cnn_model = create_cnn_model()
lstm_model = create_lstm_model()

cnn_model.save("cnn_model.h5")
lstm_model.save("lstm_model.h5")


cnn_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=12, batch_size=32)
lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=12, batch_size=32)

cnn_results = cnn_model.evaluate(X_test, y_test)
lstm_results = lstm_model.evaluate(X_test, y_test)
print("CNN Test Accuracy:", cnn_results[1])
print("LSTM Test Accuracy:", lstm_results[1])

