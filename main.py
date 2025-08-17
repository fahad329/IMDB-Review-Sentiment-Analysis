#loading the libraries and model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

model = load_model('simple_rnn_imdb.h5')

word_index = imdb.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items()}

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

def predict(review):
    preprocess_review = preprocess_text(review)
    prediction = model.predict(preprocess_review)

    sentiment = "Positive" if prediction[0][0] > 0.5 else 'Negative'
    return sentiment,prediction[0][0]


#streamlit app
import streamlit as st

st.title('IMDB Movie Review Sentiment Anaylsis')
st.write('Enter a movie Review to classify as positive or negative')

#user input

user_input = st.text_area('Movie Review')

if st.button('Classify'):

    #movie prediction
    sentiment,prediction = predict(user_input)
    #Display Results
    st.write("Sentiment: ",sentiment)
    st.write("Prediction Score: ", prediction)
else:
    st.write('Please enter a movie review')
    

