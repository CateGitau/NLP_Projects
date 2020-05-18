import datetime as dt
import re

import pandas as pd
import streamlit as st
from flair.data import Sentence
from flair.models import TextClassifier
#from twitterscraper import query_tweets


# Set page title
st.title('Twitter Sentiment Analysis')

# Load classification model
#st.spinner allows us to display some text while something is computing
with st.spinner('Loading classification model...'):
    classifier = TextClassifier.load('model-saves/best-model.pt')


#pre-processing function
allowed_chars = ' AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz0123456789~`!@#$%^&*()-=_+[]{}|;:",./<>?'
punct = '!?,.@#'
maxlen = 280

def preprocess(text):
    return ''.join([' ' + char + ' ' if char in punct else char for char in [char for char in re.sub(r'http\S+', 'http', text, flags=re.MULTILINE)\
    if char in allowed_chars]])[:maxlen]


#classifying inidividual tweets
st.subheader('Single tweet classification')

tweet_input = st.text_input('Tweet:')

#As long as the input string is not empty, we want to:
# - Pre-process the tweet
# - Make predictions (with the spinner as we wait)
# - Show the predictions

if tweet_input != '':
    # Pre-process tweet
    sentence = Sentence(preprocess(tweet_input))

    # Make predictions
    with st.spinner('Predicting...'):
        classifier.predict(sentence)

    # Show predictions
    label_dict = {'0': 'Negative', '4': 'Positive'}

    if len(sentence.labels) > 0:
        st.write('Prediction:')
        st.write(label_dict[sentence.labels[0].value] + ' with ',
                sentence.labels[0].score*100, '% confidence')
