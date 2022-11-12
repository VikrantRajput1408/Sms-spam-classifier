import streamlit as st
import pickle
import nltk
import re
import sklearn
import string
from nltk.corpus import stopwords


# stemmer
stemmer = nltk.SnowballStemmer("english")

# Stopwords
stop_words = stopwords.words('english')
stop_words = stop_words + ['u', 'im', 'c']


def clean_text(text):
    """Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers."""
    text = str(text).lower()

    # Clean Text with removing speacial Charactor
    text = re.sub("\[.*?\]", '', text)
    text = re.sub("https?://\S+|www\.\S+", '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)

    text = ' '.join(word for word in text.split(' ') if word not in stop_words)  # Remove Stopwords
    text = ' '.join(stemmer.stem(word) for word in text.split(' '))  # stemming
    return text


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


st.title("Email Spam Classifier")
input_sms = st.text_area("Enter The Massage")

if st.button("SPAM / NOT SPAM"):

    # preprocess
    clean_sms = clean_text(input_sms)

    # vectorization
    vect_sms = tfidf.transform([clean_sms])

    # prediction
    result = model.predict(vect_sms)

    # Display
    if result == 0:
        st.header("Not Spam")
    else:
        st.header("Spam")

