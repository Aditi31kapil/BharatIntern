import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle



#lets load the saved vectorizer and naive model

tfidf = pickle.load(open(r'C:\Users\DELL\Desktop\Email Classifier\vectorizer.pkl','rb'))
model = pickle.load(open(r'C:\Users\DELL\Desktop\Email Classifier\model.pkl','rb'))

#transform_text function
import nltk
nltk.download(['punkt', 'stopwords'])
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    #remove special character
    text = [word for word in text if not word.isalnum()]

    #remove punctuations
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]

    #apply stemming
    text = [ps.stem(word) for word in text]
    
    return " ".join(text)

#saving streamlit code
st.title("Sms Spam Classifier")
input_sms = st.text_area("Enter Message")

if st.button('Predict'):
    #preprocess
    transformed_sms = transform_text(input_sms)
    # vectorize
    vector_input = tfidf.transform([transformed_sms])
    #predict
    result = model.predict(vector_input)[0]
    #display
    if result == 0:
        st.header("Spam")
    else:
        st.header("not Spam")