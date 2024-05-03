import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pickle

model = pickle.load(open(r'C:\Users\DELL\Desktop\Email Classifier\model.pkl','rb'))


"""from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['v1'])
X_train,X_test,y_train,y_test = train_test_split(df.v2,df.target,test_size=0.2)
cv = CountVectorizer()
x_train_count = cv.fit_transform(X_train.values)
x_train_count.toarray()
model = MultinomialNB()
model.fit(x_train_count,y_train)"""

def classifier(a):
    cv = CountVectorizer()
    count = cv.fit_transform(a)
    count.toarray()
    result = model.predict(count)[0]
    if result == 0:
        st.header("Spam")
    else:
        st.header("not Spam") 
    


st.title("Sms Spam Classifier")
input_sms = st.text_area("Enter Message")
if st.button('Predict'):
    classifier(input_sms)
    