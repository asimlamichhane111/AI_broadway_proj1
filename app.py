import streamlit as st
import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

words=stopwords.words('english')
st.title("Streamlit Project")

stemmer=PorterStemmer()

st.write("News Classification")
with open('LogisticRegression.pickle', 'rb') as file:
    model = pickle.load(file)

#Taking value from users
data=st.text_area("Enter news for classification:")
if st.button('Submit'):
    d={'predict_news':[data]}
    df=pd.DataFrame(d)
    df['predict_news'] = list(map(lambda x: " ".join([i for i in x.lower().split() if i not in words]), df['predict_news']))
    df['predict_news'] = df['predict_news'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
    # st.write(df)
    pred=model.predict(df['predict_news'])
    st.write(pred)





