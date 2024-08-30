import streamlit as st 
import pickle
import sklearn
from nltk.corpus import stopwords
import nltk 
import string 
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

nltk.download('punkt')
nltk.download('stopwords')




def transform_text(text):
    text = text.lower()
    
    text= nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and  i not in string.punctuation:
            y.append(i)
            
    text = y[:]       
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
    
    return " ".join(y)



tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


st.title('Email or Spam Classifier')

input_sms = st.text_area('Enter the Message ')


if st.button('Click to Predict'):
    # Preprocess
    transform_sms = transform_text(input_sms)
    # Vectorize
    vector_input = tfidf.transform([transform_sms])
    # Predict
    result = model.predict(vector_input)[0]

    # Show

    if result == 1:
        st.header("spam")
    else:
        st.header('ham')

