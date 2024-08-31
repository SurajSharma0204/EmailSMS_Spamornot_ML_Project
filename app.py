import streamlit as st 
import pickle
import sklearn
import logging
from nltk.corpus import stopwords
import nltk 
import string 
from nltk.stem.porter import PorterStemmer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)

# Initialize PorterStemmer
ps = PorterStemmer()

# Download NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')

def transform_text(text):
    logging.info("Transforming text...")
    text = text.lower()

    # Tokenize
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    # Remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text = y[:]       
    y.clear()
    
    # Apply stemming
    for i in text:
        y.append(ps.stem(i))
    
    transformed_text = " ".join(y)
    logging.info("Text transformation complete.")
    return transformed_text

# Load pre-trained models and vectorizer
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    logging.info("Models and vectorizer loaded successfully.")
except Exception as e:
    logging.error(f"Error loading models or vectorizer: {e}")
    st.error("Error loading models or vectorizer. Please check the logs for details.")
    st.stop()

# Streamlit app
st.title('Email or SMS Classifier')

input_sms = st.text_area('Enter the Message')

if st.button('Click to Predict'):
    # Preprocess
    transform_sms = transform_text(input_sms)
    logging.info(f"User input: {input_sms}")
    logging.info(f"Transformed input: {transform_sms}")

    # Vectorize
    try:
        vector_input = tfidf.transform([transform_sms])
        logging.info("Text vectorization complete.")
    except Exception as e:
        logging.error(f"Error during vectorization: {e}")
        st.error("Error during vectorization. Please check the logs for details.")
        st.stop()

    result = None    

    # Predict
    try:
        result = model.predict(vector_input)[0]
        logging.info(f"Prediction result: {result}")
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        st.error("Error during prediction. Please check the logs for details.")
        st.stop()

    # Show results
    if result is not None:
        if result == 1:
            st.header("Spam")
        else:
            st.header('Ham')

        logging.info(f"Displayed result: {'Spam' if result == 1 else 'Ham'}")