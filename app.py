import streamlit as st
import pickle

# Load the model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Streamlit app interface
st.title("Spam or Ham Classifier")

# Text input from the user
user_input = st.text_area("Enter a message:")

if st.button("Classify"):
    # Preprocessing user input
    input_cleaned = user_input.lower()

    # Vectorizing the input
    input_vec = vectorizer.transform([input_cleaned])

    # Predicting the label
    prediction = model.predict(input_vec)[0]

    # Displaying the result
    if prediction == 'spam':
        st.write("This message is **Spam**.")
    else:
        st.write("This message is **Ham**.")
