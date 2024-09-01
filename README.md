# EmailSMS_Spamornot_ML_Project

Hi, My name is Suraj Sharma and i have prepared a spam classifer machine learning project.

This project aims to build a machine learning model to classify text messages or emails as either "spam" or "ham" (non-spam). It leverages various machine learning algorithms and preprocessing techniques to achieve accurate classification. The project includes an end-to-end pipeline from data preprocessing to model deployment using Streamlit.

I tried to use the entire project in python 3.8. However, there were compatibility issues with streamlit and dependencies. Then I used 3.12 versions and updated dependencies. Now it worked.

Features:

Text Preprocessing: Uses NLTK for text normalization, including tokenization, stopword removal, and stemming.
Feature Extraction: Utilizes CountVectorizer and TfidfVectorizer for transforming text data into numerical features.
Model Training: Trains multiple classifiers such as MultinomialNB, Logistic Regression, and SVM, with ensemble methods including Voting and Stacking Classifiers.
Evaluation: Provides accuracy and precision metrics for model performance.
Deployment: Deploys the model as an interactive web application using Streamlit(https://msgchecking.streamlit.app/).

Overall, EmailSMS spam classifier serves to improve the efficiency, security, and user experience of messaging systems by effectively managing and filtering out spam messages.
