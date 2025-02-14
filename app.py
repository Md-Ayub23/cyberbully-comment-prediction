import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def load_model():
    with open("cyberbullying_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

def predict_message(message, best_model, vectorizer):
    message_tfidf = vectorizer.transform([message])  # Convert input text to TF-IDF
    prediction = best_model.predict(message_tfidf)[0]  # Predict class (0 or 1)
    return "Offensive Text" if prediction == 1 else "Safe Text"

# Load the trained model and vectorizer
best_model, vectorizer = load_model()

# Streamlit UI
st.title("NLP-Powered Toxicity Analysis")
st.write("Enter a message below to Check if this message contains offensive or toxic language.")

user_input = st.text_area("Enter message:")
if st.button("Predict"):
    if user_input.strip():
        prediction = predict_message(user_input, best_model, vectorizer)
        st.write(f"### Prediction: {prediction}")
    else:
        st.warning("Please enter a valid message.")
