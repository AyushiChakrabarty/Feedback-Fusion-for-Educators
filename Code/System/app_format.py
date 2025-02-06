import os
import streamlit as st
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from transformers import pipeline, BertTokenizer, BertForSequenceClassification
import torch
from dotenv import load_dotenv
import io
import re
import string

# Load environment variables from .env file
load_dotenv()

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")

# Preprocessing functions
def clean_text(text):
    if pd.isnull(text):
        return ''
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)  # Remove punctuation
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    return text.strip()

def preprocess_dataframe(df):
    st.write(f"DataFrame shape before dropping NaNs: {df.shape}")
    df.dropna(inplace=True)  # Remove empty rows
    st.write(f"DataFrame shape after dropping NaNs: {df.shape}")
    if 'concerns' in df.columns:
        df['concerns'] = df['concerns'].apply(clean_text)
    if 'anything else' in df.columns:
        df['anything else'] = df['anything else'].apply(clean_text)
    return df

# Function to load BERT model and tokenizer
@st.cache_resource
def load_model():
    model_path = "bert_model_small.pth"
    model = BertForSequenceClassification.from_pretrained('prajjwal1/bert-tiny', num_labels=4)
    model.load_state_dict(torch.load(model_path))
    tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
    return model, tokenizer

model, tokenizer = load_model()
classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)

# Dictionary of keywords for each category
keywords = {
    "NC": ["no concern", "none", "nothing", "all good", "no problem", "okay", "fine", "satisfied", "happy", "no issues", "sorted", "No"]
}

# Function to categorize responses
def categorize_response(response):
    response = clean_text(response)  # Preprocess the response
    response_lower = response.lower()  # Convert to lowercase for consistent matching

    # Predict the category using the model
    result = classifier(response)
    predicted_label = result[0]['label']

    # Check if any keywords match for a particular category
    for label, words in keywords.items():
        if any(word in response_lower for word in words):
            return label

    # If no keywords match, return the model's prediction
    labels = {'LABEL_0': 'AC', 'LABEL_1': 'PC', 'LABEL_2': 'TC', 'LABEL_3': 'NC'}
    return labels.get(predicted_label, "NC")  # Default to "NC" if the prediction is unclear

# Function to process the DataFrame and append categorized columns
def process_dataframe(df):
    df = preprocess_dataframe(df)  # Preprocess the DataFrame
    if 'concerns' in df.columns:
        df['concerns_category'] = df['concerns'].apply(categorize_response)
    if 'anything else' in df.columns:
        df['anything_else_category'] = df['anything else'].apply(categorize_response)
    return df

# Function to convert DataFrame to binary CSV
def to_csv_bytes(df):
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer.read()

# Function to load data with multiple encoding retries
def load_data(file):
    encodings = ['utf-8', 'ISO-8859-1', 'latin1', 'utf-16']  # List of encodings to try
    for encoding in encodings:
        try:
            df = pd.read_csv(file, encoding=encoding)
            st.success(f"Successfully loaded file with encoding: {encoding}")
            return df
        except UnicodeDecodeError:
            continue  # Try the next encoding
    st.error("Unable to decode the file with available encodings.")
    return None

# Streamlit app
st.title('Survey Response Categorizer')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    if df is not None:
        st.write(df.head())

        # Check if 'email' column exists in the CSV
        if 'email' not in df.columns:
            st.error("No 'email' column found in the uploaded CSV file.")
        else:
            # Button to categorize responses and update CSV
            if st.button('Categorize Responses and Update CSV'):
                processed_df = process_dataframe(df)
                csv_bytes = to_csv_bytes(processed_df)
                st.download_button(
                    label="Download Categorized CSV",
                    data=csv_bytes,
                    file_name="categorized_survey_responses.csv",
                    mime="text/csv"
                )
                st.success("Responses categorized and CSV updated!")

        # Streamlit UI for user input categorization
        st.header("Categorize Your Own Response")
        user_input = st.text_area("Enter your response here:")

        if st.button("Categorize Response"):
            user_label = categorize_response(user_input)
            st.write(f"Categorized as: {user_label}")
