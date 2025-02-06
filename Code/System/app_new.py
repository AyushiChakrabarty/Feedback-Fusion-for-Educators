import os
import streamlit as st
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from transformers import pipeline, BertTokenizer, BertForSequenceClassification
import torch
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import io
import re
import string
from matplotlib.patches import Patch

# Load environment variables
load_dotenv()

# API keys and email credentials
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")

# Preprocessing functions
def clean_text(text):
    if pd.isnull(text):
        return ''
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text.strip()

def preprocess_dataframe(df):
    df['concerns'] = df['concerns'].fillna('')
    df['anything else'] = df['anything else'].fillna('')
    df['concerns'] = df['concerns'].apply(clean_text)
    df['anything else'] = df['anything else'].apply(clean_text)
    return df

@st.cache_resource
def load_model():
    model_path = "bert_model_small.pth"
    model = BertForSequenceClassification.from_pretrained('prajjwal1/bert-tiny', num_labels=4)
    model.load_state_dict(torch.load(model_path))
    tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
    return model, tokenizer

model, tokenizer = load_model()
classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)

keywords = {"NC": ["no concern", "none", "nothing", "all good", "no problem", "okay", "fine", "satisfied", "happy", "no issues", "sorted", "No"]}
sensitive_words = ["anxiety", "stress", "depression", "abuse", "bully", "harassment", "mental health", "suicidal"]

def categorize_response(response):
    response_lower = response.lower()
    result = classifier(response)
    predicted_label = result[0]['label']
    for label, words in keywords.items():
        if any(word in response_lower for word in words):
            return label
    labels = {'LABEL_0': 'AC', 'LABEL_1': 'PC', 'LABEL_2': 'TC', 'LABEL_3': 'NC'}
    return labels.get(predicted_label, "NC")

def filter_sensitive_responses(df):
    return df[df['concerns'].str.contains('|'.join(sensitive_words), case=False, na=False)]

def plot_category_distribution(df):
    counts = df['concerns_category'].value_counts(normalize=True) * 100
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
    explode = [0.1 if count == max(counts) else 0 for count in counts]
    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, colors=colors, explode=explode, pctdistance=0.85)
    
    for text in texts:
        text.set_fontsize(12)
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_color('black')

    # Draw a circle at the center to make it a donut chart
    center_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(center_circle)

    ax.set_title("Percentage of Each Concern Category", fontsize=16)
    st.pyplot(fig)

def to_csv_bytes(df):
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer.read()

st.title("\U0001F4C8 Survey Response Categorizer and Insights Dashboard")
st.markdown(
    """
    Welcome to the **Survey Categorizer Tool**! This tool uses a **BERT model** to classify survey responses into the following categories:
    - **AC**: Academic Concerns  
    - **PC**: Personal Concerns  
    - **TC**: Technical Concerns  
    - **NC**: No Concerns  

    ### Features:
    - **Upload CSV Files**: Categorize all responses.
    - **Visualize Data**: Pie charts for category distribution.
    - **Sensitive Responses**: Highlight and download sensitive responses.
    - **Custom Input**: Test with individual responses.
    """
)

uploaded_file = st.file_uploader("\U0001F4BE Upload a CSV file with 'name', email', 'concerns', and 'anything else' columns", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
        st.write("Preview of the uploaded data:")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
    else:
        if 'concerns' not in df.columns or 'email' not in df.columns:
            st.error("The CSV must have 'concerns' and 'email' columns.")
        else:
            if st.button("\U0001F527 Categorize Responses"):
                df = preprocess_dataframe(df)
                df['concerns_category'] = df['concerns'].apply(categorize_response)
                df['anything_else_category'] = df['anything else'].apply(categorize_response)

                st.success("Responses categorized successfully!")
                st.write("Categorized Data:")
                st.dataframe(df)

                st.subheader("\U0001F4CA Category Distribution")
                plot_category_distribution(df)

                sensitive_df = filter_sensitive_responses(df)
                if not sensitive_df.empty:
                    st.subheader("\U0001F512 Sensitive Responses")
                    st.dataframe(sensitive_df)

                    sensitive_csv = to_csv_bytes(sensitive_df)
                    st.download_button("Download Sensitive Responses CSV", data=sensitive_csv, file_name="sensitive_responses.csv", mime="text/csv")

                categorized_csv = to_csv_bytes(df)
                st.download_button("Download Categorized Data CSV", data=categorized_csv, file_name="categorized_responses.csv", mime="text/csv")

                pc_responses = df[df['concerns_category'] == 'PC']
                if not pc_responses.empty:
                    st.subheader("\U0001F4DD Personal Concerns Data")
                    st.dataframe(pc_responses)
                    pc_csv = to_csv_bytes(pc_responses[['name', 'email', 'concerns', 'anything else']])
                    st.download_button("Download Personal Concerns CSV", data=pc_csv, file_name="personal_concerns.csv", mime="text/csv")

st.markdown("---")
st.header("🔍 Try Manual Categorization")
st.write("Enter a student response below to see its category.")
user_input = st.text_area("Enter your response here:")
if st.button("Categorize Response"):
    user_label = categorize_response(user_input)
    st.write(f"Categorized as: **{user_label}**")

st.markdown("---")
st.info("🔒 Your data is not stored and remains private.")