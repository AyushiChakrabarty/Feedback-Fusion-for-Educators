import os
import streamlit as st
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from transformers import pipeline, BertTokenizer, BertForSequenceClassification
import torch
from dotenv import load_dotenv
from groq import Groq
import io

# Load environment variables from .env file
load_dotenv()
professor_name = 'Greg Mayer Virtual Assistant'

# Access the API key from the environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the environment variables")

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")

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
    #"TC": ["technical", "tech issue", "system", "software", "hardware", "internet", "glitch", "wifi", "network", "connectivity"],
    #"PC": ["personal", "family", "home", "health", "wellbeing", "mental", "stress", "homework", "schedule", "time"],
    #"AC": ["academic", "study", "coursework", "exam", "assignment", "grade", "project", "research", "learning", "lecture"],
    "NC": ["no concern", "none", "nothing", "all good", "no problem", "okay", "fine", "satisfied", "happy", "no issues", "sorted", "No"]
}

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Function to categorize responses
#def categorize_response(response):
#    labels = {'LABEL_0': 'AC', 'LABEL_1': 'PC', 'LABEL_2': 'TC', 'LABEL_3': 'NC'}
#    result = classifier(response)
#    return labels[result[0]['label']]

def categorize_response(response):
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
    
    # Ensure that the predicted label is used if no keywords match
    return labels.get(predicted_label, "NC")  # Default to "NC" if the prediction is unclear

# Function to generate email content using Groq API
def generate_email_content(name, concerns_label, anything_else_label, response, professor_name):
    prompt = (
        f"Please generate only the content of the email without any introductory text or explanation like here's the email etc. "
        f"Generate a concise and professional email from a professor to a student named {name}, responding to their survey submission. "
        f"Keep the content brief, supportive, and to the point. Do not include any introductory text or mention of labels. "
        f"Their response was: '{response}'. "
        f"Structure the email with a simple greeting, a direct response addressing the student's concerns, and a polite closing. "
        f"Use the following format:\n\n"
        f"Hello {name},\n\n"
        f"[Concise content directly addressing the student's concerns]\n\n"
        f"Best regards,\n"
        f"{professor_name}"
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-8b-8192",
    )

    email_content = chat_completion.choices[0].message.content
    return email_content

# Function to send email
def send_email(to_email, subject, body):
    from_email = EMAIL_USER
    password = EMAIL_PASS

    if not from_email or not password:
        raise ValueError("Email credentials are not set in the environment variables.")

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email

    try:
        with smtplib.SMTP('smtp.office365.com', 587) as server:
            server.starttls()
            server.login(from_email, password)
            server.sendmail(from_email, to_email, msg.as_string())
        st.write(f"Email sent to {to_email}")
    except Exception as e:
        st.error(f"Failed to send email to {to_email}: {e}")    

# Function to process the DataFrame and append categorized columns
def process_dataframe(df):
    df['concerns_category'] = df['concerns'].apply(categorize_response)
    df['anything_else_category'] = df['anything else'].apply(categorize_response)
    return df

# Function to generate email content for each row and add to DataFrame
def generate_email_responses(df):
    df['email_response'] = df.apply(
        lambda row: generate_email_content(row['name'], row['concerns_category'], row['anything_else_category'], row['concerns'], professor_name),
        axis=1
    )
    return df

# Function to convert DataFrame to binary CSV
def to_csv_bytes(df):
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer.read()

# Streamlit app
st.title('Survey Response Categorizer and Email Sender')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
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

        # Button to generate email responses and update CSV
        if st.button('Generate Email Responses and Update CSV'):
            processed_df = process_dataframe(df)
            processed_df = generate_email_responses(processed_df)
            csv_bytes = to_csv_bytes(processed_df)
            st.download_button(
                label="Download CSV with Email Responses",
                data=csv_bytes,
                file_name="survey_responses_with_emails.csv",
                mime="text/csv"
            )
            st.success("Email responses generated and CSV updated!")

        # Button to send emails
        if st.button('Send Emails'):
            processed_df = process_dataframe(df)
            processed_df = generate_email_responses(processed_df)

            for index, row in processed_df.iterrows():
                email_body = row['email_response']
                send_email(row['email'], "Feedback to Survey", email_body)

            st.success("Emails sent successfully!")

# Mapping of labels to full phrases and icons
label_map = {
    "TC": ("Technical Concern", "💻"),
    "PC": ("Personal Concern", "🏠"),
    "AC": ("Academic Concern", "📚"),
    "NC": ("No Concern", "✅")
}

# Streamlit UI
st.header("Categorize Your Own Response")
user_input = st.text_area("Enter your response here:")

if st.button("Categorize Response"):
    user_label = categorize_response(user_input)
    full_label, icon = label_map[user_label]
    st.write(f"Categorized as: {icon} {full_label}")

