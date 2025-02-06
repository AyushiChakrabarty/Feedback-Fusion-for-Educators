import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.nn.functional import softmax
import numpy as np
from sklearn.metrics import accuracy_score

def load_model_and_tokenizer(model_name='distilbert-base-uncased', labels=4):
    # Gets the tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    # Load the pre-trained model
    loaded_model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=labels)
    loaded_model.eval()  # Set the model to evaluation mode

    return tokenizer, loaded_model


def predict_with_loaded_model(input_text, tokenizer, loaded_model):
    # Tokenize and encode the text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)

    # Forward pass through the model
    with torch.no_grad():
        outputs = loaded_model(**inputs)

    return outputs


def predict_column(df, category_mapping, text_column, model_name='distilbert-base-uncased'):
    # Load model and tokenizer
    tokenizer, loaded_model = load_model_and_tokenizer(model_name)
    df[text_column] = df[text_column].astype(str)

    # Apply prediction function to each value in the specified column
    predictions = df[text_column].apply(lambda x: predict_with_loaded_model(x, tokenizer, loaded_model))

    # Extract predicted classes and probabilities
    df['Predicted_Class: ' + text_column] = predictions.apply(lambda x: torch.argmax(softmax(x.logits, dim=1), dim=1).item()).map(category_mapping)
    df['Probabilities: ' + text_column] = predictions.apply(lambda x: softmax(x.logits, dim=1).tolist())

    # Calculate confidence intervals (dummy example, actual method depends on model specifics)
    df['Confidence_Interval: ' + text_column] = predictions.apply(lambda x: np.percentile(softmax(x.logits, dim=1).numpy(), [2.5, 97.5], axis=1))

    return df

def calculate_accuracy(df, text_column, ground_truth_column):
    # Calculate accuracy if ground truth is available
    if ground_truth_column in df.columns:
        y_true = df[ground_truth_column]
        y_pred = df['Predicted_Class: ' + text_column].map({v: k for k, v in category_mapping.items()})
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy
    else:
        return None