import pandas as pd
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import numpy as np

#####################################################################################################################################
#To read lists from the opened file
def read_list_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            # Read the file content as a string
            file_content_str = file.read()

            # Safely evaluate the string as a Python literal (list)
            file_contents = ast.literal_eval(file_content_str)

        return file_contents

    except FileNotFoundError:
        print(f"The file '{file_path}' does not exist.")
    except (SyntaxError, ValueError) as e:
        print(f"Error while evaluating the file content: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        
#Create dataframe from the file that was read        
def df_from_file(file_path):
    listy = read_list_from_file(file_path)

    category = file_path.replace(".txt", "")
    df = pd.DataFrame({'response': listy, 'category': [category] * len(listy)})
    return df

#Files is a list of files to add it from. Put it all in one dataframe
def multiple_df(files):
    dfs = []
    for file in files:
        dfs.append(df_from_file(file))

    result_df = pd.concat(dfs, ignore_index=True)
    return result_df

#####################################################################################################################################
#-----------------------------------------------------------Model training related--------------------------------------------------#
#####################################################################################################################################

def __init__(self,model_path):
        self.tokenizer, self.loaded_model = self.load_model_and_tokenizer(4, model_path)

def load_model_and_tokenizer(self, labels=4, model_path='./Results/bert_model.pth'):
        # Gets the tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Load the model from the file
        loaded_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=labels)
        loaded_model.load_state_dict(torch.load(model_path))
        loaded_model.eval()  # Set the model to evaluation mode

        return tokenizer, loaded_model

def predict_with_loaded_model(self, input_text, tokenizer = None, loaded_model = None):
        if loaded_model is None:
            loaded_model = self.model_location
        if tokenizer is None:
            tokenizer = self.tokenizer
        #Tokenize and encode the text
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)

        #Forward pass through the model
        with torch.no_grad():
            outputs = loaded_model(**inputs)
        return outputs

def predict_column(self, df, text_column, category_mapping = {0: 'AC', 1: "PC", 2: "TC", 3: "NC"}):
        # Load model and tokenizer
        df[text_column] = df[text_column].astype(str)
        # Apply prediction function to each value in the specified column
        predictions = df[text_column].apply(lambda x: self.predict_with_loaded_model(x, self.tokenizer, self.loaded_model))

        # Extract predicted classes and probabilities
        df['Predicted_Class: ' + text_column] = predictions.apply(
            lambda x: torch.argmax(softmax(x.logits, dim=1), dim=1).item()).map(category_mapping)
        df['Probabilities: ' + text_column] = predictions.apply(lambda x: softmax(x.logits, dim=1).tolist())

        # Calculate confidence intervals
        df['Confidence_Interval: ' + text_column] = predictions.apply(
            lambda x: np.percentile(softmax(x.logits, dim=1).numpy(), [2.5, 97.5], axis=1))
        return df

#####################################################################################################################################
#-----------------------------------------------------Related to file management---------------------------------------------------#
#####################################################################################################################################

def predict_returnExcel(self,location_path,text_column, edit_orignal = True):
        df = pd.read_excel(location_path)
        df = self.predict_column(df,text_column)
        if(edit_orignal == True):
            df.to_excel(location_path)
        else:
            location_path = location_path + "withPredictions"
            df.to_excel(location_path)

def predict_returnCSV(self,location_path,text_column, edit_orignal = True):
        print("hello")
        df = pd.read_excel(location_path)
        df = self.predict_column(df,text_column)
        if(edit_orignal == True):
            df.to_csv(location_path)
        else:
            location_path = location_path + "withPredictions"
            df.to_csv(location_path)


#####################################################################################################################################
#-----------------------------------------------------Main exception checks---------------------------------------------------#
#####################################################################################################################################

def is_valid_excel_file(file_path):
        #Checks if the Excel file exists
        if not os.path.exists(file_path):
            print("Error: Excel file does not exist.")
            return False
        return True

#Returns true if column exists
def is_valid_column_name(file_path, column_name):
        try:
            #Read the Excel file and check if the specified column exists
            df = pd.read_excel(file_path)
            if column_name not in df.columns:
                print(f"Error: Column '{column_name}' does not exist in the Excel file.")
                return False
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            return False

        if column_name:
            return True
        else:
            print("Error: Column name cannot be empty.")
            return False

