import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import transformers
import torch
from huggingface_hub import login
import os
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from transformers import pipeline
from huggingface_hub import HfApi

#This is the current most up-to-date version of the classifier, built toward swappable huggingface pretrained models.

load_dotenv()
# user = os.getenv('username')
token = os.getenv('token')

os.environ["TORCH_USE_CUDA_DSA"] = "1"
torch.cuda.empty_cache()
torch.cuda.synchronize()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

#with open('creds.txt', "r", encoding="utf-8") as file:
    #token = file.read()
# token="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"
login(token)
# api = HfApi()
# api.set_access_token(token)

print("Huggingface credentials passed...")

# load dataset
df = pd.read_csv('Github repo/Suicide_Detection.csv', header=0)
# change labels to binary
df['class'] = [0 if x == "non-suicide" else 1 for x in df['class']]

# Initialize a dataframe made of only the feature column
df = df[2:]
print("Dataset Loaded...")
try:
    # Use a pipeline as a high-level helper
    pipe = pipeline("text-generation", model="Qwen/Qwen2.5-3B-Instruct")
except MemoryError as e:
    print("Memory error encountered: ", e)

# initialize variables outside for loop scope
# list of actual binary label
y = []
# list of predicted binary label, parallel array
y_pred = []
# non-binary label list for 1-10 predictions
scores = []
# combined output
out = []


def get_preds(input_text):
    # prompt_message = [
    #     {"role": "system",
    #      "content": 'You are a health specialist.  Answer the following text by stating whether or not the patient is suicidal or not by giving rating of 1 to 10 formatted as number/10'},
    #     {"role": "user", "content": input_text},
    # ]
    #alternate
    prompt_message = (
        'You are a health specialist. Answer the following text by stating whether or not the patient is suicidal '
        'or not by giving rating of 1 to 10 formatted as number/10.\n\nPatient text: "'
        + input_text + '"\n'
    )

    # Get the model's prediction
    response = pipe(prompt_message, max_new_tokens=50)
    returned_response = response[0]['generated_text']#[2]["content"]
    return returned_response


try:
    batch_size = 2
    sub_size = len(df)
    threshold = 6
    sub_df = df.iloc[50000 : sub_size] #NOTE: iloc gets commented out to process the entire dataset
    for i in tqdm(range(0, len(sub_df), batch_size), desc="Generating Responses"):
        # if i > sub_size: break
        batch = sub_df.iloc[i: i + batch_size]
        txt = [t for t in batch["text"].tolist() if isinstance(t, str) and len(t.strip()) > 0]
        if len(txt) == 0:
            print("Warning: Empty batch detected. Skipping iteration:", i)
            continue
        true = batch["class"].tolist()
        sequences = batch["text"].apply(get_preds)

        for text, true1, response in zip(txt, true, sequences):
        
            # if the response triggers gpt's apology message, return a max score of 10 USE FOR GPT ONLY
            # if re.search(r".*sorry.*feeling this way.*", response, re.IGNORECASE):
            #     output_text = "10/10"
            # else:
            output_text = response

            # store 1-10 prediction then append as integer to scores
            for match in re.finditer(r'\b(\d{1,2})/10\b', output_text):
                score = match.group(1)  # Extract the digit part of the score
                scores.append(int(score))

                # if int(score) >= threshold:
                #     y_pred.append(1)
                # else:
                #     y_pred.append(0)
                y_pred.append(int(score)/10)
                y.append(int(true1))

            # store output
            # print("appending to out")
            out.append([(text), output_text])

except Exception as e:
    # output the error without damaging the process
    print("Error encountered:", e)

if len(y) == 0 or len(y_pred) == 0:
    print("Error: No valid predictions. Using placeholder labels.")
    y = [0]
    y_pred = [0]

# convert binary to numpy array
y = np.array(y)
y_pred = np.array(y_pred)

print(f"True answers:      {y}")
print(f"Predicted answers: {y_pred}")
pred_labels = pd.DataFrame(columns=['True', 'Predicted'])
pred_labels['True'] = y
pred_labels['Predicted'] = y_pred
pred_labels.to_csv('Github repo/qwen_pred_scale_labels.csv', index=False)

# Save the results(or the predictions?) to a csv named preds.csv
my_df = pd.DataFrame(out)
my_df.to_csv('Github repo/qwen_pred_texts.csv', index=False,
             header=False)