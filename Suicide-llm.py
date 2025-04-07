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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

os.environ["TORCH_USE_CUDA_DSA"] = "1"
torch.cuda.empty_cache()
torch.cuda.synchronize()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def add_prompt(text):
    #NOTE: Will the "imaginary" qualifier to patient have any effect on the output?
    #Preamble to each feature text
    # prompt = 'You are a professional psychiatrist, this is a comment from an imaginary patient, does he have suicidal thoughts? If yes, how dangerous is it on a scale from 1 to 10 (10 is the most dangerous): '
    prompt = 'You are a health specialist.  Answer the following text by stating whether or not the patient is suicidal or not by giving rating of 1 to 10 formatted as number/10'
    return prompt + text + '"\n'

#load dataset
df = pd.read_csv('~/Github repo/Suicide_Detection.csv', header=0)
#change labels to binary
df['class'] = [0 if x == "non-suicide" else 1 for x in df['class']]
#Add prompt to each text within the dataframe
df["text"] = df["text"].apply(add_prompt)

#Initialize a dataframe made of only the feature column
df = df[2:]
print("Dataset Loaded...")

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

response = client.responses.create(
    model="gpt-4o",
    instructions='You are a health specialist. Give a rating from 1 to 10, depending on whether the following message indicates that its author is suicidal, 1 being not at all, and 10 being extremely suicidal.  Format the rating as number/10.  ',
    input="How do I check if a Python object is an instance of a class?",
)

print(response.output_text)

exit()

#initialize variables outside for loop scope
#list of actual binary label
y = []
#list of predicted binary label, parallel array
y_pred = []
#non-binary label list for 1-10 predictions
scores = []
#combined output
out = []

try:
    batch_size = 4
    sub_size = 50
    threshold = 6
    sub_df = df.iloc[0 : sub_size]
    for i in tqdm(range(0, len(sub_df), batch_size), desc="Generating Responses"):
        # if i > sub_size: break
        batch = sub_df.iloc[i : i + batch_size]
        txt = [t for t in batch["text"].tolist() if isinstance(t, str) and len(t.strip()) > 0]
        if len(txt) == 0:
            print("Warning: Empty batch detected. Skipping iteration:", i)
            continue
        true = batch["class"].tolist()

        with torch.no_grad(): #torch.autograd.set_detect_anomaly(True)
            sequences = pipeline(
                txt,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=100,
                # max_length=100,
                truncation=True,
                batch_size=batch_size
            )
        for text, true1, response in zip(txt, true, sequences):
            output_text = response[0]['generated_text'][len(text):].strip()
            
            #store 1-10 prediction then append as integer to scores
            for match in re.finditer(r'\b(\d{1,2})/10\b', output_text):
                score = match.group(1)  # Extract the digit part of the score
                scores.append(int(score))
            
                if int(score) >= threshold:
                    y_pred.append(1)
                else:
                    y_pred.append(0)
                y.append(int(true1))

            #store output
            # print("appending to out")
            out.append([(text), output_text])

except Exception as e:
    #output the error without damaging the process
    print("Error encountered:", e)

if len(y) == 0 or len(y_pred) == 0:
    print("Error: No valid predictions. Using placeholder labels.")
    y = [0]
    y_pred = [0]

#convert binary to numpy array
y = np.array(y)
y_pred = np.array(y_pred)

#Get prediction results
print("Accuracy: ",accuracy_score(y, y_pred))
print("Precision: ",precision_score(y, y_pred))
print("Recall: ",recall_score(y, y_pred))
print("F1: ",f1_score(y, y_pred))
# print out the failed results

print(f"True answers:      {y}")
print(f"Predicted answers: {y_pred}")    

#Save the results(or the predictions?) to a csv named preds.csv
my_df = pd.DataFrame(out)
my_df.to_csv('/home/umflint.edu/brayclou/Github repo/preds.csv', index=False, header=False)