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

def add_prompt(text):
    #NOTE: Will the "imaginary" qualifier to patient have any effect on the output?
    #Preamble to each feature text
    # prompt = 'You are a professional psychiatrist, this is a comment from an imaginary patient, does he have suicidal thoughts? If yes, how dangerous is it on a scale from 1 to 10 (10 is the most dangerous)'
    prompt = 'You are a health specialist.  Answer the following text by stating whether or not the patient is suicidal or not, and give a rating from 1 to 10.'
    return prompt + text + '"\n'

#load dataset
df = pd.read_csv('~/Github repo/Suicide_Detection.csv', header=0)
#change labels to binary
df['class'] = [0 if x == "non-suicide" else 1 for x in df['class']]
#Add prompt to each text within the dataframe
df["text"] = df["text"].apply(add_prompt)

#Initialize a dataframe made of only the feature column
df = df[2:]

#store access credentials to "Hub" in cache
#create a txt file named creds.txt, then place read-only access token within the first line.
#NOTE: check out the os library
with open('/home/umflint.edu/brayclou/Github repo/creds.txt', "r", encoding="utf-8") as file:
    token = file.read()
# token="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"
login(token)

#Select model base
# alt: model = "meta-llama/Llama-2-7b-chat-hf"
model = "stabilityai/StableBeluga-7B"
# model = "NEU-HAI/mental-alpaca"
# model = "perplexity-ai/r1-1776"

#looks like the token allows access to the pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained(model, token=token, trust_remote_code=True)

#Initialize transformers pipeline using selected model, set to gpu
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    # device=0,
    trust_remote_code=True
)
pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id

#initialize variables outside for loop scope
#list of actual binary label
y = []
#list of predicted binary label, parallel array
y_pred = []
#non-binary label list for 1-10 predictions
scores = []
#combined output
out = []

torch.cuda.empty_cache()
torch.cuda.synchronize()

try:
    batch_size = 4
    for i in tqdm(range(0, len(df), batch_size), desc="Generating Responses"):
        if i > 130: break
        batch = df.iloc[i : i + batch_size]
        # txt = batch["text"].tolist()
        txt = [t for t in batch["text"].tolist() if isinstance(t, str) and len(t.strip()) > 0]
        if len(txt) == 0:
            print("Warning: Empty batch detected. Skipping iteration:", i)
            continue
        true = batch["class"].tolist()
        
        sequences = pipeline(
            txt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=4000,
            truncation=True,
            # max_total_length=4000, #should fix the error at iteration 1130
            batch_size=batch_size
        )
        print("Begin sequence sample:")
        print(sequences[0])

        for text, true1, response in zip(txt, true, sequences):
            output_text = response[0]['generated_text'][len(text):].strip

            #retrieve binary label and store in y_pred
            positive = re.findall(r'(?i)\byes\b', output_text)
            if len(positive) >= 1:
                y_pred.append(1)
            else:
                y_pred.append(0)
            y.append(int(true1))
            
            #store 1-10 prediction then append as integer to scores
            for match in re.finditer(r'\b(\d{1,2})/10\b', output_text):
                score = match.group(1)  # Extract the digit part of the score
                scores.append(int(score))
            
            #store output
            print("appending to out")
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
# plotting the confusion matrix
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#Save the results(or the predictions?) to a csv named preds.csv
my_df = pd.DataFrame(out)
my_df.to_csv('/home/umflint.edu/brayclou/Github repo/preds.csv', index=False, header=False)