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


#load dataset
df = pd.read_csv('/home/umflint.edu/brayclou/Github repo/Suicide_Detection.csv', header=0)
#change labels to binary
df['class'] = [0 if x == "non-suicide" else 1 for x in df['class']]

#Initialize a dataframe made of only the feature column
df = df[2:]

#store access credentials to "Hub" in cache
token="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"
login(token)

#Select model base
# alt: model = "meta-llama/Llama-2-7b-chat-hf"
model = "stabilityai/StableBeluga-7B"

#looks like the token allows access to the pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained(model, token=token, trust_remote_code=True)

#Initialize transformers pipeline using selected model, set to gpu
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device=0,
    trust_remote_code=True
)

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
    #Loop through each row in the feature set, keep count with iterator i
    for i, pair in tqdm(enumerate(df.iterrows())):
        #subsample the first 100 rows
        # if i == 100:
        #     break
        #Print progress
        # if i % 10 == 0:
        #     print(f"Processing: {i}")

        #Store each row; predetermined risk score out of 10, the feature text, and the binary label respectively.
        score_out_of_10, txt, true = (pair[1].values)

        #NOTE: Will the "imaginary" qualifier to patient have any effect on the output?
        #Preamble to each feature text
        prompt = 'You are a professional psychiatrist, this is a comment from an imaginary patient, does he have suicidal thoughts? If yes, how dangerous is it on a scale from 1 to 10 (10 is the most dangerous)'

        #length of the prompt plus the feature text together
        offset = len((prompt + txt + '"\n'))

        #Pipeline to gain final llm output?
        sequences = pipeline(
            (prompt + txt + '"\n'),
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=4000
        )
        
        #retrieve binary label and store in y_pred
        positive = re.findall(r'(?i)\byes\b', sequences[0]['generated_text'][offset:])
        if len(positive) >= 1:
            y_pred.append(1)
        else:
            y_pred.append(0)
        y.append(int(true))
        
        #store 1-10 prediction then append as integer to scores
        for match in re.finditer(r'\b(\d)/10\b', sequences[0]['generated_text'][offset:]):
            score = match.group(1)  # Extract the digit part of the score
            scores.append(int(score))
        
        #store output
        out.append([(prompt+txt+'\n'), sequences[0]['generated_text'][offset:]])
except Exception as e:
    #output the error without damaging the process
    print("Error encountered:", e)

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
my_df.to_csv('Github repo/preds.csv', index=False, header=False)