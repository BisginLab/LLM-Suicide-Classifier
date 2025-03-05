#llm based classifier
#Interpreted from llm-inference.ipynb
#Brayden Cloutier
#2/26/2025

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

#Collects all directories and file names from /kaggle/input
#May not be needed, as I am using an absolute file path for now
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#load dataset
df = pd.read_csv('/home/umflint.edu/brayclou/Github repo/Suicide_Detection.csv', header=0)
#change labels to binary
df['class'] = [0 if x == "non-suicide" else 1 for x in df['class']]
#NOTE: why is train initialized but never used?  df holds the text features; the only relevant part for this exercise
train = df[:2]
df = df[2:] #initialized in dataframe format

#allows access to pretrained model
token="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"
login(token)

# model = "meta-llama/Llama-2-7b-chat-hf"
model = "stabilityai/StableBeluga-7B"

#looks like the token allows access to the pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained(model, token=token, trust_remote_code=True)
#A function that does the full ml process using a pretrained model from preprocessing to inference
pipeline = transformers.pipeline(
    #What type of task this model will be doing
    "text-generation",
    #Name of the model to be retrieved and used
    model=model,
    #The level of precision that will be used
    torch_dtype=torch.float16,
    #Specifically use gpu for the pipeline
    device=0,
    #bypass confirmation request from terminal
    trust_remote_code=True
)


y = []
y_pred = []
scores = []
out = []


try:

    for i, pair in tqdm(enumerate(df.iterrows())):

        #Only use a subsample of the first 100 data rows
        if i == 100:
            break
        
        #TODO: Figure out what is wrong with txt here
        #the variable underscore seems to be the 1-10 value
        _, txt, true = (pair[1].values)
        # print("Printing txt")
        # print(txt)
        # print("printing true")
        # print(true)
        # print("Printing underscore?")
        # print(_)

        #Does this not use a test set?
        #Correction: This is the preamble for the llm prompt that tells the llm what to do with the input
        #NOTE: Will the "imaginary" qualifier to patient have any effect on the output?
        prompt = 'You are a professional psychiatrist, this is a comment from an imaginary patient, does he have suicidal thoughts? If yes, how dangerous is it on a scale from 1 to 10 (10 is the most dangerous)'

        #Seems to get the character count for the llm prompt appended with text/txt appended to it.
        offset = len((prompt + txt + '"\n'))

        #Same as above line for txt
        #Is this storing the results fo the pipeline's use in sequences?
        sequences = pipeline(
        (prompt + txt + '"\n'),
    #TODO: Figure out why this sentence specifically was set aside
    #         'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
        #Adds some variation in output generation, allowing for a bit more creative writing
        do_sample=True,
        #Restricts word choice randomness from previous line to top 10 most likely
        top_k=10,
        #Only generate one response
        num_return_sequences=1,
        #Specifies which token marks the end of a response?
        eos_token_id=tokenizer.eos_token_id,
        #Restricts final length of output
        max_length=4000,
        )

        #Combs through generated responses.
        #Returns a prediction of 1 if the generated result contains the word "yes"
        #NOTE: Will need to check this.  Could mean a positive answer could be missed
        #  if the model generates something like "affirmative" instead of yes.
        positive = re.findall(r'(?i)\byes\b', sequences[0]['generated_text'][offset:])
        if len(positive) >= 1:
            y_pred.append(1)
        #else returns 0
        else:
            y_pred.append(0)
        y.append(int(true))
        
        for match in re.finditer(r'\b(\d)/10\b', sequences[0]['generated_text'][offset:]):
            score = match.group(1)  # Extract the digit part of the score
            scores.append(int(score))

        #NOTE: Changing text to txt to comply with earlier initialization?
        out.append([(prompt+txt+'\n'), sequences[0]['generated_text'][offset:]])
except Exception as e:
    #output the error without damaging the process
    print("Error encountered:", e)

print(len(y))

y = np.array(y)
y_pred = np.array(y_pred)

#NOTE: the commented out code was there before my modifications
#TODO: Figure out why it was left in the final kaggle notebook
# Find indices where array1 is not -1
# valid_indices = np.where(y_pred != -1)[0]

# print("Number of Unlabeled Texts: ", len(y)-len(valid_indices))

# Select elements from both arrays using these indices
# y = y[valid_indices]
# y_pred = y_pred[valid_indices]

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
my_df.to_csv('First/preds.csv', index=False, header=False)