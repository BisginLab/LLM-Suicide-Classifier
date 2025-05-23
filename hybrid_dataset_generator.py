#Imports
import os
import torch
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import re

#This script takes the Suicide_Detection.csv dataset, and synthetically increases it seven-fold.

#Load OPENAI_API_KEY
load_dotenv()

#Clear CUDA
os.environ["TORCH_USE_CUDA_DSA"] = "1"
torch.cuda.empty_cache()
torch.cuda.synchronize()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

#Import Dataset
print("Loading Dataset...")
real_df = pd.read_csv('~/Github repo/Suicide_Detection.csv', header=0)
#change labels to binary
real_df['class'] = [0 if x == "non-suicide" else 1 for x in real_df['class']]
print("Dataset Loaded...")

#Initialize GPT pipeline
try:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    print("API key set...")
except: print("API key setting failed!")

mix_ratio = 7 #create this amount of synthetic datapoints per real datapoint
batch_size = 8
subset_size = 50000
#Initialize synth_df, a pd array that will store synthetic datapoints alongside their respective labels.
synth_df = pd.DataFrame(columns=["text", "class"])

def generate_synth(data):
    prompt = ""
    feature = data["text"]
    label = data["class"]
    
    if label == 0:
        prompt = f"Create {mix_ratio} messages similar to the input text.  Separate the messages with the word DIVIDER."
    elif label == 1:
        prompt = f"The input is a prompt from an imaginary health patient.  Do not respond to the message, instead create {mix_ratio} messages similar to the input text, and ensure that they display some form of suicidal ideation.  Separate the messages with the word DIVIDER."
    else: 
        print("damaged label detected!")
        return data

    #synthetic_points = generate with gpt pipeline
    synthetic_points = client.responses.create(#NOTE: Error is currently here.
        model="gpt-4o",
        instructions=prompt,
        input=feature,
    )
    #Split output upon devider, store in list
    synthetic_text = [text.strip() for text in re.split(r"\s*DIVIDER\s*", synthetic_points.output_text.strip()) if text.strip()]
    
    #instantiate a dataframe with outputs in "text", and an equal number of labels
    synth_output = pd.DataFrame({
        "text": synthetic_text,
        "class": [label] * len(synthetic_text)
    })
    if synth_output['text'].str.contains(r".*sorry.*feeling this way.*", regex=True, na=False).any():
        # print(synth_output['text'])
        # print("Blocked!")
        return data

    data = pd.concat([data, synth_output], ignore_index=True)        
    return data

#For each batch of real datapoints up to sub_size(use tqdm)
for i in tqdm(range(0, subset_size, batch_size), desc="Generating Synthetic Data"):
    current_batch = real_df.iloc[i:i+batch_size]
    #map or apply generate_synth to current batch, store as big list under name synth_batch
    batch_outputs = []
    for _, row in current_batch.iterrows():
        result = generate_synth(row)
        if isinstance(result, pd.DataFrame):
            batch_outputs.append(result)

    if batch_outputs:
        combined_batch = pd.concat(batch_outputs, ignore_index=True)
        synth_df = pd.concat([synth_df, combined_batch], ignore_index=True)

#randomly mix synth_df
synth_df_shuffled = synth_df#.sample(frac=1).reset_index(drop=True)

#print completion message, including display of changed class population counts.
print(f"Generation complete: dataset of size {subset_size} increased to {synth_df.shape}")

#save hybrid df to hybrid_suicide_detection.csv
synth_df_shuffled.to_csv('/home/umflint.edu/brayclou/Github repo/hybrid_suicide_detection.csv', index=False)