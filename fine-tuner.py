#Load Libraries
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
import os
from dotenv import load_dotenv

#This is my custom fine-tuning script currently set to fine-tune Llama-3.2-3B-Instruct on the hybrid_only_suicide.csv dataset
#Temporarily replaced by test-fine-tuner.py

#Log in with environment variable
load_dotenv()
token = os.getenv('token')
login(token)

# Load model directly
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
dataset = load_dataset("csv", data_files="/home/umflint.edu/brayclou/Github repo/hybrid_only_suicide.csv")

#NOTE: code inspired by huggingface documentation
# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True, batch_size=16)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="/home/umflint.edu/brayclou/Github repo/Pretrained-Model",
    eval_strategy="no",
    num_train_epochs=4,
    push_to_hub=False,
)

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Train the model
trainer.train()

#Save the model to output file
model.save_pretrained("/home/umflint.edu/brayclou/Github repo/Pretrained-Model")
tokenizer.save_pretrained("/home/umflint.edu/brayclou/Github repo/Pretrained-Model")