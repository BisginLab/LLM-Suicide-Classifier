import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType


#This is a fine-tuning script found online, currently set to fine-tune Llama-3.2-3B-Instruct on the hybrid_only_suicide.csv dataset

# === LOAD DATA FROM CSV ===
df = pd.read_csv("/home/umflint.edu/brayclou/Github repo/hybrid_only_suicide.csv")
df = df[["text"]].dropna()
dataset = Dataset.from_pandas(df)
# === LOAD MODEL AND TOKENIZER ===
model_id = "meta-llama/Llama-3.2-3B-Instruct"  # or LLaMA 3 if you have access
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,
    device_map="auto"
)
# === LoRA CONFIGURATION ===
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, peft_config)
# === TOKENIZATION ===
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, max_length=512, padding="max_length")
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
# === TRAINING ARGS ===
training_args = TrainingArguments(
    output_dir="./llama-csv-finetuned",
    per_device_train_batch_size=6, #NOTE: play around with batch size
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    save_strategy="epoch",
    eval_strategy="no"
)
# === TRAINER ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)
# === TRAIN ===
trainer.train()
# === SAVE MODEL AND TOKENIZER ===
trainer.save_model("./llama-csv-finetuned")
tokenizer.save_pretrained("./llama-csv-finetuned")