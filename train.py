import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_metric

# Load dataset
dataset = load_dataset("conll2003")

# Load tokenizer and model checkpoint
model_checkpoint = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Tokenize the dataset
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# Load model for token classification (with specified number of labels)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=9)

# Training arguments
training_args = TrainingArguments(
    output_dir="./models/ner_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Load metric
metric = load_metric("seqeval")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(-1)
    return metric.compute(predictions=predictions, references=labels)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train model
trainer.train()

# Ensure the output directory exists
output_dir = "./models/ner_model"
os.makedirs(output_dir, exist_ok=True)

# Make sure the model config has a model_type key.
# Since we started with a BERT checkpoint, we set it to "bert".
if not hasattr(model.config, "model_type") or not model.config.model_type:
    model.config.model_type = "bert"

# Save the trained model and tokenizer
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)