import os
import shutil
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_metric

# Define output directory
output_dir = "./models/ner_model"

# Remove the old model directory (if exists) to ensure a clean save
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

# Load the CoNLL2003 dataset
dataset = load_dataset("conll2003")

# Load the pretrained tokenizer and model checkpoint
model_checkpoint = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Tokenize the dataset; note that we use `is_split_into_words=True`
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# Load the model for token classification, specifying number of labels
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=9)

# Define training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Load evaluation metric
metric = load_metric("seqeval")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(-1)
    return metric.compute(predictions=predictions, references=labels)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Explicitly set model_type in the configuration if it is missing or empty.
if not hasattr(model.config, "model_type") or not model.config.model_type:
    model.config.model_type = "bert"

# Save the trained model and tokenizer
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)