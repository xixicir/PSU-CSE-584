import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load your data
train_file_path = 'Processed_Training.csv'
test_file_path = 'Processed_Testing.csv'

# Load the CSV files
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# Create a huggingface datasets object
train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)
datasets = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the data
def tokenize_function(example):
    # Concatenate xi and xj with a separator token
    return tokenizer(example['xi'], example['xj'], truncation=True, padding='max_length', max_length=128)

# Apply tokenization to the train and test datasets
tokenized_datasets = datasets.map(tokenize_function, batched=True)

# Convert LLM column to numerical labels
label2id = {label: i for i, label in enumerate(train_data['LLM'].unique())}
id2label = {i: label for label, i in label2id.items()}

def convert_labels(example):
    example['labels'] = label2id[example['LLM']]
    return example

tokenized_datasets = tokenized_datasets.map(convert_labels, batched=False)

# Set format for PyTorch
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

# Initialize BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label2id))

# Move model to the appropriate device (GPU or CPU)
model.to(device)

# Define metrics for evaluation (without detailed print during training)
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')
    f1 = f1_score(labels, predictions, average='macro')

    return {
        'accuracy': accuracy,
        'precision_macro_avg': precision,
        'recall_macro_avg': recall,
        'f1_macro_avg': f1,
    }

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=10,
    load_best_model_at_end=True,
)

# Data collator for padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model on the test set (report metrics only here)
evaluation_results = trainer.evaluate()

# Generate the detailed classification report at the end
predictions, labels, _ = trainer.predict(tokenized_datasets['test'])
predictions = np.argmax(predictions, axis=1)

print("\nDetailed Classification Report:")
print(classification_report(labels, predictions, target_names=list(label2id.keys())))

print("\nEvaluation Results:")
print(evaluation_results)

# Save the fine-tuned model
model.save_pretrained('./fine-tuned-bert')
tokenizer.save_pretrained('./fine-tuned-bert')
