from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# Load IOB data
df = pd.read_csv('data/processed/sample_2020_gold_iob.csv')
df = df.dropna()  # Remove sentence boundaries

# Split into train and validation
vaers_ids = df['VAERS_ID'].unique()
train_ids, val_ids = train_test_split(vaers_ids, test_size=0.2, random_state=42)
train_df = df[df['VAERS_ID'].isin(train_ids)]
val_df = df[df['VAERS_ID'].isin(val_ids)]

# Define labels
label_list = ['O', 'B-ADE', 'I-ADE', 'B-DRUG', 'I-DRUG']
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for idx, label in enumerate(label_list)}

# Compute class weights
tag_counts = train_df['Tag'].value_counts()
total_tags = sum(tag_counts)
class_weights = torch.tensor([
    total_tags / (len(label_list) * tag_counts.get(label, 1)) if label == 'O' else
    10.0 * total_tags / (len(label_list) * tag_counts.get(label, 1)) if label in ['B-ADE', 'I-ADE'] else
    2.0 * total_tags / (len(label_list) * tag_counts.get(label, 1)) for label in label_list
], dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')

# Custom Dataset
class NERDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        self.sentences = df.groupby('VAERS_ID')['Token'].apply(list).values
        self.labels = df.groupby('VAERS_ID')['Tag'].apply(list).values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens = self.sentences[idx]
        labels = self.labels[idx]
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_len
        )
        label_ids = [label2id[label] for label in labels]
        label_ids = label_ids[:self.max_len] + [label2id['O']] * (self.max_len - len(label_ids))
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }

# Custom Trainer with weighted loss
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
model = AutoModelForTokenClassification.from_pretrained('dmis-lab/biobert-v1.1', num_labels=len(label_list), hidden_dropout_prob=0.3)

# Create datasets
train_dataset = NERDataset(train_df, tokenizer)
val_dataset = NERDataset(val_df, tokenizer)

# Data collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# Compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    true_labels = labels.flatten()
    pred_labels = predictions.flatten()
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted', zero_division=0)
    return {"precision": precision, "recall": recall, "f1": f1}

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=40,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,  # Simulate batch size 16
    save_steps=2000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=500,
    learning_rate=4e-5,
    weight_decay=0.01,
    warmup_steps=500,
    eval_strategy="steps",
    eval_steps=2000,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

# Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train
trainer.train()

# Save model
model.save_pretrained('models/biobert_ner')
tokenizer.save_pretrained('models/biobert_ner')
print("Saved BioBERT model to models/biobert_ner")
