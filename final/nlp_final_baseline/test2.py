# Import Libraries
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import AdamW, get_scheduler
import torch.nn as nn

# Configuration
class CFG:
    seed = 42
    model_name = "microsoft/deberta-v3-small"
    max_length = 512
    epochs = 3
    batch_size = 16
    lr = 2e-5
    label2name = {0: 'winner_model_a', 1: 'winner_model_b', 2: 'winner_tie'}
    name2label = {v: k for k, v in label2name.items()}

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Reproducibility
torch.manual_seed(CFG.seed)
np.random.seed(CFG.seed)

# Dataset Class
class PromptResponseDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=CFG.max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return {**inputs, "labels": label}
        return inputs

# Load Data
df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')

# Preprocessing
def make_pairs(row):
    try: prompt = row.prompt.encode("utf-8").decode("utf-8")
    except: prompt = ""
    try: response_a = row.response_a.encode("utf-8").decode("utf-8")
    except: response_a = ""
    try: response_b = row.response_b.encode("utf-8").decode("utf-8")
    except: response_b = ""
    row['options'] = [f"Prompt: {prompt}\n\nResponse: {response_a}", f"Prompt: {prompt}\n\nResponse: {response_b}"]
    return row

df = df.apply(make_pairs, axis=1)
test_df = test_df.apply(make_pairs, axis=1)
df["class_label"] = df[["winner_model_a", "winner_model_b", "winner_tie"]].idxmax(axis=1).map(CFG.name2label)

# Split Data
train_df, valid_df = train_test_split(df, test_size=0.2, stratify=df["class_label"])

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(CFG.model_name, use_fast=True, add_prefix_space=False)

# Datasets and Dataloaders
train_dataset = PromptResponseDataset(train_df["options"].tolist(), train_df["class_label"].tolist(), tokenizer)
valid_dataset = PromptResponseDataset(valid_df["options"].tolist(), valid_df["class_label"].tolist(), tokenizer)
test_dataset = PromptResponseDataset(test_df["options"].tolist(), tokenizer=tokenizer)

train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False)

# Model
class DebertaV3Classifier(nn.Module):
    def __init__(self, model_name, num_classes=3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self.backbone.config.hidden_size * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        # Get outputs for both response A and response B
        outputs_a = self.backbone(input_ids[:, 0], attention_mask=attention_mask[:, 0])
        outputs_b = self.backbone(input_ids[:, 1], attention_mask=attention_mask[:, 1])
        
        # Pool embeddings (mean-pooling across sequence length)
        pooled_output_a = torch.mean(outputs_a.last_hidden_state, dim=1)
        pooled_output_b = torch.mean(outputs_b.last_hidden_state, dim=1)

        # Concatenate the embeddings
        pooled_output = torch.cat((pooled_output_a, pooled_output_b), dim=-1)
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)
        return logits

model = DebertaV3Classifier(CFG.model_name).to(device)

# Loss, Optimizer, Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=CFG.lr)
scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * CFG.epochs)

from tqdm import tqdm
import torch.cuda.amp as amp  # Mixed precision

# Training Loop with TQDM and Mixed Precision
scaler = amp.GradScaler()  # Initialize GradScaler for mixed precision

for epoch in range(CFG.epochs):
    model.train()
    train_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG.epochs}", leave=True)  # TQDM progress bar

    for batch in loop:
        inputs = {k: v.squeeze(1).to(device) for k, v in batch.items() if k != "labels" and k != "token_type_ids"}
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        with amp.autocast():  # Enable mixed precision
            outputs = model(**inputs)
            loss = criterion(outputs, labels)

        # Backpropagation with mixed precision scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

        # Update TQDM progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1}, Train Loss: {train_loss / len(train_loader):.4f}")

# Save Model
torch.save(model.state_dict(), "deberta_model.pth")

# Prediction Loop with TQDM
model.eval()
predictions = []
with torch.no_grad():
    loop = tqdm(test_loader, desc="Prediction", leave=True)  # TQDM progress bar
    for batch in loop:
        inputs = {k: v.squeeze(1).to(device) for k, v in batch.items()}
        with amp.autocast():  # Enable mixed precision during inference
            outputs = model(**inputs)
        preds = torch.argmax(outputs, dim=-1).cpu().tolist()
        predictions.extend(preds)

# Submission
submission = test_df[["id"]].copy()
submission[CFG.label2name.values()] = predictions
submission.to_csv("submission.csv", index=False)
