import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import AdamW, get_scheduler
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
import torch.amp as amp  # Mixed precision
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk   # Natural Language Toolkit
from nltk.corpus import stopwords
import re
from tqdm import tqdm

nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('stopwords') # Download stopwords list from NLTK

# =============
# Configuration
# =============

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Params
batch_size = 16
num_epochs = 2
learning_rate = 3e-5
max_length = 512
model_name = "distilbert-base-uncased"

# ===============
# Preprocess Data
# ===============

# Load Data
train_data = pd.read_csv('./dataset/train.csv')
test_data = pd.read_csv('./dataset/test.csv')

# Preprocess Data
def preprocess_text(text):
    #convert text to lower case
    text = text.lower()
    #remove digits and special characters using regular expressions
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    #tokenize the text
    text = nltk.word_tokenize(text)

    return text

# Remove stopwords
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    text_no_stopwords = [word for word in text if word not in stop_words]

    return text_no_stopwords

def lemmatization(text):
    lemmatizer = nltk.WordNetLemmatizer()
    lemmatizer_text = [lemmatizer.lemmatize(text) for text in text]

    return lemmatizer_text


def preprocess(text):
    #convert text to lower case, remove digits and special characters using regular expressions and remove stopwords combined together
    text = preprocess_text(text)
    filtered_text = remove_stopwords(text)
    lemmatizer_text = lemmatization(filtered_text)
    clean_text = ' '.join(lemmatizer_text)

    return clean_text

print("Preprocessing data...")
train_data["prompt"] = train_data["prompt"].apply(preprocess)
train_data["response_a"] = train_data["response_a"].apply(preprocess)
train_data["response_b"] = train_data["response_b"].apply(preprocess)

test_data["prompt"] = test_data["prompt"].apply(preprocess)
test_data["response_a"] = test_data["response_a"].apply(preprocess)
test_data["response_b"] = test_data["response_b"].apply(preprocess)

# Format Data
def make_pairs(row):
    try: prompt = row.prompt.encode("utf-8").decode("utf-8")
    except: prompt = ""
    try: response_a = row.response_a.encode("utf-8").decode("utf-8")
    except: response_a = ""
    try: response_b = row.response_b.encode("utf-8").decode("utf-8")
    except: response_b = ""
    row['options'] = [f"Prompt: {prompt}\n\nResponse: {response_a}", f"Prompt: {prompt}\n\nResponse: {response_b}"]
    return row

train_data = train_data.apply(make_pairs, axis=1)
test_data = test_data.apply(make_pairs, axis=1)

# Label (winner_model_a: 0, winner_model_b: 1, winner_tie: 2) -> map to numbers 0 1 2
train_data["label"] = train_data[["winner_model_a", "winner_model_b", "winner_tie"]].idxmax(axis=1).map({"winner_model_a": 0, "winner_model_b": 1, "winner_tie": 2})

# Split Data
train_data, valid_data = train_test_split(train_data, test_size=0.2, stratify=train_data["label"])

# Dataset Class
class PromptResponseDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        inputs_a = self.tokenizer(self.texts[idx][0], truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        inputs_b = self.tokenizer(self.texts[idx][1], truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")

        input_ids = torch.stack([inputs_a["input_ids"].squeeze(0), inputs_b["input_ids"].squeeze(0)], dim=0)
        attention_mask = torch.stack([inputs_a["attention_mask"].squeeze(0), inputs_b["attention_mask"].squeeze(0)], dim=0)


        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": label}

        return {"input_ids": input_ids, "attention_mask": attention_mask}



tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, add_prefix_space=False)

# Datasets and Dataloaders
train_dataset = PromptResponseDataset(train_data["options"].tolist(), train_data["label"].tolist(), tokenizer)
valid_dataset = PromptResponseDataset(valid_data["options"].tolist(), valid_data["label"].tolist(), tokenizer)
test_dataset = PromptResponseDataset(test_data["options"].tolist(), tokenizer=tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ============
# Define Model
# ============

# Model
class BertForSequenceClassification(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.2)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, 3)  # Explicitly for 3 classes
    
    def forward(self, input_ids, attention_mask):
        outputs_a = self.bert(input_ids=input_ids[:, 0, :], attention_mask=attention_mask[:, 0, :])
        outputs_b = self.bert(input_ids=input_ids[:, 1, :], attention_mask=attention_mask[:, 1, :])

        pooled_output_a = torch.mean(outputs_a.last_hidden_state, dim=1)
        pooled_output_b = torch.mean(outputs_b.last_hidden_state, dim=1)

        pooled_output_a = self.dropout(pooled_output_a)
        pooled_output_b = self.dropout(pooled_output_b)

        logits_a = self.classifier(pooled_output_a)
        logits_b = self.classifier(pooled_output_b)

        # Compute logits for "tie"
        logits_tie = torch.mean(torch.stack([logits_a, logits_b], dim=0), dim=0)

        # Concatenate logits for A, B, and Tie
        logits = torch.stack([logits_a, logits_b, logits_tie], dim=1)
        return logits



def dpo_loss(outputs, labels, beta=1.0):
    logits_a, logits_b, logits_tie = outputs[:, 0], outputs[:, 1], outputs[:, 2]
    
    # Pairwise difference for logits
    diff = logits_a - logits_b
    exp_diff = torch.exp(beta * diff)

    # Compute loss for each case
    loss = torch.where(
        labels == 0,  # Prefer A
        -torch.log(exp_diff / (1 + exp_diff)),
        torch.where(
            labels == 1,  # Prefer B
            -torch.log(1 / (1 + exp_diff)),
            -torch.log(torch.softmax(torch.stack([logits_a, logits_b, logits_tie], dim=-1), dim=-1)[:, 2])  # Tie
        )
    )
    return loss.mean()



# Model Initialization
model = BertForSequenceClassification(model_name).to(device)
# loss_fn = dpo_loss
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scaler = amp.GradScaler()  # Initialize GradScaler for mixed precision

# =============
# Training Loop
# =============

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

    for batch in loop:
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        with amp.autocast(device_type="cuda"):  # Enable mixed precision
            outputs = model(**inputs)
            loss = dpo_loss(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1}, Train Loss: {train_loss / len(train_dataloader):.4f}")


# Save Model
torch.save(model.state_dict(), "bert_model.pth")

# Prediction Loop with TQDM
state_dict = torch.load("bert_model.pth")
model.load_state_dict(state_dict)
model.to(device)
model.eval()
predictions = []
ids = test_data["id"].tolist()
probs = []

with torch.no_grad():
    loop = tqdm(test_dataloader, desc="Prediction", leave=True)  # TQDM progress bar
    for batch in loop:
        for k, v in batch.items():
            if k != "labels":
                batch[k] = v.to(device)

        outputs = model(**batch)

        # Apply softmax across all three logits
        logits = torch.softmax(outputs, dim=-1)
        probs.extend(logits.tolist())

        preds = torch.argmax(logits, dim=1)
        predictions.extend(preds.tolist())


# 保存到 DataFrame
submission = pd.DataFrame({
    "id": ids,
    "winner_model_a": [p[0] for p in probs],
    "winner_model_b": [p[1] for p in probs],
    "winner_tie": [p[2] for p in probs],
})

# 保存為 CSV
submission.to_csv("submission.csv", index=False)
print("Submission saved successfully!")