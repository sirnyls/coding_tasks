import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader
import torch
import numpy as np

# 1. Load the dataset
data=pd.read_csv("final_results_paws.csv")
#dataset='PAWS'
#outcome_variable='helpfulness'

data=data.assign(text="Sentence 1: "+data.premise_+"\nAMR 1: "+data.amr_p+"\nSentence 2: "+data.hypothesis_+"\nAMR 2: "+data.amr_h)
data=data.assign(label=np.where(data.helpfulness<=0,0,1))
data=data.loc[:,['id','text','label']]
data=data.loc[~data.text.isna()]

train_data, val_data = train_test_split(data, test_size=0.1, stratify=data['label'])

# 2. Tokenize the data
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

def tokenize_data(data):
    tokenized = tokenizer(data['text'].tolist(), padding=True, truncation=True, max_length=512, return_tensors='pt')
    tokenized['label'] = torch.tensor(data['label'].tolist())
    return tokenized

train_dataset = tokenize_data(train_data)
val_dataset = tokenize_data(val_data)

# 3. Prepare data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# 4. Load the model
model = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=2).cuda()

# 5. Train the model
EPOCHS = 3  # You can adjust the number of epochs as needed
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    for batch in train_loader:
        inputs = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        labels = batch['label'].cuda()

        optimizer.zero_grad()
        outputs = model(inputs, attention_mask=attention_mask)[0]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation (you can expand this part for more comprehensive validation metrics)
    model.eval()
    total_correct = 0
    total_count = 0
    for batch in val_loader:
        inputs = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        labels = batch['label'].cuda()

        with torch.no_grad():
            outputs = model(inputs, attention_mask=attention_mask)[0]
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_count += labels.size(0)

    accuracy = total_correct / total_count
    print(f"Epoch {epoch + 1}/{EPOCHS} | Validation Accuracy: {accuracy * 100:.2f}%")
