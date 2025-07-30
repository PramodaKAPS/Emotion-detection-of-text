import torch
from torch.utils.data import DataLoader
from transformers import DebertaV3ForSequenceClassification, AdamW, get_scheduler, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoTokenizer

class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.dataset.items() if k in ['input_ids', 'attention_mask']}
        item['labels'] = self.dataset['label'][idx]
        return item

def create_dataloaders(tokenized_train, tokenized_valid, tokenized_test, batch_size=32):
    train_ds = EmotionDataset(tokenized_train)
    val_ds = EmotionDataset(tokenized_valid)
    test_ds = EmotionDataset(tokenized_test)
    data_collator = DataCollatorWithPadding(tokenizer=None, padding='longest')  # If tokenizer is needed, pass it here

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, device, epochs=5, lr=1e-5):
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    best_val_loss, patience, patience_count = float('inf'), 2, 0
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            total_loss += loss.item()
        val_loss = 0; model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                val_loss += outputs.loss.item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss; patience_count = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience_count += 1
            if patience_count >= patience:
                print("Early stopping triggered.")
                model.load_state_dict(torch.load("best_model.pt"))
                break
    return model

def evaluate_model(model, test_loader, device):
    model.eval(); y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            preds = outputs.logits.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(batch['labels'].cpu().numpy())
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

def save_model_and_tokenizer(model, tokenizer, save_path):
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
