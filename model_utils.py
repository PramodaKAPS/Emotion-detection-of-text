import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    DebertaV3ForSequenceClassification,
    AdamW,
    get_scheduler,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class EmotionDataset(Dataset):
    def __init__(self, tokenized_dataset):
        self.dataset = tokenized_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.dataset.items() if key in ["input_ids", "attention_mask"]}
        label = torch.tensor(self.dataset["label"][idx])
        item["labels"] = label
        return item


def create_dataloaders(tokenized_train, tokenized_valid, tokenized_test, batch_size=32, tokenizer=None):
    train_dataset = EmotionDataset(tokenized_train)
    valid_dataset = EmotionDataset(tokenized_valid)
    test_dataset = EmotionDataset(tokenized_test)

    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt") if tokenizer else None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, device, epochs=5, learning_rate=1e-5):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    best_val_loss = float("inf")
    patience = 2
    patience_counter = 0

    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
                total_val_loss += outputs.loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                model.load_state_dict(torch.load("best_model.pt"))
                break

    return model


def evaluate_model(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            preds = torch.argmax(outputs.logits, dim=-1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(batch["labels"].cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")

    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}


def save_model_and_tokenizer(model, tokenizer, save_path):
    if not save_path.exists() and not save_path.is_dir():
        save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved to {save_path}")
