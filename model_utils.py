import os
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, get_scheduler
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix  # Added for confusion matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler


def create_torch_datasets(tokenized_train, tokenized_valid, tokenized_test, selected_indices):
    mapping = {old: new for new, old in enumerate(selected_indices)}

    def remap_labels(example):
        if "label" in example:
            example["labels"] = mapping[example["label"]]
        return example

    tokenized_train = tokenized_train.map(remap_labels)
    tokenized_valid = tokenized_valid.map(remap_labels)
    tokenized_test = tokenized_test.map(remap_labels)

    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_valid.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    if "label" in tokenized_train.column_names:
        tokenized_train = tokenized_train.remove_columns(["label"])
        tokenized_valid = tokenized_valid.remove_columns(["label"])
        tokenized_test = tokenized_test.remove_columns(["label"])

    return tokenized_train, tokenized_valid, tokenized_test


def setup_model_and_optimizer(model_name, num_labels, epochs=5, lr=1e-5, cache_dir=None):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, cache_dir=cache_dir)
    optimizer = AdamW(model.parameters(), lr=lr)
    return model, optimizer


def compile_and_train(model, optimizer, tokenized_train, tokenized_valid, tokenizer, epochs=5, batch_size=32):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        tokenized_train, 
        shuffle=True, 
        batch_size=batch_size, 
        collate_fn=data_collator
    )
    valid_dataloader = DataLoader(
        tokenized_valid, 
        batch_size=batch_size, 
        collate_fn=data_collator
    )

    num_training_steps = epochs * len(train_dataloader)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    scaler = GradScaler()
    accumulation_steps = 4  # Accumulate for effective larger batch size

    for epoch in range(epochs):
        model.train()
        train_progress = tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training")
        optimizer.zero_grad()
        for step, batch in enumerate(train_progress):
            batch = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}
            with autocast():
                outputs = model(**batch)
                loss = outputs.loss / accumulation_steps
            scaler.scale(loss).backward()
            
            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            
            train_progress.set_postfix(loss=loss.item())

        model.eval()
        val_progress = tqdm(valid_dataloader, desc=f"Epoch {epoch+1} Validation")
        for batch in val_progress:
            batch = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}
            with torch.no_grad(), autocast():
                outputs = model(**batch)

    return model


def save_model_and_tokenizer(model, tokenizer, path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Model saved at {path}")


def evaluate_model(model, tokenized_test, tokenizer, batch_size=32):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    test_dataloader = DataLoader(
        tokenized_test, 
        batch_size=batch_size, 
        collate_fn=data_collator
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        y_pred.extend(predictions.cpu().numpy())
        y_true.extend(batch["labels"].cpu().numpy())
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    
    # Added: Compute and print confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)  # Prints as a 2D array; rows=true labels, columns=predicted labels
    
    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}
