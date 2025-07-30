import os
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, get_scheduler
from torch.optim import AdamW  # Using torch.optim.AdamW (non-deprecated)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader


def create_torch_datasets(tokenized_train, tokenized_valid, tokenized_test, selected_indices):
    mapping = {old: new for new, old in enumerate(selected_indices)}

    tokenized_train = tokenized_train.map(lambda x: {"labels": mapping[x["label"]]})  # Renamed to "labels" for model compatibility
    tokenized_valid = tokenized_valid.map(lambda x: {"labels": mapping[x["label"]]})
    tokenized_test = tokenized_test.map(lambda x: {"labels": mapping[x["label"]]})

    # Set format again after mapping (only relevant columns)
    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_valid.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    return tokenized_train, tokenized_valid, tokenized_test


def setup_model_and_optimizer(model_name, num_labels, epochs=5, lr=1e-5, cache_dir=None):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, cache_dir=cache_dir)
    optimizer = AdamW(model.parameters(), lr=lr)  # Using torch.optim.AdamW
    return model, optimizer


def compile_and_train(model, optimizer, tokenized_train, tokenized_valid, tokenizer, epochs=5, batch_size=32):  # Added tokenizer param
    # Use DataCollatorWithPadding for dynamic padding (as fallback)
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

    for epoch in range(epochs):
        model.train()
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        model.eval()
        for batch in valid_dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}
            with torch.no_grad():
                outputs = model(**batch)

    return model


def save_model_and_tokenizer(model, tokenizer, path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Model saved at {path}")


def evaluate_model(model, tokenized_test, tokenizer, batch_size=32):  # Added tokenizer param
    # Use DataCollatorWithPadding for evaluation
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
    
    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}

