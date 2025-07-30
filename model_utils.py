import os
import numpy as np
import tensorflow as tf
from transformers import TFDebertaV3ForSequenceClassification, create_optimizer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def create_tf_datasets(tokenized_train, tokenized_valid, tokenized_test, tokenizer, selected_indices, batch_size=32):
    """
    Convert tokenized Hugging Face datasets to TensorFlow datasets.
    """
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
    mapping = {old: new for new, old in enumerate(selected_indices)}

    tokenized_train = tokenized_train.map(lambda x: {"label": mapping[x["label"]]})
    tokenized_valid = tokenized_valid.map(lambda x: {"label": mapping[x["label"]]})
    tokenized_test = tokenized_test.map(lambda x: {"label": mapping[x["label"]]})

    tf_train = tokenized_train.to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols=["label"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    tf_val = tokenized_valid.to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols=["label"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )
    tf_test = tokenized_test.to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols=["label"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )
    return tf_train, tf_val, tf_test

def setup_model_and_optimizer(model_name, num_labels, tf_train_dataset, epochs=5, lr=1e-5, cache_dir=None):
    model = TFDebertaV3ForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, cache_dir=cache_dir)
    steps = len(tf_train_dataset) * epochs
    optimizer, _ = create_optimizer(lr, 0, steps)
    return model, optimizer

def compile_and_train(model, optimizer, tf_train_dataset, tf_val_dataset, epochs=5):
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    model.fit(tf_train_dataset, validation_data=tf_val_dataset, epochs=epochs)
    return model

def save_model_and_tokenizer(model, tokenizer, path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Model saved at {path}")

def evaluate_model(model, tf_test):
    y_true = np.concatenate([y for x, y in tf_test], axis=0)
    y_pred_logits = model.predict(tf_test).logits
    y_pred = np.argmax(y_pred_logits, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }
