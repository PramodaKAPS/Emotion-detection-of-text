import os
import torch
from pathlib import Path
from transformers import AutoTokenizer, DebertaV3ForSequenceClassification
from data_utils import load_and_filter_goemotions, oversample_training_data, prepare_tokenized_datasets
from model_utils import create_dataloaders, train_model, evaluate_model, save_model_and_tokenizer


def main():
    cache_dir = Path("/root/huggingface_cache")
    save_path = Path("/root/emotion_model")
    emotions = [
        "anger", "sadness", "joy", "disgust", "fear",
        "surprise", "gratitude", "remorse", "curiosity", "neutral"
    ]

    config = {
        "num_train": 0,          # Use full dataset
        "num_epochs": 5,
        "batch_size": 32,
        "learning_rate": 1e-5
    }

    print("Starting emotion detection training pipeline...")
    train_df, valid_df, test_df, sel_indices = load_and_filter_goemotions(cache_dir, emotions, config["num_train"])
    oversampled_train_df = oversample_training_data(train_df)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", cache_dir=cache_dir)
    tokenized_train, tokenized_valid, tokenized_test = prepare_tokenized_datasets(
        tokenizer, oversampled_train_df, valid_df, test_df
    )

    # Prepare the tokenized dataset for PyTorch
    for dataset in [tokenized_train, tokenized_valid, tokenized_test]:
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_loader, val_loader, test_loader = create_dataloaders(
        tokenized_train, tokenized_valid, tokenized_test, config["batch_size"], tokenizer
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DebertaV3ForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-base", num_labels=len(emotions)
    ).to(device)

    model = train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=config["num_epochs"],
        learning_rate=config["learning_rate"]
    )

    metrics = evaluate_model(model, test_loader, device)
    print("Test set evaluation metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    save_model_and_tokenizer(model, tokenizer, save_path)
    print("Training completed and model saved.")


if __name__ == "__main__":
    main()
