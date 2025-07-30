import torch
from transformers import AutoTokenizer
from data_utils import load_and_filter_goemotions, oversample_training_data, prepare_tokenized_datasets
from model_utils import create_dataloaders, train_model, evaluate_model, save_model_and_tokenizer
import os

def main():
    cache_dir = "/root/huggingface_cache"
    save_path = "/root/emotion_model"
    emotions = ["anger", "sadness", "joy", "disgust", "fear", "surprise", "gratitude", "remorse", "curiosity", "neutral"]
    config = {"num_train": 0, "num_epochs": 5, "batch_size": 32, "learning_rate": 1e-5}
    print("🚀 Emotion Detection Training Pipeline Starting...")

    train_df, valid_df, test_df, sel_indices = load_and_filter_goemotions(cache_dir, emotions, config["num_train"])
    oversampled_train_df = oversample_training_data(train_df)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", cache_dir=cache_dir)
    tok_train, tok_valid, tok_test = prepare_tokenized_datasets(tokenizer, oversampled_train_df, valid_df, test_df)
    for ds in [tok_train, tok_valid, tok_test]:
        ds.set_format("torch")
    train_loader, val_loader, test_loader = create_dataloaders(tok_train, tok_valid, tok_test, batch_size=config["batch_size"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load("huggingface/pytorch-transformers", "deberta_v3_for_sequence_classification", "microsoft/deberta-v3-base", num_labels=len(emotions)).to(device)
    print(f"Model loaded on: {device}")

    model = train_model(model, train_loader, val_loader, device, epochs=config["num_epochs"], lr=config["learning_rate"])
    metrics = evaluate_model(model, test_loader, device)

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    save_model_and_tokenizer(model, tokenizer, save_path)

    print("\n🎉 Training completed successfully!")
    print(f"Test metrics: {metrics}")
    print(f"Model saved to: {save_path}")

if __name__ == "__main__":
    main()
