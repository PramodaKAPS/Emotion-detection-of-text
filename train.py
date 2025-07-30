import os
from transformers import AutoTokenizer
from data_utils import load_and_filter_goemotions, oversample_training_data, prepare_tokenized_datasets
from model_utils import create_tf_datasets, setup_model_and_optimizer, compile_and_train, save_model_and_tokenizer, evaluate_model

def main():
    cache_dir = "/root/huggingface_cache"
    save_path = "/root/emotion_model"
    emotions = ["anger", "sadness", "joy", "disgust", "fear", "surprise",
                "gratitude", "remorse", "curiosity", "neutral"]

    config = {
        "num_train": 0,       # Use full dataset
        "num_epochs": 5,      # Increased epochs for better learning
        "batch_size": 32,
        "learning_rate": 1e-5 # Lower learning rate for stable training
    }

    print("🚀 Starting Emotion Detection Training")
    print("=" * 60)
    print(f"Configuration:")
    print(f"- Cache directory: {cache_dir}")
    print(f"- Save path: {save_path}")
    print(f"- Emotions: {emotions}")
    print(f"- Training on full dataset")
    print(f"- Epochs: {config['num_epochs']}")
    print(f"- Batch size: {config['batch_size']}")
    print(f"- Learning rate: {config['learning_rate']}")
    print("-" * 60)

    try:
        train_df, valid_df, test_df, sel_indices = load_and_filter_goemotions(cache_dir, emotions, config["num_train"])
        oversampled_train_df = oversample_training_data(train_df)

        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", cache_dir=cache_dir)
        tokenized_train, tokenized_valid, tokenized_test = prepare_tokenized_datasets(tokenizer, oversampled_train_df, valid_df, test_df)

        tf_train, tf_val, tf_test = create_tf_datasets(tokenized_train, tokenized_valid, tokenized_test, tokenizer, sel_indices, config["batch_size"])

        model, optimizer = setup_model_and_optimizer("microsoft/deberta-v3-base", len(emotions), tf_train, config["num_epochs"], config["learning_rate"], cache_dir)

        model = compile_and_train(model, optimizer, tf_train, tf_val, config["num_epochs"])

        save_model_and_tokenizer(model, tokenizer, save_path)

        metrics = evaluate_model(model, tf_test)
        print("\n🎉 Training completed successfully!")
        print(f"Test metrics: {metrics}")
        print(f"Model saved to: {save_path}")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
