
import os
from transformers import AutoTokenizer
from data_utils import load_and_filter_goemotions, oversample_training_data, prepare_tokenized_datasets
from model_utils import create_torch_datasets, setup_model_and_optimizer, compile_and_train, save_model_and_tokenizer, evaluate_model


def train_emotion_model(cache_dir, save_path, selected_emotions, num_train=0, epochs=5, batch_size=32):
    train_df, valid_df, test_df, sel_indices = load_and_filter_goemotions(cache_dir, selected_emotions, num_train)
    oversampled_train_df = oversample_training_data(train_df)
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", cache_dir=cache_dir)
    tokenized_train, tokenized_valid, tokenized_test = prepare_tokenized_datasets(tokenizer, oversampled_train_df, valid_df, test_df)
    
    tokenized_train, tokenized_valid, tokenized_test = create_torch_datasets(tokenized_train, tokenized_valid, tokenized_test, sel_indices)
    
    model, optimizer = setup_model_and_optimizer("microsoft/deberta-v3-base", len(selected_emotions), epochs, 1e-5, cache_dir)
    
    model = compile_and_train(model, optimizer, tokenized_train, tokenized_valid, epochs, batch_size)
    
    save_model_and_tokenizer(model, tokenizer, save_path)
    
    metrics = evaluate_model(model, tokenized_test, batch_size)
    
    return metrics


if __name__ == "__main__":
    cache_dir = "/root/huggingface_cache"
    save_path = "/root/emotion_model"
    emotions = ["anger", "sadness", "joy", "disgust", "fear", "surprise", "gratitude", "remorse", "curiosity", "neutral"]
    
    # Training parameters for improved accuracy
    config = {
        "num_train": 0,  # Full dataset
        "num_epochs": 5,  # Increased epochs
        "batch_size": 32,
        "learning_rate": 1e-5  # Lower LR for stability
    }
    
    print("🚀 Starting Emotion Detection Training")
    print("=" * 60)
    print(f"📊 Training Configuration:")
    print(f"   - Cache directory: {cache_dir}")
    print(f"   - Save path: {save_path}")
    print(f"   - Selected emotions: {emotions}")
    print(f"   - Training samples: Full dataset")
    print(f"   - Epochs: {config['num_epochs']}")
    print(f"   - Batch size: {config['batch_size']}")
    print(f"   - Learning rate: {config['learning_rate']}")
    print("-" * 60)
    
    try:
        metrics = train_emotion_model(
            cache_dir=cache_dir,
            save_path=save_path,
            selected_emotions=emotions,
            num_train=config["num_train"],
            epochs=config["num_epochs"],
            batch_size=config["batch_size"]
        )
        
        print("\n🎉 Training completed successfully!")
        print(f"📊 Final test results: {metrics}")
        print(f"💾 Model saved to: {save_path}")
        
    except Exception as e:
        print(f" Training failed: {e}")
        raise
