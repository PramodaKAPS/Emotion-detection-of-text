import pandas as pd
from datasets import load_dataset, Dataset
from imblearn.over_sampling import RandomOverSampler


def load_and_filter_goemotions(cache_dir, selected_emotions, num_train=0):
    """
    Load and filter the GoEmotions dataset for selected emotions.
    Returns filtered train, valid, test pandas DataFrames and the list of selected emotion indices.
    """
    print("Loading GoEmotions dataset...")
    dataset = load_dataset("go_emotions", "simplified", cache_dir=cache_dir)

    train_df = pd.DataFrame(dataset["train"])
    valid_df = pd.DataFrame(dataset["validation"])
    test_df = pd.DataFrame(dataset["test"])

    emotion_names = dataset["train"].features["labels"].feature.names
    selected_indices = [emotion_names.index(e) for e in selected_emotions if e in emotion_names]

    def filter_emotions(df):
        df = df.copy()
        df["labels"] = df["labels"].apply(lambda x: [label for label in x if label in selected_indices])
        df = df[df["labels"].apply(len) > 0]
        return df

    train_df = filter_emotions(train_df)
    valid_df = filter_emotions(valid_df)
    test_df = filter_emotions(test_df)

    # Extract first label as the target
    train_df["label"] = train_df["labels"].apply(lambda lbls: lbls[0])
    valid_df["label"] = valid_df["labels"].apply(lambda lbls: lbls[0])
    test_df["label"] = test_df["labels"].apply(lambda lbls: lbls[0])

    train_df = train_df[["text", "label"]]
    valid_df = valid_df[["text", "label"]]
    test_df = test_df[["text", "label"]]

    if num_train > 0:
        train_df = train_df.head(num_train)

    print(f"Train set size after filtering: {len(train_df)}")
    print(f"Validation set size after filtering: {len(valid_df)}")
    print(f"Test set size after filtering: {len(test_df)}")

    if train_df.empty:
        raise ValueError("No training data found after filtering. Check selected emotions.")

    return train_df, valid_df, test_df, selected_indices


def oversample_training_data(train_df):
    """
    Apply random oversampling to balance the training classes.
    """
    X = train_df["text"].values.reshape(-1, 1)
    y = train_df["label"]
    ros = RandomOverSampler(sampling_strategy="not majority", random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    resampled_df = pd.DataFrame({"text": X_resampled.flatten(), "label": y_resampled})
    print("Training data class distribution after oversampling:")
    print(resampled_df["label"].value_counts())

    return resampled_df


def prepare_tokenized_datasets(tokenizer, train_df, valid_df, test_df, max_length=256):
    """
    Tokenize the datasets using the provided tokenizer.
    """

    def tokenize_function(batch):
        return tokenizer(batch["text"], truncation=True, padding="longest", max_length=max_length)

    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    test_dataset = Dataset.from_pandas(test_df)

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_valid = valid_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    return tokenized_train, tokenized_valid, tokenized_test
