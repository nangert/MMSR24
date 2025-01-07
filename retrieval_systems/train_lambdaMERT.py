import os
import csv
import bz2
import json
import shutil
import sys
import numpy as np
import pandas as pd
import subprocess
import tempfile
from tqdm import tqdm
import lightgbm as lgb
import pickle

"""
Trains a LambdaMART (LambdaMERT) learning to rank model using LightGBM.
Loads user-track interactions, track features, merges them, and trains a model
with LightGBM's ranker objective. After training, saves the model as a .pth file.
"""

###############################################################################
# Data loading
###############################################################################
def load_large_track_ids(meta_path: str) -> set:
    """
    Reads track metadata (the large file) and returns a set of valid track IDs.
    """
    df = pd.read_csv(meta_path, sep='\t')
    return set(str(x) for x in df["id"].unique())


def load_eval_track_ids(mmsr_path: str) -> set:
    """
    Reads the smaller subset (eval) track IDs (the MMSR part)
    to exclude them from training.
    """
    df = pd.read_csv(mmsr_path, sep="\t")
    return set(str(x) for x in df["id"].unique())


def load_user_track_counts(counts_path: str, large_ids: set, exclude_ids: set) -> pd.DataFrame:
    """
    Reads user-track interactions from a BZ2-compressed TSV.
    Filters out track IDs not in large_ids, or in exclude_ids.
    Returns a DataFrame [user_id, track_id, count].
    """
    rows = []
    with bz2.open(counts_path, mode="rt", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            track_id = str(row["track_id"])
            if track_id not in large_ids or track_id in exclude_ids:
                continue
            user_id = row["user_id"]
            c = float(row["count"])
            # Clip the range to keep counts more stable:
            c = max(min(c, 100.0), 1e-6)
            rows.append((user_id, track_id, c))

    df = pd.DataFrame(rows, columns=["user_id", "track_id", "count"])
    return df


def load_track_features(features_path: str, large_ids: set, exclude_ids: set) -> dict:
    """
    Reads track-level features (MFCC BoW, for example) from a BZ2-compressed TSV.
    Filters out track IDs not in large_ids, or in exclude_ids.
    Returns a dict {track_id -> np.array(features)}.
    """
    feature_dict = {}
    with bz2.open(features_path, mode="rt", encoding="utf-8") as f:
        df = pd.read_csv(f, sep="\t")
        cols = list(df.columns)
        feat_cols = cols[1:]  # everything except 'id'
        for _, row in df.iterrows():
            track_id = str(row["id"])
            if track_id not in large_ids or track_id in exclude_ids:
                continue
            feats = row[feat_cols].values.astype(float)
            feature_dict[track_id] = feats
    return feature_dict


###############################################################################
# Data preprocessing
###############################################################################
def stratified_subsample(df_counts: pd.DataFrame, sample_fraction: float = 0.1) -> pd.DataFrame:
    """
    Subsamples user-track interactions at the given fraction.
    Each user's subset is sampled proportionally.
    """
    subsampled = (
        df_counts
        .groupby("user_id", group_keys=False)
        .apply(lambda group: group.sample(frac=sample_fraction, random_state=100))
    )
    return subsampled


def build_ltr_dataframe(df_counts: pd.DataFrame, feat_dict: dict, chunk_size: int = 100000) -> pd.DataFrame:
    """
    Merges user-track interactions with track features.
    Produces a DataFrame:
      user_id, track_id, feat_1, feat_2, ..., feat_n, label
    chunk_size controls memory usage by partial merging.
    """
    rows = []
    total_rows = len(df_counts)
    for i in tqdm(range(0, total_rows, chunk_size), desc="Merging interactions with features"):
        chunk = df_counts.iloc[i: i + chunk_size]
        for _, row in chunk.iterrows():
            user_id = row["user_id"]
            track_id = str(row["track_id"])
            label = float(row["count"])  # Relevance label
            if track_id not in feat_dict:
                continue
            feats = feat_dict[track_id]
            merged_row = [user_id, track_id] + feats.tolist() + [label]
            rows.append(merged_row)

    # Determine feature count from first row
    if not rows:
        raise ValueError("No training rows found after merging. Check your data paths.")

    feat_count = len(rows[0]) - 3  # user_id + track_id + label => remainder are features
    cols = ["user_id", "track_id"] + [f"feat_{i+1}" for i in range(feat_count)] + ["label"]
    df_merged = pd.DataFrame(rows, columns=cols)
    return df_merged


def find_longest_slate(df_merged: pd.DataFrame) -> int:
    """
    Identifies the largest group size (number of items per user).
    Used to understand query grouping or potential padding (if needed).
    """
    slate_lengths = df_merged.groupby("user_id").size()
    longest_slate = slate_lengths.max()
    return int(longest_slate)


###############################################################################
# Train with LightGBM's LambdaRank
###############################################################################
def train_lambdamart(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feature_cols: list,
    label_col: str,
    user_col: str,
    num_boost_round: int = 1000
):
    """
    Trains a LambdaMART (LambdaRank) model using LightGBM with ranking objective.

    df_train, df_val: training and validation sets
    feature_cols: columns with numeric features
    label_col: column name for relevance label
    user_col: column name with user/group ID for grouping
    """
    # Sort by user_col so that grouping information is consistent
    df_train = df_train.sort_values(user_col)
    df_val = df_val.sort_values(user_col)

    X_train = df_train[feature_cols].values
    y_train = df_train[label_col].values
    # Grouping: number of items per user in the same order
    train_groups = df_train.groupby(user_col).size().tolist()

    X_val = df_val[feature_cols].values
    y_val = df_val[label_col].values
    val_groups = df_val.groupby(user_col).size().tolist()

    # Create LightGBM datasets
    train_dataset = lgb.Dataset(X_train, label=y_train, group=train_groups)
    val_dataset = lgb.Dataset(X_val, label=y_val, group=val_groups, reference=train_dataset)

    # Define the maximum label value
    max_label_value = max(df_train[label_col].max(), df_val[label_col].max())

    # Create a custom label_gain array
    label_gain = [i for i in range(max_label_value + 1)]
    print(f"Max Label value: {max_label_value}, Custom label_gain: {label_gain}")

    # LightGBM parameters for LambdaRank
    params = {
        "objective": "lambdarank",
        "metric": ["ndcg", "map"],  # Specify metrics to evaluate
        "eval_at": [5, 10],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "verbosity": 1,
        "label_gain": label_gain
    }

    model = lgb.train(
        params,
        train_dataset,
        num_boost_round=num_boost_round,
        valid_sets=[train_dataset, val_dataset],
        valid_names=["train", "val"]
    )

    return model


###############################################################################
# Main script to run everything
###############################################################################
def main():
    # Adapt these to your local structure
    train_counts_path = "dataset/train/userid_trackid_count.tsv.bz2"
    train_features_path = "dataset/train/id_mfcc_bow.tsv.bz2"
    large_meta_path = "dataset/train/id_information.csv"
    eval_subset_path = "dataset/id_information_mmsr.tsv"

    # Load sets of track IDs
    print("Loading track ID sets...")
    large_ids = load_large_track_ids(large_meta_path)
    print(f"Loaded {len(large_ids)} track IDs from the large metadata file.")

    eval_ids = load_eval_track_ids(eval_subset_path)
    print(f"Loaded {len(eval_ids)} evaluation track IDs to exclude from training.")

    # Load user-track interactions and track features
    print("\nLoading user-track interactions and track features...")
    df_counts = load_user_track_counts(train_counts_path, large_ids, eval_ids)
    print(f"Loaded {len(df_counts)} user-track interaction rows after filtering.")

    feat_dict = load_track_features(train_features_path, large_ids, eval_ids)
    print(f"Loaded {len(feat_dict)} track feature vectors.")

    # Subsample to reduce dataset size
    print("\nSubsampling user-track interactions...")
    sample_fraction = 0.01
    original_count = len(df_counts)
    df_counts = stratified_subsample(df_counts, sample_fraction)
    print(f"Subsampled from {original_count} to {len(df_counts)} interactions.")

    # Build a DataFrame merging interactions & features
    print("\nBuilding LTR DataFrame...")
    df_merged = build_ltr_dataframe(df_counts, feat_dict)
    print(f"Built LTR DataFrame with {len(df_merged)} rows.")

    print("\nSplitting data into train and validation sets...")
    unique_users = df_merged["user_id"].unique()
    print(f"Found {len(unique_users)} unique users.")

    np.random.seed(100)
    np.random.shuffle(unique_users)

    val_ratio = 0.2
    val_size = int(len(unique_users) * val_ratio)
    val_users = set(unique_users[:val_size])
    train_users = set(unique_users[val_size:])

    df_train = df_merged[df_merged["user_id"].isin(train_users)].copy()
    df_val = df_merged[df_merged["user_id"].isin(val_users)].copy()
    print(f"Split data into {len(df_train)} training rows and {len(df_val)} validation rows.")

    # Identify feature columns
    feature_cols = [c for c in df_train.columns if c.startswith("feat_")]
    label_col = "label"
    user_col = "user_id"
    print(f"Identified {len(feature_cols)} feature columns.")

    # Combine training and validation labels to ensure consistent mapping
    all_labels = np.unique(np.concatenate([df_train[label_col].values, df_val[label_col].values]))
    label_mapping = {label: idx for idx, label in enumerate(sorted(all_labels))}
    print(f"Label mapping: {label_mapping}")

    # Map labels in training and validation datasets
    df_train[label_col] = df_train[label_col].map(label_mapping)
    df_val[label_col] = df_val[label_col].map(label_mapping)

    # Verify the new labels
    print(f"New training labels: {df_train[label_col].unique()}")
    print(f"New validation labels: {df_val[label_col].unique()}")

    # Train the LambdaRank model
    print("\nraining the LambdaRank model...")
    model = train_lambdamart(df_train, df_val, feature_cols, label_col, user_col)
    print("LambdaRank model training complete.")

    print("\nSaving the trained model...")
    model_path = "lambdamart_model.pth"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}.")


if __name__ == "__main__":
    main()

