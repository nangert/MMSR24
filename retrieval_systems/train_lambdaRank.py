# train_lambdaRank.py
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
Trains a LambdaRank learning to rank model using LightGBM.
Loads user-track interactions, track features, merges them, and trains a model
with LightGBM's ranker objective. After training, saves the model as a .pth file.
"""

# Data loading
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


# Data preprocessing
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

    if not rows:
        raise ValueError("No training rows found after merging. Check your data paths.")

    feat_count = len(rows[0]) - 3  # user_id + track_id + label => remainder are features
    cols = ["user_id", "track_id"] + [f"feat_{i+1}" for i in range(feat_count)] + ["label"]
    df_merged = pd.DataFrame(rows, columns=cols)
    return df_merged


# Ensure all tracks remain (at least 1% of their occurrences)
def sample_one_percent_with_min5_coverage(
        df_counts: pd.DataFrame,
        global_fraction: float = 0.01,  # 1% overall
        coverage_count: int = 5,
        random_seed: int = 100
) -> pd.DataFrame:
    """
    1) Randomly sample 1% of the entire DataFrame.
    2) For each track, ensure it has at least 5 rows in the sample.
    3) For each user, ensure it has at least 5 rows in the sample.
    4) If final set exceeds 1%, try to reduce it down from non-coverage rows.
       Coverage rows are mandatory, so if coverage alone exceeds 1%, final size > 1%.

    This version sorts the original dataframe once for tracks and once for users,
    making coverage steps faster than repeated groupby calls.
    """
    total_rows = len(df_counts)
    desired_rows = max(int(round(total_rows * global_fraction)), 1)

    df_sampled = df_counts.sample(frac=global_fraction, random_state=random_seed)

    # Track which coverage rows we must keep, so we can avoid dropping them
    coverage_mandatory_keys = set()  # (user_id, track_id, count) to keep

    # Ensure each track appears at least 5 times in the sample
    track_counts_in_sample = df_sampled.groupby("track_id").size()
    needed_by_track = {}
    all_tracks = df_counts["track_id"].unique()
    for trk in all_tracks:
        have = track_counts_in_sample.get(trk, 0)
        if have < coverage_count:
            needed_by_track[trk] = coverage_count - have

    if needed_by_track:
        # Pre-sort df_counts once by "count" descending
        df_sorted_by_track = df_counts.sort_values(["track_id", "count"],
                                                   ascending=[True, False])

        # Single pass groupby to fetch needed coverage rows
        def take_track_coverage(group):
            trk_id = group.name
            needed = needed_by_track.get(trk_id, 0)
            if needed <= 0:
                return pd.DataFrame(columns=group.columns)
            # pick top 'needed' rows from the already descending-sorted group
            return group.iloc[:needed]

        coverage_for_tracks = (
            df_sorted_by_track
            .groupby("track_id", group_keys=False)
            .apply(take_track_coverage))

        # Mark coverage keys
        for row in coverage_for_tracks[["user_id", "track_id", "count"]].itertuples(index=False):
            coverage_mandatory_keys.add((row.user_id, row.track_id, row.count))

        # Merge coverage rows in a single step
        df_sampled = pd.concat([df_sampled, coverage_for_tracks], ignore_index=True)
        df_sampled.drop_duplicates(subset=["user_id", "track_id", "count"], inplace=True)

    # Ensure each user appears at least 5 times in the sample
    user_counts_in_sample = df_sampled.groupby("user_id").size()
    needed_by_user = {}
    all_users = df_counts["user_id"].unique()
    for usr in all_users:
        have = user_counts_in_sample.get(usr, 0)
        if have < coverage_count:
            needed_by_user[usr] = coverage_count - have

    if needed_by_user:
        # Pre-sort df_counts by "count" descending, grouping by user
        df_sorted_by_user = df_counts.sort_values(["user_id", "count"],
                                                  ascending=[True, False])

        def take_user_coverage(group):
            usr_id = group.name
            needed = needed_by_user.get(usr_id, 0)
            if needed <= 0:
                return pd.DataFrame(columns=group.columns)
            return group.iloc[:needed]

        coverage_for_users = (
            df_sorted_by_user
            .groupby("user_id", group_keys=False)
            .apply(take_user_coverage))

        # Mark coverage keys
        for row in coverage_for_users[["user_id", "track_id", "count"]].itertuples(index=False):
            coverage_mandatory_keys.add((row.user_id, row.track_id, row.count))

        df_sampled = pd.concat([df_sampled, coverage_for_users], ignore_index=True)
        df_sampled.drop_duplicates(subset=["user_id", "track_id", "count"], inplace=True)

    # If final set exceeds x%, reduce from non-coverage rows
    final_count = len(df_sampled)
    if final_count > desired_rows:
        # Mark coverage rows
        df_sampled["is_coverage"] = df_sampled.apply(
            lambda row: (row["user_id"], row["track_id"], row["count"]) in coverage_mandatory_keys,
            axis=1)

        # Keep coverage rows aside
        df_coverage = df_sampled[df_sampled["is_coverage"] == True]
        df_non_coverage = df_sampled[df_sampled["is_coverage"] == False]

        # Calculate how many non-coverage rows we can keep
        slots_for_non_coverage = desired_rows - len(df_coverage)
        if slots_for_non_coverage < 0:
            # Coverage alone exceeds x% -> Keep coverage as is
            df_sampled = df_coverage.copy()
        else:
            # Randomly keep slots_for_non_coverage from the non-coverage
            df_non_cov_sampled = df_non_coverage.sample(
                n=slots_for_non_coverage,
                random_state=random_seed
            )
            df_sampled = pd.concat([df_coverage, df_non_cov_sampled], ignore_index=True)

        # Drop our helper column
        if "is_coverage" in df_sampled.columns:
            df_sampled.drop(columns=["is_coverage"], inplace=True)

    # Final stats
    final_fraction = len(df_sampled) / total_rows
    print(f"Original rows: {total_rows}")
    print(f"Desired 1% of rows: {desired_rows}")
    print(f"Final rows: {len(df_sampled)} (~{final_fraction:.4%} of original)")

    # Quick checks:
    # Every track >= 5?
    track_counts_after = df_sampled.groupby("track_id").size()
    min_track_count = track_counts_after.min() if len(track_counts_after) > 0 else 0
    print(f"Min track count after coverage: {min_track_count}")

    # Every user >= 5?
    user_counts_after = df_sampled.groupby("user_id").size()
    min_user_count = user_counts_after.min() if len(user_counts_after) > 0 else 0
    print(f"Min user count after coverage: {min_user_count}")

    return df_sampled


def train_lambdarank(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feature_cols: list,
    label_col: str,
    user_col: str,
    num_boost_round: int = 1000
):
    """
    Trains a LambdaRank model using LightGBM with ranking objective.

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
        "learning_rate": 0.001,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
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
    df_counts = sample_one_percent_with_min5_coverage(df_counts, global_fraction=0.05, coverage_count=5, random_seed=100)
    print(f"Dataset now has {len(df_counts)} rows after ensuring minimum track inclusion.")

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
    print("\nTraining the LambdaRank model...")
    model = train_lambdarank(df_train, df_val, feature_cols, label_col, user_col)
    print("LambdaRank model training complete.")

    print("\nSaving the trained model...")
    model_path = "lambdarank_model.pth"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}.")


if __name__ == "__main__":
    main()

