# train_lambdaMERT.py

"""
Dependencies:
  pip install allRank lightgbm
"""

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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

###############################################################################
# Step 1: Load user-track interactions from a .bz2 TSV
###############################################################################
def load_user_track_counts(counts_path: str, large_ids: set, exclude_ids: set) -> pd.DataFrame:
    """
    Reads user-track interaction data from a BZ2-compressed TSV and filters out unwanted data.
    Columns: user_id, track_id, count
    Returns a DataFrame with columns [user_id, track_id, count].
    """
    rows = []
    with bz2.open(counts_path, mode="rt", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            track_id = str(row["track_id"])
            if track_id not in large_ids or track_id in exclude_ids:
                continue  # Exclude unwanted tracks

            user_id = row["user_id"]
            c = float(row["count"])
            c = max(min(c, 100.0), 1e-6)

            rows.append((user_id, track_id, c))
    df = pd.DataFrame(rows, columns=["user_id", "track_id", "count"])
    return df


###############################################################################
# Step 2: Load track-level features
###############################################################################
def load_track_features(features_path: str, large_ids: set, exclude_ids: set) -> dict:
    """
    Reads track-level features from a BZ2-compressed TSV, filtering out unwanted tracks.
    Expects columns: [id, feat_1, feat_2, ..., feat_n]
    Returns {track_id -> np.array([...features...])}.
    """
    feature_dict = {}
    with bz2.open(features_path, mode="rt", encoding="utf-8") as f:
        df = pd.read_csv(f, sep="\t")
        cols = list(df.columns)
        feat_cols = cols[1:]  # everything except the 'id'

        for _, row in df.iterrows():
            track_id = str(row["id"])
            if track_id not in large_ids or track_id in exclude_ids:
                continue  # Exclude unwanted tracks
            feats = row[feat_cols].values.astype(float)
            feature_dict[track_id] = feats
    return feature_dict


###############################################################################
# Step 3: Load track IDs from the large CSV
###############################################################################
def load_large_track_ids(meta_path: str) -> set:
    """
    Reads track metadata from 'id_information.csv' (109k+ IDs).
    Returns a set of all valid track IDs in that CSV.
    """
    df = pd.read_csv(meta_path, sep='\t')
    return set(str(x) for x in df["id"].unique())


###############################################################################
# Step 4: Load evaluation track IDs and exclude them from training
###############################################################################
def load_eval_track_ids(mmsr_path: str) -> set:
    """
    Reads the subset track IDs from 'id_information_mmsr.tsv'.
    Returns a set of track IDs that must be excluded from training.
    """
    df = pd.read_csv(mmsr_path, sep="\t")
    return set(str(x) for x in df["id"].unique())


###############################################################################
# Step 5: Merge user, track, and features, producing a DataFrame
###############################################################################
def build_ltr_dataframe(
        df_counts: pd.DataFrame,
        feat_dict: dict,
        chunk_size: int = 100000
) -> pd.DataFrame:
    """
    Merges user-track rows with track features to form a DataFrame:
      user_id, track_id, [feature_1..feature_n], relevance_label
    Processes the data in chunks to reduce memory usage and track progress.
    """
    rows = []
    total_rows = len(df_counts)
    print(f"Processing {total_rows} rows in chunks of {chunk_size}...")

    for i in tqdm(range(0, total_rows, chunk_size), desc="Building LTR DataFrame"):
        chunk = df_counts.iloc[i: i + chunk_size]
        for _, row in chunk.iterrows():
            user_id = row["user_id"]
            track_id = str(row["track_id"])
            label = float(row["count"])
            feats = feat_dict[track_id]
            merged_row = [user_id, track_id] + feats.tolist() + [label]
            rows.append(merged_row)

    # Build column names
    if not rows:
        raise ValueError("No valid training rows after merging. Check data paths.")
    feat_count = len(rows[0]) - 3  # user_id + track_id + label => remainder = features
    cols = ["user_id", "track_id"] + [f"feat_{i + 1}" for i in range(feat_count)] + ["label"]
    df_merged = pd.DataFrame(rows, columns=cols)

    return df_merged


###############################################################################
# Step 6: Convert to .libsvm format for allRank
###############################################################################
def df_to_libsvm(df: pd.DataFrame, out_file: str):
    """
    allRank expects .libsvm with group/query. Each row:
      <label> qid:<user_id> <feature_index>:<feature_value> ...
    We'll map user_id => integer for qid.
    """
    # Sort by user_id for consistency
    df_sorted = df.sort_values("user_id").reset_index(drop=True)

    # Map user IDs to an integer qid
    unique_users = df_sorted["user_id"].unique()
    user2qid = {uid: i for i, uid in enumerate(unique_users)}

    # Build lines
    with open(out_file, "w", encoding="utf-8") as f:
        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        label_col = "label"
        user_col = "user_id"
        for _, row in df_sorted.iterrows():
            label = row[label_col]
            qid = user2qid[row[user_col]]
            feats = []
            for i, c in enumerate(feat_cols, start=1):
                feats.append(f"{i}:{row[c]}")
            line = f"{label} qid:{qid} " + " ".join(feats) + "\n"
            f.write(line)

    print(f"Saved .libsvm file to {out_file}")


###############################################################################
# Step 7: Build a minimal allRank config file
###############################################################################
def write_allrank_config(
        combined_data_dir: str,
        output_dir: str,
        config_path: str,
        slate_length: int
):
    """
    Writes a JSON config for allRank that matches the official config.py structure
    and expects train/val data to be found in one directory.
    """
    config = {
        "data": {
            "path": combined_data_dir,
            "num_workers": 4,
            "batch_size": 16,
            "slate_length": slate_length,
            "validation_ds_role": "test"  # instructs allRank that part of data is for testing/validation
        },
        "model": {
            "fc_model": {
                "sizes": [128, 64],
                "input_norm": True,
                "activation": "ReLU",
                "dropout": 0.2
            },
            "transformer": {
                "N": 2,
                "d_ff": 256,
                "h": 4,
                "positional_encoding": {
                    "strategy": "fixed",
                    "max_indices": 512
                },
                "dropout": 0.1
            },
            "post_model": {
                "output_activation": "Sigmoid",
                "d_output": 1
            }
        },
        "optimizer": {
            "name": "Adam",
            "args": {
                "lr": 0.1,
                "weight_decay": 0.0001
            }
        },
        "training": {
            "epochs": 20,
            "gradient_clipping_norm": 10.0,
            "early_stopping_patience": 3
        },
        "loss": {
            "name": "neuralNDCG",
            "args": {}
        },
        "metrics": ["ndcg_5", "ndcg_10"],
        "lr_scheduler": {
            "name": "StepLR",
            "args": {
                "step_size": 1,
                "gamma": 0.8
            }
        },
        "val_metric": "ndcg_10",  # Use correct key for validation metric
        "expected_metrics": {
            "train": {
                "ndcg_10": 0.9  # Match this key to the result subkeys
            },
            "val": {
                "ndcg_10": 0.9  # Match this key to the result subkeys
            }
        },
        "detect_anomaly": True
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    print(f"allRank config written to {config_path}")


###############################################################################
# Step 8: Subprocess call to allRank
###############################################################################
def run_allrank(python_package_path: str, config_path: str, run_id: str, job_dir: str, ):
    """
    Calls allRank using a subprocess. This will train the model and store results in the job_dir.

    Args:
        allrank_package_path (str): The root path to the allRank package.
        config_path (str): Path to the allRank config.json file.
        run_id (str): A unique identifier for the training run.
        job_dir (str): Directory where the results will be saved.
    """
    # Build the path to the main.py script within the allRank package
    main_script_path = os.path.join(python_package_path, "allrank", "main.py")

    # Use the Python interpreter currently running this script
    python_exec = sys.executable

    # Build the subprocess command
    cmd = [
        python_exec,  # Automatically use the correct Python executable
        main_script_path,  # Path to the allRank main.py script
        "--config-file-name", config_path,  # Path to the configuration file
        "--run-id", run_id,  # Unique identifier for this run
        "--job-dir", job_dir  # Directory for saving results
    ]

    # Print the constructed command for debugging
    print("Running allRank command:", " ".join(cmd))

    try:
        # Run the command and check for errors
        subprocess.run(cmd, check=True)
        print("allRank training complete.")
    except subprocess.CalledProcessError as e:
        print(f"Error while running allRank: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


###############################################################################
# Step 9: Stratify the interaction data
###############################################################################
def stratified_subsample(df_counts: pd.DataFrame, sample_fraction: float = 0.1) -> pd.DataFrame:
    """
    Performs stratified subsampling on the user-track interaction data.
    Each user's interactions are subsampled at the given fraction.

    Args:
        df_counts (pd.DataFrame): The full user-track interaction dataset.
        sample_fraction (float): The fraction of interactions to keep per user.

    Returns:
        pd.DataFrame: A stratified subsample of the dataset.
    """
    print(f"Subsampling {sample_fraction * 100}% of interactions per user...")
    subsampled = (
        df_counts
        .groupby("user_id", group_keys=False)
        .apply(lambda group: group.sample(frac=sample_fraction, random_state=100))
    )
    print(f"Subsampled dataset size: {len(subsampled)} rows (original: {len(df_counts)})")
    return subsampled


def find_longest_slate(df_merged: pd.DataFrame) -> int:
    """
    Determines the longest slate (maximum number of tracks per user).

    Args:
        df_merged (pd.DataFrame): DataFrame containing user-track interactions.

    Returns:
        int: Length of the longest slate.
    """
    slate_lengths = df_merged.groupby("user_id").size()
    longest_slate = slate_lengths.max()
    print(f"The longest slate (query length) is: {longest_slate}")
    return int(longest_slate)


def pad_queries(df: pd.DataFrame, slate_length: int, feat_count: int) -> pd.DataFrame:
    """
    Pads each user's query to the specified slate_length with zero labels and features.

    Args:
        df (pd.DataFrame): Input DataFrame with user-track interactions and features.
        slate_length (int): Desired slate length.
        feat_count (int): Number of feature columns.

    Returns:
        pd.DataFrame: Padded DataFrame.
    """
    padded_rows = []
    for user_id, group in df.groupby("user_id"):
        rows = group.to_dict("records")
        # If fewer rows than slate_length, pad with dummy rows
        while len(rows) < slate_length:
            dummy_row = {
                "user_id": user_id,
                "track_id": "PAD",  # Placeholder track ID
                **{f"feat_{i + 1}": 0.0 for i in range(feat_count)},  # Zero for all features
                "label": 0.0  # Zero relevance for padded positions
            }
            rows.append(dummy_row)
        padded_rows.extend(rows)
    return pd.DataFrame(padded_rows)


###############################################################################
# Main Script
###############################################################################
def main():
    python_package_path = "/home/jonasg/miniconda3/envs/onion/lib/python3.8/site-packages"
    train_counts_path = "dataset/train/userid_trackid_count.tsv.bz2"
    train_features_path = "dataset/train/id_mfcc_bow.tsv.bz2"
    large_meta_path = "dataset/train/id_information.csv"  # large CSV with ~109k IDs
    eval_subset_path = "../dataset/id_information_mmsr.tsv"  # file with 5,148 eval IDs
    sample_fraction = 0.001  # Use x (10)% of interactions

    # Load large IDs and eval subset IDs
    print("Loading large track IDs (from id_information.csv)...")
    large_ids = load_large_track_ids(large_meta_path)
    print(f"  => {len(large_ids)} total track IDs in the large CSV.")

    print("Loading eval subset track IDs (from id_information_mmsr.tsv)...")
    eval_ids = load_eval_track_ids(eval_subset_path)
    print(f"  => {len(eval_ids)} track IDs are in the eval subset (exclude them).")

    # Load filtered user-track counts and features
    print("Loading user-track counts...")
    df_counts = load_user_track_counts(train_counts_path, large_ids, eval_ids)
    print(f"  => {len(df_counts)} interaction rows loaded after filtering.")

    print("Loading track features...")
    feat_dict = load_track_features(train_features_path, large_ids, eval_ids)
    print(f"  => {len(feat_dict)} track feature vectors loaded after filtering.")

    # Step B: Subsample
    df_counts = stratified_subsample(df_counts, sample_fraction)

    # Merge
    print("Building LTR DataFrame (excluding eval IDs)...")
    df_merged = build_ltr_dataframe(df_counts, feat_dict)
    print(f"  => {len(df_merged)} rows remain after merging and exclusion.")

    longest_slate = find_longest_slate(df_merged)

    print("Padding queries to slate length...")
    feat_count = len([col for col in df_merged.columns if col.startswith("feat_")])
    # df_merged = pad_queries(df_merged, slate_length=longest_slate, feat_count=feat_count)  # Use detected slate length
    print(f"Padded DataFrame to {longest_slate} rows per query.")

    # Train/val split
    print("Splitting into train/val sets by user...")
    unique_users = df_merged["user_id"].unique()
    np.random.seed(100)
    np.random.shuffle(unique_users)
    val_ratio = 0.2
    val_size = int(len(unique_users) * val_ratio)
    val_users = set(unique_users[:val_size])
    train_users = set(unique_users[val_size:])

    df_train = df_merged[df_merged["user_id"].isin(train_users)].copy()
    df_val = df_merged[df_merged["user_id"].isin(val_users)].copy()
    print(f"  => Train set size: {len(df_train)} rows, Val set size: {len(df_val)} rows.")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Both train.libsvm and val.libsvm go into the same directory
        train_libsvm = os.path.join(tmpdir, "train.libsvm")
        val_libsvm = os.path.join(tmpdir, "val.libsvm")

        print("Converting train set to .libsvm...")
        df_to_libsvm(df_train, train_libsvm)
        print("Converting val set to .libsvm...")
        df_to_libsvm(df_val, val_libsvm)

        # In this example, we rely on allRank's internal logic to handle
        # both files in the same directory. Typically you'd name them
        # "train.txt" and "test.txt" or "val.txt", then point "path"
        # to this directory.
        os.rename(train_libsvm, os.path.join(tmpdir, "train.txt"))
        os.rename(val_libsvm, os.path.join(tmpdir, "test.txt"))

        job_dir = os.path.join(tmpdir, "allrank_out")
        run_id = "lambdamart_test_run"
        config_path = os.path.join(tmpdir, "config.json")

        # We'll pass the entire directory as 'data.path'
        # allRank expects "train.txt" and "test.txt" in that directory
        write_allrank_config(
            combined_data_dir=tmpdir,
            output_dir=job_dir,
            config_path=config_path,
            slate_length=longest_slate
        )

        run_allrank(python_package_path, config_path, run_id, job_dir)

        print()
        print("Training is done. The model and logs are in:", job_dir)

        # Step G: Copy results to current directory
        current_dir = os.getcwd()
        target_dir = os.path.join(current_dir, "allrank_results")
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        print(f"Copying results from {job_dir} to {target_dir}...")
        shutil.copytree(job_dir, target_dir, dirs_exist_ok=True)

        print(f"Results copied to {target_dir}. You can now use them as needed.")


if __name__ == "__main__":
    main()
