import contextlib
import os
import pickle
import bz2
import csv
import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from typing import List, Dict, Set


class EnhancedDataset:
    """
    Loads and processes all necessary data for Late Fusion SVM training and retrieval.
    """

    def __init__(self, info_file_path: str, genres_file_path: str, metadata_file_path: str,
                 word2vec_path: str, resnet_path: str, mfcc_stats_path: str):
        self.songs = []
        self.genres = {}
        self.metadata = {}
        self.word2vec_embeddings = {}
        self.resnet_embeddings = {}
        self.mfcc_stats_embeddings = {}

        # Load all necessary data
        self._load_songs(info_file_path)
        self._load_genres(genres_file_path)
        self._load_metadata(metadata_file_path)
        self.word2vec_embeddings = self._load_compressed_embeddings(word2vec_path)
        self.resnet_embeddings = self._load_compressed_embeddings(resnet_path)
        self.mfcc_stats_embeddings = self._load_compressed_embeddings(mfcc_stats_path)

        self._normalize_embeddings_per_modality()

    def _load_songs(self, file_path: str):
        """Load song information from a TSV file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file, delimiter='\t')
            for row in reader:
                self.songs.append({
                    "id": row["id"],
                    "artist": row.get("artist", ""),
                    "song_title": row.get("song", ""),
                    "album_name": row.get("album_name", "")
                })

    def _load_genres(self, file_path: str):
        """Load genres from a TSV file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file, delimiter='\t')
            for row in reader:
                # Store genres as a set
                self.genres[row["id"]] = set(eval(row["genre"]))

    def _load_metadata(self, file_path: str):
        """Load metadata such as popularity and Spotify IDs."""
        df = pd.read_csv(file_path, delimiter='\t')
        for _, row in df.iterrows():
            self.metadata[row["id"]] = {
                "popularity": row.get("popularity", 0),
                "spotify_id": row.get("spotify_id", "")
            }

    @staticmethod
    def _load_compressed_embeddings(file_path: str) -> dict:
        """
        Load embeddings from a compressed TSV file with bz2 compression.
        Columns: [id, feat1, feat2, ...].
        """
        embeddings = {}
        with bz2.open(file_path, mode='rt', encoding='utf-8') as file:
            reader = csv.DictReader(file, delimiter='\t')
            for row in reader:
                embeddings[row["id"]] = np.array(
                    [float(row[col]) for col in row if col != "id"],
                    dtype=np.float32
                )
        return embeddings

    def _normalize_embeddings_per_modality(self):
        """
        For each modality's embedding dictionary, gather all vectors into a matrix,
        apply zero-mean, unit-variance normalization, and store them back.
        """
        for embedding_dict_name in ["word2vec_embeddings", "resnet_embeddings", "mfcc_stats_embeddings"]:
            embedding_dict = getattr(self, embedding_dict_name)
            if not embedding_dict:
                continue

            # Gather all vectors in a list
            ids = list(embedding_dict.keys())
            vectors = [embedding_dict[song_id] for song_id in ids]

            # Convert to matrix
            matrix = np.stack(vectors)
            scaler = StandardScaler()
            matrix_norm = scaler.fit_transform(matrix)

            # Store the normalized vectors back in the dictionary
            for i, song_id in enumerate(ids):
                embedding_dict[song_id] = matrix_norm[i]

    def get_all_songs(self):
        """
        Return all songs with additional metadata and genre information.
        """
        for song in self.songs:
            song_id = song["id"]
            song["genres"] = self.genres.get(song_id, set())
            song["popularity"] = self.metadata.get(song_id, {}).get("popularity", 0)
            song["spotify_id"] = self.metadata.get(song_id, {}).get("spotify_id", "")
        return self.songs


def build_late_fusion_training_df(dataset: EnhancedDataset, exclude_ids: Set[str]) -> pd.DataFrame:
    """
    Build a training DataFrame with unimodal embeddings for Late Fusion.
    Each row: [song_id, word2vec_vec, resnet_vec, mfcc_stats_vec, genre_set]
    """
    rows = []
    all_songs = dataset.get_all_songs()

    # Shuffle for potential randomization, just like early fusion
    shuffled_songs = np.random.choice(all_songs, size=len(all_songs), replace=False)

    # Debug: track dimensions of each modality
    embedding_dims = {"word2vec": 0, "resnet": 0, "mfcc_stats": 0}

    for song in shuffled_songs:
        sid = song["id"]
        if sid in exclude_ids:
            continue

        # Skip if embeddings are missing for any modality
        if (sid not in dataset.word2vec_embeddings or
            sid not in dataset.resnet_embeddings or
            sid not in dataset.mfcc_stats_embeddings):
            continue

        word2vec_vec = dataset.word2vec_embeddings[sid]
        resnet_vec = dataset.resnet_embeddings[sid]
        mfcc_vec = dataset.mfcc_stats_embeddings[sid]

        # For debug, store the dimension once
        embedding_dims["word2vec"] = len(word2vec_vec)
        embedding_dims["resnet"] = len(resnet_vec)
        embedding_dims["mfcc_stats"] = len(mfcc_vec)

        genres = song["genres"]
        rows.append([sid, word2vec_vec, resnet_vec, mfcc_vec, genres])

    print("DEBUG: Embedding dimensions in build_late_fusion_training_df:")
    for k, v in embedding_dims.items():
        print(f"  {k}: {v}")

    return pd.DataFrame(rows, columns=[
        "song_id", "word2vec", "resnet", "mfcc_stats", "genres"
    ])


def stratified_subsample_by_genres(df: pd.DataFrame, fraction: float = 0.1, random_seed: int = 100) -> pd.DataFrame:
    """
    Perform stratified subsampling based on genre sets, just like in early fusion.
    """
    df["genre_hash"] = df["genres"].apply(lambda x: hash(frozenset(x)))
    sampled_df = (
        df.groupby("genre_hash", group_keys=False)
        .apply(lambda group: group.sample(frac=fraction, random_state=random_seed))
    )
    sampled_df = sampled_df.reset_index(drop=True)
    sampled_df.drop(columns=["genre_hash"], inplace=True)

    print(f"Original DataFrame size: {len(df)}")
    print(f"Sampled DataFrame size: {len(sampled_df)} "
          f"({(len(sampled_df) / len(df)) * 100:.2f}% of original)")
    return sampled_df


def train_late_fusion_calibrated(df: pd.DataFrame, output_file: str = "late_fusion_report.txt") -> dict:
    """
    Train a calibrated late-fusion pipeline:
      1. Train calibrated unimodal SVMs (one per modality).
      2. Generate unimodal probability outputs.
      3. Train a final SVM on those probabilities (late fusion).

    Returns a dictionary containing {unimodal_svms, fusion_svm}.
    """
    # Convert genre sets into numeric labels (similar to early fusion)
    def hash_genres(genre_set):
        return hash(frozenset(genre_set))

    # Use the hashed genre sets as y
    df["label"] = df["genres"].apply(hash_genres)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=100)

    # Train Unimodal Classifiers + Calibrate
    unimodal_svms = {}
    modalities = ["word2vec", "resnet", "mfcc_stats"]

    # We’ll store the debug prints in the output file
    with open(output_file, "w") as f, contextlib.redirect_stdout(f):
        print("Training Late Fusion with Calibrated Unimodal SVMs...\n")

        for modality in modalities:
            print(f"--- Training SVM for modality '{modality}' ---")
            X_train_mod = np.vstack(train_df[modality])
            y_train = train_df["label"].values

            # Train raw SVM
            svm_model_raw = svm.SVC(kernel="linear", probability=False, random_state=100)
            svm_model_raw.fit(X_train_mod, y_train)

            # Calibrate using Platt's scaling
            calibrated_model = CalibratedClassifierCV(
                estimator=svm_model_raw,
                cv="prefit",
                method="sigmoid"
            )
            calibrated_model.fit(X_train_mod, y_train)

            unimodal_svms[modality] = calibrated_model
            print("Done.\n")

        # Generate Late Fusion Features
        def generate_late_fusion_features(df_subset: pd.DataFrame):
            # For each unimodal classifier, produce predict_proba outputs
            # Then horizontally stack them to create the fusion feature
            feature_list = []
            for m in modalities:
                X_mod = np.vstack(df_subset[m])
                probs = unimodal_svms[m].predict_proba(X_mod)
                feature_list.append(probs)
            return np.hstack(feature_list)

        print("--- Generating training features for fusion ---")
        X_train_fusion = generate_late_fusion_features(train_df)
        y_train_fusion = train_df["label"].values

        print("--- Generating test features for fusion ---")
        X_test_fusion = generate_late_fusion_features(test_df)
        y_test_fusion = test_df["label"].values

        # Train Final Fusion Classifier
        print("--- Training final SVM on fused probabilities ---")
        fusion_svm = svm.SVC(kernel="linear", probability=False, random_state=100)
        fusion_svm.fit(X_train_fusion, y_train_fusion)
        print("Done.\n")

        # Evaluate the final fusion classifier
        y_pred = fusion_svm.predict(X_test_fusion)
        acc = accuracy_score(y_test_fusion, y_pred)
        cls_rep = classification_report(y_test_fusion, y_pred, zero_division=0)

        print("=== Late Fusion Evaluation ===")
        print(f"Test Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(cls_rep)

    print(f"Late Fusion training report saved to {output_file}.")

    return {
        "unimodal_svms": unimodal_svms,
        "fusion_svm": fusion_svm
    }


def main():
    # Same file paths and structure as early fusion
    info_file_path = "dataset/train/id_information.csv"
    genres_file_path = "dataset/train/id_genres_mmsr.tsv"
    metadata_file_path = "dataset/train/id_metadata.csv"
    word2vec_path = "dataset/train/id_lyrics_word2vec.tsv.bz2"
    resnet_path = "dataset/train/id_resnet.tsv.bz2"
    mfcc_stats_path = "dataset/train/id_mfcc_stats.tsv.bz2"

    # Load dataset
    print("Loading dataset...")
    dataset = EnhancedDataset(
        info_file_path,
        genres_file_path,
        metadata_file_path,
        word2vec_path,
        resnet_path,
        mfcc_stats_path
    )

    # Exclude IDs from an evaluation subset or other criteria
    eval_subset_path = "dataset/id_information_mmsr.tsv"
    exclude_ids = set(pd.read_csv(eval_subset_path, sep="\t")["id"].astype(str).tolist())

    # Build the DataFrame for late fusion
    print("Building Late Fusion DataFrame...")
    df = build_late_fusion_training_df(dataset, exclude_ids)

    # Perform stratified subsampling to ensure balanced classes
    print("Performing stratified subsampling...")
    df = stratified_subsample_by_genres(df, fraction=0.1, random_seed=100)

    # Train the late fusion pipeline with calibrated unimodal classifiers
    print("Training Calibrated Late Fusion...")
    model_bundle = train_late_fusion_calibrated(df, output_file="late_fusion_training_report.txt")

    # Save the entire model bundle
    model_path = "late_fusion_model.pkl"
    print(f"Saving model bundle to {model_path}...")
    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)
    print(f"Late Fusion model saved to {model_path}.")


if __name__ == "__main__":
    main()
