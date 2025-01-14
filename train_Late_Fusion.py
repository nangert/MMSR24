import contextlib
import os
import pickle
import bz2
import csv
import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier  # Added import
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.multioutput import ClassifierChain
from typing import List, Dict, Set
from tqdm import tqdm


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

    shuffled_songs = np.random.choice(all_songs, size=len(all_songs), replace=False)

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


def train_late_fusion_multi_label(df: pd.DataFrame, output_file: str = "late_fusion_report.txt") -> dict:
    """
    Train a multi-label late-fusion pipeline:
      1. Train a ClassifierChain for each unimodal embedding.
      2. Generate unimodal probability outputs.
      3. Train a final ClassifierChain on those probabilities for late fusion.

    Returns:
        A dictionary containing trained models and metadata.
    """

    # Remove songs with 0 or 1 genre
    df = df[df["genres"].apply(lambda x: len(x) > 1)]

    # MultiLabelBinarizer to convert genres into a binary matrix
    all_genres = sorted({g for row in df["genres"] for g in row})
    mlb = MultiLabelBinarizer(classes=all_genres)
    Y = mlb.fit_transform(df["genres"])  # shape: (num_samples, num_labels)

    # Prepare feature arrays
    word2vec_arr = np.vstack(df["word2vec"])
    resnet_arr = np.vstack(df["resnet"])
    mfcc_arr = np.vstack(df["mfcc_stats"])

    # Split into train/test sets
    X_train_w2v, X_test_w2v, Y_train, Y_test = train_test_split(word2vec_arr, Y, test_size=0.2, random_state=100)
    X_train_res, X_test_res = train_test_split(resnet_arr, test_size=0.2, random_state=100)
    X_train_mfcc, X_test_mfcc = train_test_split(mfcc_arr, test_size=0.2, random_state=100)

    valid_labels = np.any(Y_train, axis=0) & np.any(~Y_train, axis=0)
    Y_train = Y_train[:, valid_labels]
    Y_test = Y_test[:, valid_labels]
    valid_genres = [all_genres[i] for i, v in enumerate(valid_labels) if v]

    if Y.shape[1] == 0:
        raise ValueError("No valid labels remain after filtering out single-class labels.")

    # Train Unimodal ClassifierChains
    def train_classifier_chain(X_train, y_train, name):
        print(f"\n=== Training ClassifierChain for '{name}' ===")
        chain = ClassifierChain(
            base_estimator=RandomForestClassifier(n_estimators=100, random_state=100)
        )
        chain.fit(X_train, y_train)
        print(f"Done training {name}.")
        return chain

    with open(output_file, "w") as f, contextlib.redirect_stdout(f):
        print("Training Multi-Label Late Fusion...")

        w2v_chain = train_classifier_chain(X_train_w2v, Y_train, "Word2Vec")
        res_chain = train_classifier_chain(X_train_res, Y_train, "ResNet")
        mfcc_chain = train_classifier_chain(X_train_mfcc, Y_train, "MFCC")

        # Generate stacked features from unimodal probability outputs
        def get_stacked_features(X_w2v, X_res, X_mfcc):
            p_w2v = w2v_chain.predict_proba(X_w2v)  # shape: (samples, #labels)
            p_res = res_chain.predict_proba(X_res)
            p_mfcc = mfcc_chain.predict_proba(X_mfcc)
            # Stack horizontally => shape: (samples, #labels * 3)
            return np.hstack([p_w2v, p_res, p_mfcc])

        print("\n=== Generating training & test features for fusion ===")
        X_train_fusion = get_stacked_features(X_train_w2v, X_train_res, X_train_mfcc)
        X_test_fusion = get_stacked_features(X_test_w2v, X_test_res, X_test_mfcc)

        # Train final ClassifierChain for fusion
        print("\n=== Training fusion ClassifierChain ===")
        fusion_chain = ClassifierChain(
            base_estimator=RandomForestClassifier(n_estimators=100, random_state=100)
        )
        fusion_chain.fit(X_train_fusion, Y_train)

        # Evaluate
        Y_pred = fusion_chain.predict(X_test_fusion)
        cls_rep = classification_report(Y_test, Y_pred, target_names=valid_genres, zero_division=0)
        acc = accuracy_score(Y_test, Y_pred)

        print("\n=== Late Fusion Evaluation (Multi-Label) ===")
        print("Classification Report:")
        print(cls_rep)
        print(f"Multi-label Subset Accuracy: {acc:.4f}")

    print(f"Multi-label Late Fusion training report saved to {output_file}.")

    return {
        "mlb": mlb,
        "valid_genres": valid_genres,
        "word2vec_chain": w2v_chain,
        "resnet_chain": res_chain,
        "mfcc_chain": mfcc_chain,
        "fusion_chain": fusion_chain
    }


def main():
    # Same file paths and structure as before
    info_file_path = "dataset/train/id_information.csv"
    genres_file_path = "dataset/train/id_genres_mmsr.tsv"
    metadata_file_path = "dataset/train/id_metadata.csv"
    word2vec_path = "dataset/train/id_lyrics_word2vec.tsv.bz2"
    resnet_path = "dataset/train/id_resnet.tsv.bz2"
    mfcc_stats_path = "dataset/train/id_mfcc_stats.tsv.bz2"

    print("Loading dataset...")
    dataset = EnhancedDataset(
        info_file_path,
        genres_file_path,
        metadata_file_path,
        word2vec_path,
        resnet_path,
        mfcc_stats_path
    )

    eval_subset_path = "dataset/id_information_mmsr.tsv"
    exclude_ids = set(pd.read_csv(eval_subset_path, sep="\t")["id"].astype(str).tolist())

    print("Building Late Fusion DataFrame...")
    df = build_late_fusion_training_df(dataset, exclude_ids)

    print("Performing stratified subsampling...")
    df = stratified_subsample_by_genres(df, fraction=0.01, random_seed=100)

    print("Training multi-label Late Fusion...")
    model_bundle = train_late_fusion_multi_label(df, output_file="late_fusion_training_report.txt")

    model_path = "late_fusion_model.pkl"
    print(f"Saving model bundle to {model_path}...")
    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)
    print(f"Late Fusion model saved to {model_path}.")


if __name__ == "__main__":
    main()
