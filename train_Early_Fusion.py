import contextlib
import os
import pickle
import bz2
import csv
import pickle
import numpy as np

from sklearn import svm
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Set
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import ClassifierChain
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report



class EnhancedDataset:
    """
    Loads and processes all necessary data for SVM training and retrieval.
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
        """Load song information from a CSV/TSV file."""
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
        """Load embeddings from a compressed TSV file."""
        embeddings = {}
        with bz2.open(file_path, mode='rt', encoding='utf-8') as file:
            reader = csv.DictReader(file, delimiter='\t')
            for row in reader:
                embeddings[row["id"]] = np.array([float(row[col]) for col in row if col != "id"], dtype=np.float32)
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

            ids = list(embedding_dict.keys())
            vectors = [embedding_dict[song_id] for song_id in ids]
            matrix = np.stack(vectors)

            scaler = StandardScaler()
            matrix_norm = scaler.fit_transform(matrix)

            for i, song_id in enumerate(ids):
                embedding_dict[song_id] = matrix_norm[i]

    def feature_selection_pipeline(self, embedding_dict: Dict[str, np.ndarray], method: str = "all", threshold: float = 0.95) -> Dict[str, np.ndarray]:
        """
        A unified feature selection pipeline that applies the following methods:
            1. Remove highly correlated features using Pearson's correlation matrix.
            2. Recursive Feature Elimination (RFE).
            3. SelectFromModel for feature importance-based selection.
        """

        ids = list(embedding_dict.keys())
        matrix = np.stack([embedding_dict[song_id] for song_id in ids])  # (n_samples, n_features)

        selected_matrix = matrix

        if method in ("correlation", "all"):
            correlation_matrix = pd.DataFrame(selected_matrix).corr().abs()
            upper_triangle = np.triu(correlation_matrix, k=1)
            to_remove = [i for i in range(correlation_matrix.shape[1]) if any(upper_triangle[:, i] > threshold)]
            if to_remove:
                selected_matrix = np.delete(selected_matrix, to_remove, axis=1)

        if method in ("rfe", "all"):
            rfe_model = RFE(estimator=LogisticRegression(max_iter=500, random_state=100), n_features_to_select=10, step=10)
            selected_matrix = rfe_model.fit_transform(selected_matrix, np.random.randint(0, 2, len(selected_matrix)))

        if method in ("selectfrommodel", "all"):
            sfm_model = SelectFromModel(estimator=RandomForestClassifier(random_state=100), threshold="median")
            selected_matrix = sfm_model.fit_transform(selected_matrix, np.random.randint(0, 2, len(selected_matrix)))

        reduced_embeddings = {ids[i]: selected_matrix[i] for i in range(len(ids))}
        return reduced_embeddings

    def _apply_feature_selection_per_modality(self, method: str = "all", threshold: float = 0.95):
        """
        Apply feature selection for each modality using a specified method.
        """
        for embedding_dict_name in ["word2vec_embeddings", "resnet_embeddings", "mfcc_stats_embeddings"]:
            original_embedding_dict = getattr(self, embedding_dict_name, {})
            if not original_embedding_dict:
                continue

            reduced_embeddings = self.feature_selection_pipeline(original_embedding_dict, method=method, threshold=threshold)
            setattr(self, embedding_dict_name, reduced_embeddings)

    def get_all_songs(self):
        """Return all songs with additional metadata and genre information."""
        for song in self.songs:
            song_id = song["id"]
            song["genres"] = self.genres.get(song_id, set())
            song["popularity"] = self.metadata.get(song_id, {}).get("popularity", 0)
            song["spotify_id"] = self.metadata.get(song_id, {}).get("spotify_id", "")
        return self.songs


def build_training_dataframe(dataset: EnhancedDataset, exclude_ids: set) -> pd.DataFrame:
    """
    Build a training DataFrame by combining Word2Vec, ResNet, and MFCC Stat embeddings.
    """
    rows = []
    all_songs = dataset.get_all_songs()
    shuffled_songs = np.random.choice(all_songs, size=int(len(all_songs)), replace=False)

    for song in shuffled_songs:
        if song["id"] in exclude_ids:
            continue

        if (
            song["id"] not in dataset.word2vec_embeddings or
            song["id"] not in dataset.resnet_embeddings or
            song["id"] not in dataset.mfcc_stats_embeddings
        ):
            continue

        word2vec_vec = dataset.word2vec_embeddings[song["id"]]
        resnet_vec = dataset.resnet_embeddings[song["id"]]
        mfcc_vec = dataset.mfcc_stats_embeddings[song["id"]]

        combined_features = np.concatenate([word2vec_vec, resnet_vec, mfcc_vec])
        genre_set = song["genres"]
        rows.append([song["id"], combined_features.tolist(), genre_set])

    return pd.DataFrame(rows, columns=["song_id", "features", "genres"])


def stratified_subsample_by_genres(df: pd.DataFrame, fraction: float = 0.1, random_seed: int = 100) -> pd.DataFrame:
    """
    Perform stratified subsampling based on genre sets.
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


def train_svm(df: pd.DataFrame):
    """
    Train an early fusion SVM model using a multi-label approach. Stores:
      - A dictionary of {genre_string: hash} for consistent IR-related usage.
      - A multi-label binarizer for transforming predictions back to genre sets.
      - The trained multi-label SVM chain.
    """

    from sklearn.multioutput import ClassifierChain
    from sklearn.preprocessing import MultiLabelBinarizer

    df = df[df["genres"].apply(lambda x: len(x) > 1)]

    all_genres = set()
    for genres_set in df["genres"]:
        for g in genres_set:
            all_genres.add(g)
    all_genres = sorted(all_genres)

    genre2hash = {g: hash(g) for g in all_genres}

    mlb = MultiLabelBinarizer(classes=all_genres)
    Y = mlb.fit_transform(df["genres"])

    features = np.vstack(df["features"])

    X_train, X_test, Y_train, Y_test = train_test_split(
        features, Y, test_size=0.2, random_state=100
    )

    valid_labels = np.any(Y_train, axis=0) & np.any(~Y_train, axis=0)
    Y_train = Y_train[:, valid_labels]
    Y_test = Y_test[:, valid_labels]
    valid_genres = [all_genres[i] for i, v in enumerate(valid_labels) if v]

    if Y_train.shape[1] == 0:
        raise ValueError("No valid labels found after filtering.")

    chain_classifier = ClassifierChain(
        base_estimator=svm.SVC(kernel="linear", probability=True, random_state=100)
    )
    chain_classifier.fit(X_train, Y_train)

    Y_pred = chain_classifier.predict(X_test)

    cr = classification_report(Y_test, Y_pred, target_names=valid_genres, zero_division=0)
    with open("classification_report.txt", "w") as rep_file:
        rep_file.write(cr)

    acc = accuracy_score(Y_test, Y_pred)
    print("Multi-label Subset Accuracy:", acc)

    model_bundle = {
        "svm_model": chain_classifier,
        "genre2hash": genre2hash,
        "mlb": mlb,
        "valid_genres": valid_genres,
    }

    return model_bundle


def main():
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

    # Remove labels not found in exclude_ids
    exclude_labels = set()
    for eid in exclude_ids:
        if eid in dataset.genres:
            exclude_labels.update(dataset.genres[eid])
    for sid in dataset.genres:
        dataset.genres[sid] = dataset.genres[sid].intersection(exclude_labels)

    print("Building training DataFrame...")
    df = build_training_dataframe(dataset, exclude_ids)

    print("Performing stratified subsampling...")
    df = stratified_subsample_by_genres(df, fraction=0.05, random_seed=100)

    print("Determining top 50 features using LinearSVC...")

    # Convert multi-label sets to a binary matrix
    mlb = MultiLabelBinarizer()
    Y_multi = mlb.fit_transform(df["genres"])       # (n_samples, n_genres)

    # Reduce Y to single-label for feature importance
    Y_single = Y_multi.argmax(axis=1)              # (n_samples, )

    # Combine features into a matrix
    X = np.vstack(df["features"])                  # (n_samples, n_features)

    # Fit the single-label LinearSVC for feature importance
    single_label_svc = LinearSVC(random_state=100, max_iter=1000)
    single_label_svc.fit(X, Y_single)

    # For multi-class LinearSVC, 'coef_' has shape (n_classes, n_features).
    #     We can average the absolute coefficients across all classes to get importance.
    all_coefs = np.abs(single_label_svc.coef_)     # shape: (n_classes, n_features)
    feature_importances = all_coefs.mean(axis=0)   # shape: (n_features,)

    # Pick the top 50 features
    top_50_indices = np.argsort(-feature_importances)[:50]
    top_50_indices_set = set(top_50_indices)       # for quick membership checking

    #  Create a boolean mask for easier indexing
    selected_mask = np.zeros(X.shape[1], dtype=bool)
    selected_mask[top_50_indices] = True

    print("Total features:", X.shape[1])
    print("Selected top 50 features by LinearSVC average absolute coefficients.")

    # Print how many from each modality
    word2vec_dim = len(dataset.word2vec_embeddings[next(iter(dataset.word2vec_embeddings))])
    resnet_dim = len(dataset.resnet_embeddings[next(iter(dataset.resnet_embeddings))])
    mfcc_dim = len(dataset.mfcc_stats_embeddings[next(iter(dataset.mfcc_stats_embeddings))])

    word2vec_indices = range(0, word2vec_dim)
    resnet_indices = range(word2vec_dim, word2vec_dim + resnet_dim)
    mfcc_indices = range(word2vec_dim + resnet_dim, word2vec_dim + resnet_dim + mfcc_dim)

    word2vec_selected = [i for i in top_50_indices if i in word2vec_indices]
    resnet_selected   = [i for i in top_50_indices if i in resnet_indices]
    mfcc_selected     = [i for i in top_50_indices if i in mfcc_indices]

    print(f"Features selected from word2vec: {len(word2vec_selected)}")
    print(f"Features selected from resnet:   {len(resnet_selected)}")
    print(f"Features selected from mfcc:     {len(mfcc_selected)}")

    # Apply the top-50 mask to X, then re-do final multi-label training
    X_50 = X[:, selected_mask]  # shape: (n_samples, 50)

    # Update the DataFrame's features with these final 50
    for i in range(len(df)):
        df.at[i, "features"] = X_50[i]

    print("Training final multi-label classifier on these 50 features...")

    # Prepare final X, Y
    X_final = np.vstack(df["features"])  # shape: (n_samples, 50)
    Y_final = mlb.fit_transform(df["genres"])

    # Train multi-label classifier
    chain_classifier = ClassifierChain(LinearSVC(random_state=100, max_iter=5000))
    X_train, X_test, Y_train, Y_test = train_test_split(X_final, Y_final, test_size=0.2, random_state=100)

    chain_classifier.fit(X_train, Y_train)
    Y_pred = chain_classifier.predict(X_test)

    print("Accuracy:", accuracy_score(Y_test, Y_pred))

    #  Build the final model bundle with the essential pieces
    model_bundle = {
        "svm_model": chain_classifier,     # The multi-label chain classifier
        "mlb": mlb,                        # MultiLabelBinarizer
        "selected_mask": selected_mask,    # Boolean mask for the top 50 features
    }

    # Save to pickle
    model_path = "early_fusion_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)

    print(f"Saved model bundle (with 50-feature mask) to {model_path}")




if __name__ == "__main__":
    main()
