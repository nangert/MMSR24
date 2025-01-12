import contextlib
import os
import pickle
import bz2
import csv
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from typing import List, Dict, Set


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

        # Adjust the target dimensions as needed for each modality.
        # For example, reduce ResNet from 4096 → 50, Word2Vec from 300 → 50, MFCC from 104 → 50.
        pca_target_dims = {
            "word2vec_embeddings": 50,
            "resnet_embeddings": 50,
            "mfcc_stats_embeddings": 50
        }
        self._apply_pca_per_modality(pca_target_dims)

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

    # NEW: PCA step that does not remove or overwrite existing logic
    def _apply_pca_per_modality(self, pca_target_dims: Dict[str, int]):
        """
        For each modality, optionally reduce feature dimensions via PCA.
        The number of components is defined in pca_target_dims.
        """
        for embedding_dict_name, target_dim in pca_target_dims.items():
            embedding_dict = getattr(self, embedding_dict_name, {})
            if not embedding_dict:
                continue

            ids = list(embedding_dict.keys())
            vectors = [embedding_dict[song_id] for song_id in ids]
            matrix = np.stack(vectors)

            original_dim = matrix.shape[1]
            if original_dim <= target_dim:
                # No reduction if the original dimension is already small or equal
                print(f"[PCA] Skipping {embedding_dict_name}, original dim = {original_dim} <= target_dim = {target_dim}.")
                continue

            print(f"[PCA] Applying PCA on {embedding_dict_name}: from {original_dim} → {target_dim}")
            pca = PCA(n_components=target_dim, random_state=42)
            matrix_pca = pca.fit_transform(matrix)

            # Store back
            for i, song_id in enumerate(ids):
                embedding_dict[song_id] = matrix_pca[i]

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

    # Debug: Initialize counters
    embedding_dims = {"word2vec": 0, "resnet": 0, "mfcc_stats": 0}

    for song in shuffled_songs:
        if song["id"] in exclude_ids:
            continue

        if (song["id"] not in dataset.word2vec_embeddings or
            song["id"] not in dataset.resnet_embeddings or
            song["id"] not in dataset.mfcc_stats_embeddings):
            continue

        word2vec_vec = dataset.word2vec_embeddings[song["id"]]
        resnet_vec = dataset.resnet_embeddings[song["id"]]
        mfcc_vec = dataset.mfcc_stats_embeddings[song["id"]]

        # Debug
        embedding_dims["word2vec"] = len(word2vec_vec)
        embedding_dims["resnet"] = len(resnet_vec)
        embedding_dims["mfcc_stats"] = len(mfcc_vec)

        combined_features = np.concatenate([word2vec_vec, resnet_vec, mfcc_vec])
        genre_set = song["genres"]
        rows.append([song["id"], combined_features.tolist(), genre_set])

    print("DEBUG: Embedding dimensions in train_svm:")
    for key, dim in embedding_dims.items():
        print(f"{key}: {dim}")

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


def train_svm(df: pd.DataFrame, output_file: str = "svm_training_report.txt"):
    """
    Train an SVM model on the provided DataFrame and save the output to a file.
    """
    X = np.vstack(df["features"])
    y = df["genres"].apply(lambda genres: hash(frozenset(genres))).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    svm_model = svm.SVC(kernel='linear', probability=True)

    with open(output_file, "w") as f, contextlib.redirect_stdout(f):
        print("Training SVM...")
        svm_model.fit(X_train, y_train)

        y_pred = svm_model.predict(X_test)
        print("SVM Training Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

    print(f"Training report saved to {output_file}.")
    return svm_model


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

    # Exclude IDs from an eval subset, etc.
    eval_subset_path = "dataset/id_information_mmsr.tsv"
    exclude_ids = set(pd.read_csv(eval_subset_path, sep="\t")["id"].astype(str).tolist())

    print("Building training DataFrame...")
    df = build_training_dataframe(dataset, exclude_ids)

    print("Performing stratified subsampling...")
    df = stratified_subsample_by_genres(df, fraction=1, random_seed=100)

    print("Training SVM...")
    svm_model = train_svm(df)

    print("Saving the trained SVM model...")
    model_path = "svm_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(svm_model, f)
    print(f"Model saved to {model_path}.")


if __name__ == "__main__":
    main()
