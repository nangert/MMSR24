import contextlib
import os
import pickle
import bz2
import csv
import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from typing import List, Dict, Set


class EnhancedDataset:
    """
    Loads and processes all necessary data for SVC training and retrieval.
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
        Load embeddings from a compressed TSV file.
        """
        embeddings = {}
        with bz2.open(file_path, mode='rt', encoding='utf-8') as file:
            reader = csv.DictReader(file, delimiter='\t')
            for row in reader:
                embeddings[row["id"]] = np.array(
                    [float(row[col]) for col in row if col != "id"], dtype=np.float32
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

            ids = list(embedding_dict.keys())
            vectors = [embedding_dict[song_id] for song_id in ids]
            matrix = np.stack(vectors)

            scaler = StandardScaler()
            matrix_norm = scaler.fit_transform(matrix)

            for i, song_id in enumerate(ids):
                embedding_dict[song_id] = matrix_norm[i]

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
    shuffled_songs = np.random.choice(all_songs, size=len(all_songs), replace=False)

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
    print(f"Sampled DataFrame size: {len(sampled_df)} ({(len(sampled_df) / len(df)) * 100:.2f}% of original)")
    return sampled_df


def train_SVM(dataset: EnhancedDataset,
              eval_subset_path: str,
              fraction: float = 0.05,
              random_seed: int = 100,
              top_features: int = 50,
              model_save_path: str = "early_fusion_model.pkl") -> Dict:
    """
    Trains an Early Fusion multi-label classifier using local linear embeddings (LLE) for dimensionality reduction
    and a classifier chain with LinearSVC for multi-label classification.
    Additionally, applies L2 normalization to each concatenated feature vector before LLE.
    """

    # Determine which songs to exclude based on the evaluation subset
    exclude_ids = set(pd.read_csv(eval_subset_path, sep="\t")["id"].astype(str).tolist())

    # Limit genres to those present in the evaluation set
    exclude_labels = set()
    for eid in exclude_ids:
        if eid in dataset.genres:
            exclude_labels.update(dataset.genres[eid])
    for sid in dataset.genres:
        dataset.genres[sid] = dataset.genres[sid].intersection(exclude_labels)

    # Build the training DataFrame from combined embeddings
    df = build_training_dataframe(dataset, exclude_ids)
    df = stratified_subsample_by_genres(df, fraction=fraction, random_seed=random_seed)

    # Combine features into a matrix, L2-normalize, then apply LLE
    X = np.vstack(df["features"])
    print("Total original feature dimension:", X.shape[1])

    # L2 normalization (row-wise)
    for i in range(len(X)):
        norm = np.linalg.norm(X[i])
        if norm > 0:
            X[i] /= norm

    # Apply Local Linear Embedding (LLE) to reduce dimension
    from sklearn.manifold import LocallyLinearEmbedding
    lle = LocallyLinearEmbedding(n_components=top_features, n_neighbors=10, random_state=random_seed)
    X_reduced = lle.fit_transform(X)
    print("Final feature dimension after LLE reduction:", X_reduced.shape[1])

    # Update each sample's feature vector in the DataFrame with the reduced features
    for i in range(len(df)):
        df.at[i, "features"] = X_reduced[i]

    # Prepare final training data
    X_final = np.vstack(df["features"])
    mlb = MultiLabelBinarizer()
    Y_final = mlb.fit_transform(df["genres"])

    # Split data into training and testing splits
    X_train, X_test, Y_train, Y_test = train_test_split(X_final, Y_final, test_size=0.2, random_state=random_seed)

    # Filter out degenerate labels (columns with only one class)
    valid_label_mask = (Y_train.sum(axis=0) > 0) & (Y_train.sum(axis=0) < Y_train.shape[0])
    if not np.all(valid_label_mask):
        degenerate = np.array(mlb.classes_)[~valid_label_mask]
        print("Warning: The following labels are degenerate and will be removed from training:")
        print(degenerate)
    Y_train = Y_train[:, valid_label_mask]
    Y_test = Y_test[:, valid_label_mask]
    mlb.classes_ = np.array(mlb.classes_)[valid_label_mask]

    # Train the final multi-label classifier using a classifier chain of LinearSVC.
    chain_classifier = ClassifierChain(LinearSVC(random_state=100, max_iter=5000, dual=False))
    chain_classifier.fit(X_train, Y_train)
    Y_pred = chain_classifier.predict(X_test)

    # Evaluate accuracy and classification report
    acc = accuracy_score(Y_test, Y_pred)
    print("Final multi-label classification accuracy:", acc)

    cr = classification_report(Y_test, Y_pred, target_names=mlb.classes_, zero_division=0)
    with open("classification_report.txt", "w") as rep_file:
        rep_file.write(cr)
    print("Classification report saved to classification_report.txt")

    # Save the trained classifier, the MultiLabelBinarizer, and the LLE transformer
    model_bundle = {
        "svm_model": chain_classifier,   # The trained multi-label classifier chain
        "mlb": mlb,                      # MultiLabelBinarizer for transforming genre labels
        "lle_model": lle                 # The LLE transformer used for dimensionality reduction
    }

    with open(model_save_path, "wb") as f:
        pickle.dump(model_bundle, f)
    print(f"Saved model bundle (with LLE reduction to {top_features} dimensions) to {model_save_path}")

    return model_bundle


def main():
    # File paths for training data
    info_file_path = "dataset/train/id_information.csv"
    genres_file_path = "dataset/train/id_genres_mmsr.tsv"
    metadata_file_path = "dataset/train/id_metadata.csv"
    word2vec_path = "dataset/train/id_lyrics_word2vec.tsv.bz2"
    resnet_path = "dataset/train/id_resnet.tsv.bz2"
    mfcc_stats_path = "dataset/train/id_mfcc_stats.tsv.bz2"

    print("Loading dataset...")
    dataset = EnhancedDataset(info_file_path, genres_file_path, metadata_file_path,
                              word2vec_path, resnet_path, mfcc_stats_path)

    eval_subset_path = "dataset/id_information_mmsr.tsv"
    train_SVM(dataset, eval_subset_path, fraction=0.2)


if __name__ == "__main__":
    main()
