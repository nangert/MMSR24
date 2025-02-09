import contextlib
import os
import pickle
import bz2
import csv
import pandas as pd
import numpy as np

from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.manifold import LocallyLinearEmbedding
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
        """Load embeddings from a compressed TSV file."""
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
            ids = list(embedding_dict.keys())
            vectors = [embedding_dict[song_id] for song_id in ids]
            matrix = np.stack(vectors)
            scaler = StandardScaler()
            matrix_norm = scaler.fit_transform(matrix)
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
    Each row: [song_id, word2vec vector, resnet vector, mfcc_stats vector, genre set]
    """
    rows = []
    all_songs = dataset.get_all_songs()
    shuffled_songs = np.random.choice(all_songs, size=len(all_songs), replace=False)
    for song in shuffled_songs:
        sid = song["id"]
        if sid in exclude_ids:
            continue
        if (sid not in dataset.word2vec_embeddings or
                sid not in dataset.resnet_embeddings or
                sid not in dataset.mfcc_stats_embeddings):
            continue
        word2vec_vec = dataset.word2vec_embeddings[sid]
        resnet_vec = dataset.resnet_embeddings[sid]
        mfcc_vec = dataset.mfcc_stats_embeddings[sid]
        genres = song["genres"]
        rows.append([sid, word2vec_vec, resnet_vec, mfcc_vec, genres])
    return pd.DataFrame(rows, columns=["song_id", "word2vec", "resnet", "mfcc_stats", "genres"])


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


def train_late_fusion_multi_label(df: pd.DataFrame, output_file: str = "late_fusion_training_report.txt") -> dict:
    """
    Train a multi-label late-fusion pipeline using SVCs for each modality and a final fusion SVC.

    For each song, unimodal classifiers (wrapped in a ClassifierChain with a calibrated LinearSVC)
    are trained individually on LLE-reduced features (with 17 components) to produce probability outputs.
    These outputs are then concatenated, L2-normalized, and used directly (without further reduction)
    as input to train a final classifier chain for late fusion.
    """
    # Remove songs with fewer than 2 genres
    df = df[df["genres"].apply(lambda x: len(x) > 1)]

    # MultiLabelBinarizer for genres
    all_genres = sorted({g for genres in df["genres"] for g in genres})
    mlb = MultiLabelBinarizer(classes=all_genres)
    Y = mlb.fit_transform(df["genres"])  # shape: (samples, num_labels)

    # Prepare unimodal feature arrays
    word2vec_arr = np.vstack(df["word2vec"])
    resnet_arr = np.vstack(df["resnet"])
    mfcc_arr = np.vstack(df["mfcc_stats"])

    # Split into train/test sets (using the same indices for each modality)
    X_train_w2v, X_test_w2v, Y_train, Y_test = train_test_split(word2vec_arr, Y, test_size=0.2, random_state=100)
    X_train_res, X_test_res = train_test_split(resnet_arr, test_size=0.2, random_state=100)
    X_train_mfcc, X_test_mfcc = train_test_split(mfcc_arr, test_size=0.2, random_state=100)

    # Filter out degenerate labels: require at least 2 positive examples in training.
    min_examples = 2
    valid_label_mask = (Y_train.sum(axis=0) >= min_examples) & (Y_train.sum(axis=0) < Y_train.shape[0])
    Y_train = Y_train[:, valid_label_mask]
    Y_test = Y_test[:, valid_label_mask]
    valid_genres = [all_genres[i] for i, v in enumerate(valid_label_mask) if v]
    if Y_train.shape[1] == 0:
        raise ValueError("No valid labels found after filtering.")

    # Define base estimator: Calibrated LinearSVC (with cv=2 to avoid insufficient samples)
    base_estimator = CalibratedClassifierCV(
        LinearSVC(random_state=100, max_iter=5000, dual=False),
        cv=2
    )

    # Apply LLE (with 17 components) to each modality before training the unimodal classifier chains.
    lle_w2v = LocallyLinearEmbedding(n_components=17, n_neighbors=10, random_state=100)
    X_train_w2v_reduced = lle_w2v.fit_transform(X_train_w2v)
    X_test_w2v_reduced = lle_w2v.transform(X_test_w2v)

    lle_res = LocallyLinearEmbedding(n_components=17, n_neighbors=10, random_state=100)
    X_train_res_reduced = lle_res.fit_transform(X_train_res)
    X_test_res_reduced = lle_res.transform(X_test_res)

    lle_mfcc = LocallyLinearEmbedding(n_components=17, n_neighbors=10, random_state=100)
    X_train_mfcc_reduced = lle_mfcc.fit_transform(X_train_mfcc)
    X_test_mfcc_reduced = lle_mfcc.transform(X_test_mfcc)

    # Train unimodal classifier chains for each modality on the LLE-reduced features.
    def train_classifier_chain(X_train_mod, y_train, name):
        print(f"\n=== Training ClassifierChain for '{name}' ===")
        chain = ClassifierChain(base_estimator)
        chain.fit(X_train_mod, y_train)
        print(f"Done training {name}.")
        return chain

    with open(output_file, "w") as f, contextlib.redirect_stdout(f):
        print("Training Late Fusion Unimodal Models using SVCs on LLE-reduced features...")
        w2v_chain = train_classifier_chain(X_train_w2v_reduced, Y_train, "Word2Vec")
        res_chain = train_classifier_chain(X_train_res_reduced, Y_train, "ResNet")
        mfcc_chain = train_classifier_chain(X_train_mfcc_reduced, Y_train, "MFCC")

        # Generate stacked probability outputs for fusion
        def get_stacked_features(X_w2v_mod, X_res_mod, X_mfcc_mod):
            p_w2v = w2v_chain.predict_proba(X_w2v_mod)  # shape: (samples, num_labels)
            p_res = res_chain.predict_proba(X_res_mod)
            p_mfcc = mfcc_chain.predict_proba(X_mfcc_mod)
            return np.hstack([p_w2v, p_res, p_mfcc])

        print("\n=== Generating stacked fusion features ===")
        X_train_fusion = get_stacked_features(X_train_w2v_reduced, X_train_res_reduced, X_train_mfcc_reduced)
        X_test_fusion = get_stacked_features(X_test_w2v_reduced, X_test_res_reduced, X_test_mfcc_reduced)

        # L2-normalize the stacked fusion features (row-wise)
        for i in range(len(X_train_fusion)):
            norm = np.linalg.norm(X_train_fusion[i])
            if norm > 0:
                X_train_fusion[i] /= norm
        for i in range(len(X_test_fusion)):
            norm = np.linalg.norm(X_test_fusion[i])
            if norm > 0:
                X_test_fusion[i] /= norm

        # Train final classifier chain for fusion using SVC (with same base estimator)
        fusion_chain = ClassifierChain(base_estimator)
        fusion_chain.fit(X_train_fusion, Y_train)

        # Evaluate the final fusion model
        Y_pred = fusion_chain.predict(X_test_fusion)
        cls_rep = classification_report(Y_test, Y_pred, target_names=valid_genres, zero_division=0)
        acc = accuracy_score(Y_test, Y_pred)
        print("\n=== Late Fusion Evaluation (Multi-Label) ===")
        print("Classification Report:")
        print(cls_rep)
        print(f"Multi-label Subset Accuracy: {acc:.4f}")

    print(f"Late Fusion training report saved to {output_file}.")

    # Save all unimodal LLE models along with the classifier chains, mlb, and fusion chain.
    return {
        "mlb": mlb,
        "valid_genres": valid_genres,
        "word2vec_chain": w2v_chain,
        "resnet_chain": res_chain,
        "mfcc_chain": mfcc_chain,
        "fusion_chain": fusion_chain,
        "lle_w2v": lle_w2v,
        "lle_res": lle_res,
        "lle_mfcc": lle_mfcc
    }


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
    exclude_ids = set(pd.read_csv(eval_subset_path, sep="\t")["id"].astype(str).tolist())

    print("Building Late Fusion DataFrame...")
    df = build_late_fusion_training_df(dataset, exclude_ids)

    print("Performing stratified subsampling...")
    df = stratified_subsample_by_genres(df, fraction=0.1, random_seed=100)

    print("Training multi-label Late Fusion using SVC and LLE on unimodal features...")
    model_bundle = train_late_fusion_multi_label(df, output_file="late_fusion_training_report.txt")

    model_path = "late_fusion_model.pkl"
    print(f"Saving model bundle to {model_path}...")
    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)
    print(f"Late Fusion model saved to {model_path}.")


if __name__ == "__main__":
    main()
