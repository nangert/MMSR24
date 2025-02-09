import numpy as np
from typing import List, Dict
import pickle
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import MultiLabelBinarizer
from Music4All import Dataset, Song


class EarlyFusionRetrievalSystem:
    """
    A retrieval system that fuses multiple feature modalities (Word2Vec, ResNet, and MFCC‐Stats)
    and uses a trained multi-label SVM model for genre-based retrieval via Jaccard similarity.
    This system uses the early fusion model that applies LLE for dimensionality reduction.
    """

    def __init__(self, dataset: Dataset, svm_pkl_path: str):
        """
        Initializes the retrieval system by loading the trained SVM model bundle and
        computing fused representations for each song using the saved LLE transformer.

        Args:
            dataset (Dataset): The dataset containing song metadata and embeddings.
            svm_pkl_path (str): Path to the pickle file with the trained model bundle.
                                The bundle must contain:
                                  - "svm_model": a ClassifierChain instance
                                  - "mlb": a MultiLabelBinarizer for genre labels
                                  - "lle_model": the LLE transformer used for dimensionality reduction
        """
        self.dataset = dataset

        # Load the trained multi-label SVM model bundle.
        with open(svm_pkl_path, "rb") as f:
            model_bundle = pickle.load(f)
            self.svm_model: ClassifierChain = model_bundle["svm_model"]
            self.mlb: MultiLabelBinarizer = model_bundle["mlb"]
            self.lle_model = model_bundle["lle_model"]

        # Build fused embeddings using the saved LLE transformer.
        # Concatenation order must match training: word2vec, resnet, then mfcc_embeddings_stat.
        self.fused_embeddings = {}
        for song in self.dataset.get_all_songs():
            song_id = song.song_id
            if (song_id in self.dataset.word2vec_embeddings and
                song_id in self.dataset.resnet_embeddings and
                song_id in self.dataset.mfcc_embeddings_stat):
                concatenated = np.concatenate([
                    self.dataset.word2vec_embeddings[song_id],
                    self.dataset.resnet_embeddings[song_id],
                    self.dataset.mfcc_embeddings_stat[song_id]
                ])
                # L2 normalization
                norm = np.linalg.norm(concatenated)
                if norm > 0:
                    concatenated = concatenated / norm
                # Transform using the LLE transformer
                reduced_features = self.lle_model.transform(concatenated.reshape(1, -1))[0]
                self.fused_embeddings[song_id] = reduced_features

    @staticmethod
    def jaccard_similarity(labels_a: set, labels_b: set) -> float:
        """
        Computes the Jaccard similarity between two sets of labels.
        """
        if not labels_a and not labels_b:
            return 0.0
        return float(len(labels_a & labels_b)) / float(len(labels_a | labels_b))

    def get_retrieval(self, query_song: Song, N: int) -> List[Song]:
        """
        Retrieves the top N songs based on the Jaccard similarity of predicted genre labels.

        Args:
            query_song (Song): The query song.
            N (int): The number of songs to retrieve.

        Returns:
            List[Song]: A list of the top N most similar songs.
        """
        query_id = query_song.song_id
        if query_id not in self.fused_embeddings:
            return []

        # Predict genre labels for the query song using the reduced features.
        query_vec = self.fused_embeddings[query_id].reshape(1, -1)
        query_pred = self.svm_model.predict(query_vec)[0]
        query_labels = set(self.mlb.inverse_transform(query_pred.reshape(1, -1))[0])

        similarities = []
        for song in self.dataset.get_all_songs():
            if song.song_id == query_id:
                continue
            if song.song_id in self.fused_embeddings:
                cand_vec = self.fused_embeddings[song.song_id].reshape(1, -1)
                cand_pred = self.svm_model.predict(cand_vec)[0]
                cand_labels = set(self.mlb.inverse_transform(cand_pred.reshape(1, -1))[0])
                sim = self.jaccard_similarity(query_labels, cand_labels)
                similarities.append((song, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in similarities[:N]]

    def generate_retrieval_results(self, N: int) -> Dict[str, Dict]:
        """
        Generates retrieval results for every song in the dataset based on predicted genre labels.

        Args:
            N (int): The number of songs to retrieve for each query.

        Returns:
            Dict[str, Dict]: A dictionary mapping each query song ID to its retrieval results.
        """
        retrieval_results = {}
        for query_song in self.dataset.get_all_songs():
            retrieved = self.get_retrieval(query_song, N)
            retrieval_results[query_song.song_id] = {
                'query': query_song.to_dict(),
                'retrieved': [song.to_dict() for song in retrieved]
            }
        return retrieval_results
