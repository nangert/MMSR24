import numpy as np
from typing import List, Dict
import pickle
from sklearn.decomposition import PCA
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import MultiLabelBinarizer
from Music4All import Dataset, Song


class EarlyFusionRetrievalSystem:
    """
    A retrieval system that fuses multiple feature modalities (e.g., Word2Vec, ResNet, MFCC) and
    uses a multi-label SVM model for genre-based retrieval via Jaccard similarity.
    """

    def __init__(
        self,
        dataset: Dataset,
        svm_pkl_path: str
    ):
        """
        Initializes the retrieval system with a dataset, applies PCA, and loads a trained SVM model.

        Args:
            dataset (Dataset): The dataset of songs.
            svm_pkl_path (str): Path to a pickle file containing the trained multi-label SVM model.
        """
        self.dataset = dataset

        # Apply PCA to feature modalities (copying the original embeddings)
        self.pca_embeddings = {}  # Store reduced embeddings separately
        pca_target_dims = {
            "word2vec_embeddings": 50,
            "resnet_embeddings": 50,
            "mfcc_embeddings_stat": 50
        }
        self._apply_pca_per_modality(pca_target_dims)

        # Fuse embeddings for each song
        fused_embeddings = {}
        for song in dataset.get_all_songs():
            song_id = song.song_id
            if (
                song_id in self.pca_embeddings.get("word2vec_embeddings", {})
                and song_id in self.pca_embeddings.get("resnet_embeddings", {})
                and song_id in self.pca_embeddings.get("mfcc_embeddings_stat", {})
            ):
                fused_embeddings[song_id] = np.concatenate([
                    self.pca_embeddings["word2vec_embeddings"][song_id],
                    self.pca_embeddings["resnet_embeddings"][song_id],
                    self.pca_embeddings["mfcc_embeddings_stat"][song_id]
                ])

        self.fused_embeddings = fused_embeddings

        # Load the trained multi-label SVM model
        with open(svm_pkl_path, "rb") as f:
            model_bundle = pickle.load(f)
            self.svm_model: ClassifierChain = model_bundle["svm_model"]
            self.genre2hash: Dict[str, int] = model_bundle["genre2hash"]
            self.mlb: MultiLabelBinarizer = model_bundle["mlb"]

    def _apply_pca_per_modality(self, pca_target_dims: Dict[str, int]):
        """
        For each modality, optionally reduce feature dimensions via PCA.
        The number of components is defined in pca_target_dims.
        This method ensures the original embeddings remain unaffected by creating copies.
        """
        for embedding_dict_name, target_dim in pca_target_dims.items():
            original_embedding_dict = getattr(self.dataset, embedding_dict_name, {})
            if not original_embedding_dict:
                continue

            # Create a copy of the embeddings to avoid modifying the original data
            embedding_dict = {song_id: np.copy(vec) for song_id, vec in original_embedding_dict.items()}

            ids = list(embedding_dict.keys())
            vectors = [embedding_dict[song_id] for song_id in ids]
            matrix = np.stack(vectors)

            original_dim = matrix.shape[1]

            pca = PCA(n_components=target_dim, random_state=100)
            matrix_pca = pca.fit_transform(matrix)

            # Store the reduced embeddings in the separate pca_embeddings dictionary
            reduced_embeddings = {ids[i]: matrix_pca[i] for i in range(len(ids))}
            self.pca_embeddings[embedding_dict_name] = reduced_embeddings

    @staticmethod
    def jaccard_similarity(labels_a: set, labels_b: set) -> float:
        """
        Computes the Jaccard similarity between two sets of labels.

        Args:
            labels_a (set): First set of labels.
            labels_b (set): Second set of labels.

        Returns:
            float: Jaccard similarity (0.0 to 1.0).
        """
        if not labels_a and not labels_b:
            return 0.0
        return float(len(labels_a & labels_b)) / float(len(labels_a | labels_b))

    def get_retrieval(self, query_song: Song, N: int) -> List[Song]:
        """
        Retrieves the top N songs based on Jaccard similarity of predicted genre labels.

        Args:
            query_song (Song): The query song.
            N (int): Number of songs to retrieve.

        Returns:
            List[Song]: A list of the top N most similar songs.
        """
        query_id = query_song.song_id
        if query_id not in self.fused_embeddings:
            return []

        # Predict labels for the query song
        query_vec = self.fused_embeddings[query_id].reshape(1, -1)
        query_pred_bin = self.svm_model.predict(query_vec)[0]

        # Align shape if needed
        aligned_query_pred = np.zeros(len(self.mlb.classes_), dtype=int)
        aligned_query_pred[:len(query_pred_bin)] = query_pred_bin
        query_labels = set(self.mlb.inverse_transform(aligned_query_pred.reshape(1, -1))[0])

        # Compare with all other songs in the dataset
        similarities = []
        for song in self.dataset.get_all_songs():
            if song.song_id == query_id:
                continue

            if song.song_id in self.fused_embeddings:
                cand_vec = self.fused_embeddings[song.song_id].reshape(1, -1)
                cand_pred_bin = self.svm_model.predict(cand_vec)[0]

                # Align shape if needed
                aligned_cand_pred = np.zeros(len(self.mlb.classes_), dtype=int)
                aligned_cand_pred[:len(cand_pred_bin)] = cand_pred_bin
                cand_labels = set(self.mlb.inverse_transform(aligned_cand_pred.reshape(1, -1))[0])

                # Compute Jaccard similarity
                similarity = self.jaccard_similarity(query_labels, cand_labels)
                similarities.append((song, similarity))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return the top N
        return [item[0] for item in similarities[:N]]

    def generate_retrieval_results(self, N: int) -> Dict[str, Dict]:
        """
        Generates retrieval results for all songs in the dataset based on Jaccard similarity of predicted labels.

        Args:
            N (int): Number of songs to retrieve for each query.

        Returns:
            Dict[str, Dict]: A dictionary containing queries and their retrieved songs.
        """
        retrieval_results = {}
        for query_song in self.dataset.get_all_songs():
            retrieved_songs = self.get_retrieval(query_song, N)
            retrieval_results[query_song.song_id] = {
                'query': query_song.to_dict(),
                'retrieved': [song.to_dict() for song in retrieved_songs]
            }
        return retrieval_results
