import numpy as np
import pickle
from typing import Dict, List
from Music4All import Dataset, Song


class LateFusionRetrievalSystem:
    """
    A late fusion retrieval system that combines probabilities from unimodal classifiers
    (Word2Vec, ResNet, MFCC Stat) and uses a final ClassifierChain for multi-label retrieval.
    This version applies Locally Linear Embedding (LLE) with 17 components to each modality
    before the unimodal classifiers (as was done during training), and does not apply any additional
    dimensionality reduction after stacking.
    """

    def __init__(
        self,
        dataset: Dataset,
        word2vec_embeddings: Dict[str, np.ndarray],
        resnet_embeddings: Dict[str, np.ndarray],
        mfcc_embeddings_stat_cos: Dict[str, np.ndarray],
        late_fusion_model_path: str
    ):
        """
        Initialize the retrieval system with the dataset, embeddings, and the late fusion model.
        The model bundle (loaded from late_fusion_model_path) contains:
            - "mlb": MultiLabelBinarizer
            - "word2vec_chain": ClassifierChain for Word2Vec modality
            - "resnet_chain":   ClassifierChain for ResNet modality
            - "mfcc_chain":     ClassifierChain for MFCC modality
            - "fusion_chain":   Final fusion ClassifierChain
            - "lle_w2v":       LLE transformer for Word2Vec (17 components)
            - "lle_res":       LLE transformer for ResNet (17 components)
            - "lle_mfcc":      LLE transformer for MFCC (17 components)
        """
        self.dataset = dataset
        self.word2vec_embeddings = word2vec_embeddings
        self.resnet_embeddings = resnet_embeddings
        self.mfcc_embeddings_stat_cos = mfcc_embeddings_stat_cos

        # Load the late fusion model bundle.
        with open(late_fusion_model_path, "rb") as f:
            model_bundle = pickle.load(f)

        self.mlb = model_bundle["mlb"]
        self.word2vec_chain = model_bundle["word2vec_chain"]
        self.resnet_chain = model_bundle["resnet_chain"]
        self.mfcc_chain = model_bundle["mfcc_chain"]
        self.fusion_chain = model_bundle["fusion_chain"]

        # Load the LLE transformers used during training for each modality.
        self.lle_w2v = model_bundle["lle_w2v"]
        self.lle_res = model_bundle["lle_res"]
        self.lle_mfcc = model_bundle["lle_mfcc"]

    @staticmethod
    def jaccard_similarity(labels_a: set, labels_b: set) -> float:
        """
        Compute the Jaccard similarity between two sets of labels.

        Args:
            labels_a (set): First set of labels.
            labels_b (set): Second set of labels.

        Returns:
            float: Jaccard similarity (0.0 to 1.0).
        """
        if not labels_a and not labels_b:
            return 0.0
        return float(len(labels_a & labels_b)) / float(len(labels_a | labels_b))

    def get_late_fusion_vector(self, song_id: str) -> np.ndarray:
        """
        Produce a late-fusion feature vector by:
         1. Taking each modality's raw embedding,
         2. L2-normalizing it,
         3. Transforming it with the corresponding LLE transformer (to 17 dimensions),
         4. Passing the reduced vector into its unimodal ClassifierChain to get probability outputs,
         5. Concatenating these probability vectors.

        Args:
            song_id (str): The ID of the song.

        Returns:
            np.ndarray: The fused probability vector for input to the final fusion classifier.
        """
        if (song_id not in self.word2vec_embeddings or
            song_id not in self.resnet_embeddings or
            song_id not in self.mfcc_embeddings_stat_cos):
            raise ValueError(f"Missing embeddings for song ID: {song_id}")

        # Get raw embeddings for each modality.
        word2vec_raw = self.word2vec_embeddings[song_id]
        resnet_raw = self.resnet_embeddings[song_id]
        mfcc_raw = self.mfcc_embeddings_stat_cos[song_id]

        # L2-normalize each modality's raw embedding.
        def l2_normalize(vec: np.ndarray) -> np.ndarray:
            norm = np.linalg.norm(vec)
            return vec if norm == 0 else vec / norm

        word2vec_norm = l2_normalize(word2vec_raw)
        resnet_norm = l2_normalize(resnet_raw)
        mfcc_norm = l2_normalize(mfcc_raw)

        # Apply the corresponding LLE transformer (which was trained to reduce to 17 components).
        word2vec_reduced = self.lle_w2v.transform(word2vec_norm.reshape(1, -1))
        resnet_reduced = self.lle_res.transform(resnet_norm.reshape(1, -1))
        mfcc_reduced = self.lle_mfcc.transform(mfcc_norm.reshape(1, -1))

        # Get unimodal probability outputs from each ClassifierChain.
        word2vec_probs = self.word2vec_chain.predict_proba(word2vec_reduced)
        resnet_probs = self.resnet_chain.predict_proba(resnet_reduced)
        mfcc_probs = self.mfcc_chain.predict_proba(mfcc_reduced)

        # Concatenate the probability outputs.
        late_fusion_vector = np.hstack([word2vec_probs, resnet_probs, mfcc_probs])
        return late_fusion_vector

    def get_retrieval(self, query_song: Song, N: int) -> List[Song]:
        """
        Retrieve the top N songs most similar to the query based on Jaccard similarity
        of the predicted genre sets.

        Args:
            query_song (Song): The query song.
            N (int): Number of songs to retrieve.

        Returns:
            List[Song]: A list of the top N most similar songs.
        """
        if not query_song.genres:
            return []

        query_id = query_song.song_id
        # Check that required embeddings exist.
        if (query_id not in self.word2vec_embeddings or
            query_id not in self.resnet_embeddings or
            query_id not in self.mfcc_embeddings_stat_cos):
            return []

        # Get the late-fusion vector for the query song.
        query_vector = self.get_late_fusion_vector(query_id)
        query_pred_bin = self.fusion_chain.predict(query_vector)[0]

        # Align predictions to the expected size (if necessary).
        aligned_query_pred = np.zeros(len(self.mlb.classes_), dtype=int)
        aligned_query_pred[:len(query_pred_bin)] = query_pred_bin

        # Convert the binary prediction to a set of genre labels.
        query_labels = set(self.mlb.inverse_transform(aligned_query_pred.reshape(1, -1))[0])

        similarities = []
        for song in self.dataset.get_all_songs():
            if song.song_id == query_id:
                continue

            # Ensure the song has all required embeddings.
            if (song.song_id in self.word2vec_embeddings and
                song.song_id in self.resnet_embeddings and
                song.song_id in self.mfcc_embeddings_stat_cos):
                candidate_vector = self.get_late_fusion_vector(song.song_id)
                candidate_pred_bin = self.fusion_chain.predict(candidate_vector)[0]

                aligned_candidate_pred = np.zeros(len(self.mlb.classes_), dtype=int)
                aligned_candidate_pred[:len(candidate_pred_bin)] = candidate_pred_bin
                candidate_labels = set(self.mlb.inverse_transform(aligned_candidate_pred.reshape(1, -1))[0])

                similarity = self.jaccard_similarity(query_labels, candidate_labels)
                similarities.append((song, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in similarities[:N]]

    def generate_retrieval_results(self, N: int) -> Dict[str, Dict]:
        """
        Generate retrieval results for all songs in the dataset.

        Args:
            N (int): Number of songs to retrieve per query.

        Returns:
            Dict[str, Dict]: Retrieval results containing both the query info and the retrieved songs.
        """
        retrieval_results = {}
        for query_song in self.dataset.get_all_songs():
            retrieved_songs = self.get_retrieval(query_song, N)
            retrieval_results[query_song.song_id] = {
                'query': query_song.to_dict(),
                'retrieved': [song.to_dict() for song in retrieved_songs]
            }
        return retrieval_results
