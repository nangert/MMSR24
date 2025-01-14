import numpy as np
import pickle
from typing import Dict, List
from Music4All import Dataset, Song


class LateFusionRetrievalSystem:
    """
    A late fusion retrieval system that combines probabilities from unimodal classifiers
    (Word2Vec, ResNet, MFCC Stat) and uses a final ClassifierChain for multi-label retrieval.
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

        Args:
            dataset (Dataset): The dataset containing song information.
            word2vec_embeddings (Dict[str, np.ndarray]): Word2Vec embeddings for songs.
            resnet_embeddings (Dict[str, np.ndarray]): ResNet embeddings for songs.
            mfcc_embeddings_stat_cos (Dict[str, np.ndarray]): MFCC Stat embeddings for songs.
            late_fusion_model_path (str): Path to the late fusion model (.pkl) that
                                          contains unimodal ClassifierChains and final fusion ClassifierChain.
        """
        self.dataset = dataset
        self.word2vec_embeddings = word2vec_embeddings
        self.resnet_embeddings = resnet_embeddings
        self.mfcc_embeddings_stat_cos = mfcc_embeddings_stat_cos

        # Load the late fusion model bundle
        # {
        #   "mlb": <MultiLabelBinarizer>,
        #   "word2vec_chain": <ClassifierChain>,
        #   "resnet_chain":   <ClassifierChain>,
        #   "mfcc_chain":     <ClassifierChain>,
        #   "fusion_chain":   <ClassifierChain>
        # }
        with open(late_fusion_model_path, "rb") as f:
            model_bundle = pickle.load(f)

        self.mlb = model_bundle["mlb"]
        self.word2vec_chain = model_bundle["word2vec_chain"]
        self.resnet_chain = model_bundle["resnet_chain"]
        self.mfcc_chain = model_bundle["mfcc_chain"]
        self.fusion_chain = model_bundle["fusion_chain"]

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
         1. Passing each modality's raw embedding into its ClassifierChain to get probabilities.
         2. Concatenating these probability vectors.

        Args:
            song_id (str): The ID of the song.

        Returns:
            np.ndarray: The fused probability vector for the final ClassifierChain.
        """
        if song_id not in self.word2vec_embeddings \
           or song_id not in self.resnet_embeddings \
           or song_id not in self.mfcc_embeddings_stat_cos:
            raise ValueError(f"Missing embeddings for song ID: {song_id}")

        # Gather each modality's embedding
        word2vec_vec = self.word2vec_embeddings[song_id].reshape(1, -1)
        resnet_vec   = self.resnet_embeddings[song_id].reshape(1, -1)
        mfcc_vec     = self.mfcc_embeddings_stat_cos[song_id].reshape(1, -1)

        # Get unimodal probabilities from each ClassifierChain
        word2vec_probs = self.word2vec_chain.predict_proba(word2vec_vec)
        resnet_probs   = self.resnet_chain.predict_proba(resnet_vec)
        mfcc_probs     = self.mfcc_chain.predict_proba(mfcc_vec)

        # Concatenate unimodal probability outputs => final input vector
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
        if query_id not in self.word2vec_embeddings \
                or query_id not in self.resnet_embeddings \
                or query_id not in self.mfcc_embeddings_stat_cos:
            return []

        # Get predicted labels for the query song
        query_vector = self.get_late_fusion_vector(query_id)
        query_pred_bin = self.fusion_chain.predict(query_vector)[0]

        # Align predicted classes to the expected size (len(self.mlb.classes_))
        aligned_query_pred = np.zeros(len(self.mlb.classes_), dtype=int)
        aligned_query_pred[:len(query_pred_bin)] = query_pred_bin

        # Convert binary predictions to a set of genre labels
        query_labels = set(self.mlb.inverse_transform(aligned_query_pred.reshape(1, -1))[0])

        # Compare with all other songs in the dataset
        similarities = []
        for song in self.dataset.get_all_songs():
            if song.song_id == query_id:
                continue

            if song.song_id in self.word2vec_embeddings \
                    and song.song_id in self.resnet_embeddings \
                    and song.song_id in self.mfcc_embeddings_stat_cos:
                candidate_vector = self.get_late_fusion_vector(song.song_id)
                candidate_pred_bin = self.fusion_chain.predict(candidate_vector)[0]

                # Align candidate predictions to the expected size
                aligned_candidate_pred = np.zeros(len(self.mlb.classes_), dtype=int)
                aligned_candidate_pred[:len(candidate_pred_bin)] = candidate_pred_bin

                # Convert binary predictions to a set of genre labels
                candidate_labels = set(self.mlb.inverse_transform(aligned_candidate_pred.reshape(1, -1))[0])

                # Compute Jaccard similarity
                similarity = self.jaccard_similarity(query_labels, candidate_labels)
                similarities.append((song, similarity))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return the top N
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
