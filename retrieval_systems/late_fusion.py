import numpy as np
import pickle
from typing import Dict, List
from Music4All import Dataset, Song


class LateFusionRetrievalSystem:
    """
    A late fusion retrieval system that combines probabilities from unimodal classifiers
    (Word2Vec, ResNet, MFCC Stat) and uses a final SVM model for genre-based similarity.
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
                                          contains unimodal SVMs and final fusion SVM.
        """
        self.dataset = dataset
        self.word2vec_embeddings = word2vec_embeddings
        self.resnet_embeddings = resnet_embeddings
        self.mfcc_embeddings_stat_cos = mfcc_embeddings_stat_cos

        # Load the late fusion model bundle:
        # {
        #   "unimodal_svms": {
        #       "word2vec": <CalibratedClassifierCV>,
        #       "resnet":   <CalibratedClassifierCV>,
        #       "mfcc_stats": <CalibratedClassifierCV>
        #   },
        #   "fusion_svm": <SVC>
        # }
        with open(late_fusion_model_path, "rb") as f:
            model_bundle = pickle.load(f)

        self.unimodal_svms = model_bundle["unimodal_svms"]
        self.fusion_svm = model_bundle["fusion_svm"]

    def get_late_fusion_vector(self, song_id: str) -> np.ndarray:
        """
        Produce a late-fusion feature vector by:
         1. Passing each modality's raw embedding into its calibrated SVM to get probabilities.
         2. Concatenating these probability vectors.

        Args:
            song_id (str): The ID of the song.

        Returns:
            np.ndarray: The fused probability vector for the final SVM.
        """
        if song_id not in self.word2vec_embeddings \
           or song_id not in self.resnet_embeddings \
           or song_id not in self.mfcc_embeddings_stat_cos:
            raise ValueError(f"Missing embeddings for song ID: {song_id}")

        # Gather each modality's embedding
        word2vec_vec = self.word2vec_embeddings[song_id].reshape(1, -1)
        resnet_vec   = self.resnet_embeddings[song_id].reshape(1, -1)
        mfcc_vec     = self.mfcc_embeddings_stat_cos[song_id].reshape(1, -1)

        # Get unimodal probabilities from each calibrated SVM
        word2vec_probs = self.unimodal_svms["word2vec"].predict_proba(word2vec_vec)
        resnet_probs   = self.unimodal_svms["resnet"].predict_proba(resnet_vec)
        mfcc_probs     = self.unimodal_svms["mfcc_stats"].predict_proba(mfcc_vec)

        # Concatenate unimodal probability outputs => final input vector
        late_fusion_vector = np.hstack([word2vec_probs, resnet_probs, mfcc_probs])

        return late_fusion_vector

    def get_retrieval(self, query_song: Song, N: int) -> List[Song]:
        """
        Retrieve the top N songs most similar to the query based on the fused probability
        from the final SVM, focusing on the query's primary genre.

        Args:
            query_song (Song): The query song.
            N (int): Number of songs to retrieve.

        Returns:
            List[Song]: A list of the top N most similar songs.
        """
        # Basic check
        if self.fusion_svm is None or self.unimodal_svms is None:
            raise ValueError("Late fusion model not loaded properly.")

        # Get the final SVM probability distribution for the query
        query_vector = self.get_late_fusion_vector(query_song.song_id)
        all_songs = self.dataset.get_all_songs()

        similarities = []
        for song in all_songs:
            # Exclude the query itself
            if song.song_id == query_song.song_id:
                continue

            # Get the late-fusion vector for the candidate
            candidate_vector = self.get_late_fusion_vector(song.song_id)
            # SVM probability for candidate
            prob = self.fusion_svm.predict_proba(candidate_vector)[0]
            similarities.append((song, prob))

        # Identify the query's first genre (if it exists)
        query_genre = query_song.genres[0] if query_song.genres else "unknown"

        # Sort by the probability of the query's genre
        if query_genre in self.fusion_svm.classes_:
            genre_index = self.fusion_svm.classes_.tolist().index(query_genre)
            similarities = [(song, prob[genre_index]) for (song, prob) in similarities]
        else:
            # If we don't have the query genre in classes, give them probability 0
            similarities = [(song, 0.0) for (song, prob) in similarities]

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
