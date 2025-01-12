import numpy as np
import pickle
from typing import Dict, List
from Music4All import Dataset, Song


class EarlyFusionRetrievalSystem:
    """
    An early fusion retrieval system that combines Word2Vec, ResNet, and MFCC Stat Cos embeddings,
    and uses a pre-trained SVM model for genre-based similarity retrieval.
    """
    def __init__(self, dataset: Dataset, word2vec_embeddings: Dict[str, np.ndarray],
                 resnet_embeddings: Dict[str, np.ndarray], mfcc_embeddings_stat_cos: Dict[str, np.ndarray],
                 svm_model_path: str):
        """
        Initialize the retrieval system with datasets, embeddings, and a pre-trained SVM model.

        Args:
            dataset (Dataset): The dataset containing song information.
            word2vec_embeddings (Dict[str, np.ndarray]): Word2Vec embeddings for songs.
            resnet_embeddings (Dict[str, np.ndarray]): ResNet embeddings for songs.
            mfcc_embeddings_stat_cos (Dict[str, np.ndarray]): MFCC Stat Cos embeddings for songs.
            svm_model_path (str): Path to the pre-trained SVM model (.pkl file).
        """
        self.dataset = dataset
        self.word2vec_embeddings = word2vec_embeddings
        self.resnet_embeddings = resnet_embeddings
        self.mfcc_embeddings_stat_cos = mfcc_embeddings_stat_cos

        # Load the pre-trained SVM model
        with open(svm_model_path, "rb") as f:
            self.svm_model = pickle.load(f)

    def get_fused_features(self, song_id: str) -> np.ndarray:
        """
        Combine features from Word2Vec, ResNet, and MFCC Stat Cos embeddings for a given song.

        Args:
            song_id (str): The ID of the song.

        Returns:
            np.ndarray: The fused feature vector.
        """
        if song_id not in self.word2vec_embeddings or song_id not in self.resnet_embeddings \
                or song_id not in self.mfcc_embeddings_stat_cos:
            raise ValueError(f"Missing embeddings for song ID: {song_id}")

        # Retrieve individual embeddings
        word2vec_vec = self.word2vec_embeddings[song_id]
        resnet_vec = self.resnet_embeddings[song_id]
        mfcc_vec = self.mfcc_embeddings_stat_cos[song_id]

        # Concatenate normalized features into a single vector
        fused_vec = np.concatenate([word2vec_vec, resnet_vec, mfcc_vec])
        return fused_vec

    def get_retrieval(self, query_song: Song, N: int) -> List[Song]:
        """
        Retrieve the top N most similar songs based on the SVM genre optimization.

        Args:
            query_song (Song): The query song.
            N (int): Number of songs to retrieve.

        Returns:
            List[Song]: A list of the top N most similar songs.
        """
        if self.svm_model is None:
            raise ValueError("SVM model is not loaded. Ensure the model is properly initialized.")

        query_features = self.get_fused_features(query_song.song_id).reshape(1, -1)
        all_songs = self.dataset.get_all_songs()

        # Compute distances/probabilities for all songs
        similarities = []
        for song in all_songs:
            if song.song_id == query_song.song_id:
                continue

            candidate_features = self.get_fused_features(song.song_id).reshape(1, -1)
            prob = self.svm_model.predict_proba(candidate_features)[0]
            similarities.append((song, prob))

        # Sort songs by the probability of the query genre
        query_genre = query_song.genres[0] if query_song.genres else "unknown"
        similarities = [(song, prob[self.svm_model.classes_.tolist().index(query_genre)] if query_genre in self.svm_model.classes_ else 0)
                        for song, prob in similarities]
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top N songs
        return [item[0] for item in similarities[:N]]

    def generate_retrieval_results(self, N: int) -> Dict[str, Dict]:
        """
        Generate retrieval results for all songs in the dataset.

        Args:
            N (int): Number of songs to retrieve for each query.

        Returns:
            Dict[str, Dict]: Retrieval results containing query and retrieved songs.
        """
        retrieval_results = {}
        for query_song in self.dataset.get_all_songs():
            retrieved_songs = self.get_retrieval(query_song, N)
            retrieval_results[query_song.song_id] = {
                'query': query_song.to_dict(),
                'retrieved': [song.to_dict() for song in retrieved_songs]
            }
        return retrieval_results
