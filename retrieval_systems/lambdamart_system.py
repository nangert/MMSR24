# lambdamart_system.py
import pickle
from typing import List
from Music4All import Dataset, Song


class LambdaMARTRetrievalSystem:
    """
    Retrieves songs using a pre-trained LightGBM-based LambdaMART model
    that was trained on track-level MFCC BoW features from 'id_mfcc_bow.tsv.bz2'.
    """

    def __init__(self, dataset: Dataset, model_path: str, feature_dim: int):
        """
        Args:
            dataset (Dataset): The dataset (with track features in mfcc_embeddings_bow, etc.).
            model_path (str): Path to the .pth file (pickled LightGBM model).
            feature_dim (int): The dimensionality of the MFCC BoW feature vectors.
        """
        self.dataset = dataset
        self.feature_dim = feature_dim
        self.model = self._load_model(model_path)

    @staticmethod
    def _load_model(model_path: str):
        """
        Loads the trained LightGBM model (pickled) from disk.
        """
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model

    def get_retrieval(self, query_song: Song, N: int) -> List[Song]:
        """
        Ranks all other songs in the dataset with respect to the query_song.
        For each candidate, runs the track features through the LightGBM
        model to get a predicted score, then sorts descending.

        Args:
            query_song (Song): The 'query' song.
            N (int): Number of similar songs to retrieve.

        Returns:
            List[Song]: The top N most relevant songs.
        """
        query_id = query_song.song_id
        candidates = []

        for candidate_song in self.dataset.get_all_songs():
            candidate_id = candidate_song.song_id
            # Skip the same song
            if candidate_id == query_id:
                continue

            # Check if track features are available
            if candidate_id not in self.dataset.mfcc_embeddings_bow:
                continue

            mfcc_feats = self.dataset.mfcc_embeddings_bow[candidate_id]

            # LightGBM expects a 2D array, e.g. shape (1, feature_dim)
            feats_2d = mfcc_feats.reshape(1, -1)
            score = self.model.predict(feats_2d)[0]

            candidates.append((candidate_song, score))

        # Sort by predicted score descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        return [song for (song, _) in candidates[:N]]

    def generate_retrieval_results(self, N: int) -> dict:
        """
        Generates retrieval results (top N) for every song in the dataset.
        """
        results = {}
        for query_song in self.dataset.get_all_songs():
            retrieved_songs = self.get_retrieval(query_song, N)
            results[query_song.song_id] = {
                'query': query_song.to_dict(),
                'retrieved': [song.to_dict() for song in retrieved_songs]
            }
        return results
