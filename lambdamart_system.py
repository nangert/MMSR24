# lambdamart_system.py
import numpy as np
import lightgbm as lgb
from typing import List
from Music4All import Dataset, Song


class LambdaMARTRetrievalSystem:
    """
    Retrieves songs using a pre-trained LambdaMART (LightGBM) ranker from allRank.
    """

    def __init__(self, dataset: Dataset, model_path: str):
        """
        Args:
            dataset (Dataset): The dataset that provides track features.
            model_path (str): Path to the saved LightGBM model file (e.g. .txt).
        """
        self.dataset = dataset
        self.model = lgb.Booster(model_file=model_path)

    def _build_feature_vector(self, track_id: str) -> np.ndarray:
        """
        Retrieves the feature vector for a track.

        Args:
            track_id (str): The track's identifier.

        Returns:
            np.ndarray: The feature vector used by the ranker.
        """
        return self.dataset.lambdamart_features.get(track_id, np.zeros(self.dataset.lambdamart_feature_dim))

    def get_retrieval(self, query_song: Song, N: int) -> List[Song]:
        """
        Ranks all other songs in the dataset with respect to the query_song
        using the trained LambdaMART model. This is a simplified approach that
        assigns a relevance score to each candidate track based on both
        the query's and candidate's feature vectors.

        Args:
            query_song (Song): The query song.
            N (int): Number of similar songs to retrieve.

        Returns:
            List[Song]: The top N most relevant songs.
        """
        query_id = query_song.song_id
        query_vec = self._build_feature_vector(query_id)

        candidates = []
        for song in self.dataset.get_all_songs():
            if song.song_id == query_id:
                continue

            cand_vec = self._build_feature_vector(song.song_id)

            # Example approach: pass concatenated features (query + candidate) to the model
            # (One might do differences or another scheme if trained in a different way.)
            combined_vec = np.concatenate([query_vec, cand_vec], axis=0)
            score = self.model.predict(combined_vec.reshape(1, -1))[0]

            candidates.append((song, score))

        # Sort descending by predicted score
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in candidates[:N]]

    def generate_retrieval_results(self, N: int) -> dict:
        """
        Generates retrieval results (top N) for every song in the dataset.

        Args:
            N (int): Number of songs to retrieve per query.

        Returns:
            dict: {query_song_id: {"query":..., "retrieved":[...]}, ...}
        """
        results = {}
        for query_song in self.dataset.get_all_songs():
            retrieved_songs = self.get_retrieval(query_song, N)
            results[query_song.song_id] = {
                'query': query_song.to_dict(),
                'retrieved': [song.to_dict() for song in retrieved_songs]
            }
        return results
