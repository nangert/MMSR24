# mfcc_retrieval_system.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class MFCCRetrievalSystem:
    """
    A retrieval system that uses MFCC features for similarity-based retrieval.
    """

    def __init__(self, dataset):
        """
        Initializes the retrieval system using the dataset's preprocessed MFCC embeddings.

        Args:
            dataset: The dataset containing song metadata and MFCC embeddings.
        """
        self.mfcc_embeddings = dataset.mfcc_embeddings
        self.song_dict = {s.song_id: s for s in dataset.get_all_songs()}

    def recommend_similar_songs(self, query_song, k: int = 5) -> list:
        """
        Recommends the top-k most similar songs for a given query song based on cosine similarity of MFCC features.

        Args:
            query_song (Song): The query song.
            k (int, optional): Number of songs to retrieve. Defaults to 5.

        Returns:
            list: A list of the top-k most similar Song objects.
        """
        query_id = query_song.song_id
        query_features = self.mfcc_embeddings.get(query_id)
        if query_features is None:
            raise ValueError(f"No MFCC features found for song ID: {query_id}")

        similarities = []
        for song_id, features in self.mfcc_embeddings.items():
            if song_id == query_id:
                continue

            sim = cosine_similarity([query_features], [features])[0][0]
            song = self.song_dict.get(song_id)
            if song:
                similarities.append((song, sim))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        return [item[0] for item in similarities[:k]]

    def generate_retrieval_results(self, N: int) -> dict:
        """
        Generates retrieval results for all songs in the dataset.

        Args:
            N (int): The number of songs to retrieve for each query.

        Returns:
            dict: A dictionary of retrieval results.
        """
        retrieval_results = {}
        for query_song in self.song_dict.values():
            retrieved_songs = self.recommend_similar_songs(query_song, N)
            retrieval_results[query_song.song_id] = {
                'query': query_song.to_dict(),
                'retrieved': [song.to_dict() for song in retrieved_songs]
            }
        return retrieval_results
