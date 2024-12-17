import numpy as np
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity


class MFCCRetrievalSystem:
    """
    A retrieval system that uses MFCC features for similarity-based retrieval.
    """

    def __init__(self, dataset, similarity_cache_path: str = 'precomputed/mfcc_similarities.pkl'):
        """
        Initializes the retrieval system using the dataset's preprocessed MFCC embeddings.

        Args:
            dataset: The dataset containing song metadata and MFCC embeddings.
            similarity_cache_path (str): Path to the file where precomputed similarities will be stored.
        """
        self.mfcc_embeddings = dataset.mfcc_embeddings
        self.song_dict = {s.song_id: s for s in dataset.get_all_songs()}
        self.similarity_cache_path = similarity_cache_path
        self.similarities = self.load_or_compute_similarities()

    def load_or_compute_similarities(self) -> dict:
        """
        Loads precomputed similarities from a file if it exists, otherwise computes and saves them.

        Returns:
            dict: A dictionary containing precomputed similarities for all song pairs.
        """
        if os.path.exists(self.similarity_cache_path):
            with open(self.similarity_cache_path, 'rb') as file:
                return pickle.load(file)

        similarities = {}
        song_ids = list(self.mfcc_embeddings.keys())

        for i, query_id in enumerate(song_ids):
            query_features = self.mfcc_embeddings[query_id]
            similarities[query_id] = {}
            for j, song_id in enumerate(song_ids):
                if query_id == song_id:
                    continue
                similarity = cosine_similarity([query_features], [self.mfcc_embeddings[song_id]])[0][0]
                similarities[query_id][song_id] = similarity

        with open(self.similarity_cache_path, 'wb') as file:
            pickle.dump(similarities, file)

        return similarities

    def recommend_similar_songs(self, query_song, k: int = 5) -> list:
        """
        Recommends the top-k most similar songs for a given query song based on precomputed cosine similarity.

        Args:
            query_song (Song): The query song.
            k (int, optional): Number of songs to retrieve. Defaults to 5.

        Returns:
            list: A list of the top-k most similar Song objects.
        """
        query_id = query_song.song_id
        if query_id not in self.similarities:
            raise ValueError(f"No precomputed similarities found for song ID: {query_id}")

        similarities_with_songs = []
        for song_id, similarity in self.similarities[query_id].items():
            song = self.song_dict.get(song_id)
            if song:
                similarities_with_songs.append((song, similarity))

        # Sort by similarity descending
        similarities_with_songs.sort(key=lambda x: x[1], reverse=True)

        return [item[0] for item in similarities_with_songs[:k]]
