# embedding_system.py
import numpy as np
from typing import List, Dict
from Music4All import Dataset, Song


class EmbeddingRetrievalSystem:
    """
    A retrieval system that uses any type of embeddings (e.g., BERT, ResNet, VGG19) for similarity-based retrieval.
    """

    def __init__(self, dataset: Dataset, embeddings: Dict[str, np.ndarray], embedding_name: str):
        """
        Initializes the retrieval system with a dataset and embeddings.

        Args:
            dataset (Dataset): The dataset to use for retrieval.
            embeddings (Dict[str, np.ndarray]): A dictionary mapping song IDs to their embeddings.
            embedding_name (str): A label for the embedding type (e.g., 'BERT', 'ResNet', 'VGG19').
        """
        self.dataset = dataset
        self.embeddings = embeddings
        self.embedding_name = embedding_name

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def get_retrieval(self, query_song: Song, N: int) -> List[Song]:
        """
        Retrieves the top N most similar songs based on embeddings.

        Args:
            query_song (Song): The song used as the query.
            N (int): The number of songs to retrieve.

        Returns:
            List[Song]: A list of the top N most similar Song objects.
        """
        query_id = query_song.song_id
        if query_id not in self.embeddings:
            raise ValueError(f"No {self.embedding_name} embedding found for song ID: {query_id}")

        query_vec = self.embeddings[query_id]
        similarities = []

        for song in self.dataset.get_all_songs():
            if song.song_id == query_id:
                continue

            cand_vec = self.embeddings[song.song_id]
            sim = self.cosine_similarity(query_vec, cand_vec)
            similarities.append((song, sim))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top N songs
        return [item[0] for item in similarities[:N]]

    def generate_retrieval_results(self, N: int) -> dict:
        """
        Generates retrieval results for all songs in the dataset based on embeddings.

        Args:
            N (int): The number of songs to retrieve for each query.

        Returns:
            dict: A dictionary where each key is a query song ID and the value contains the query and retrieved songs.
        """
        retrieval_results = {}
        for query_song in self.dataset.get_all_songs():
            retrieved_songs = self.get_retrieval(query_song, N)
            retrieval_results[query_song.song_id] = {
                'query': query_song.to_dict(),
                'retrieved': [song.to_dict() for song in retrieved_songs]
            }
        return retrieval_results
