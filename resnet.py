# resnet_retrieval_system.py
import numpy as np
from typing import List
from Music4All import Dataset, Song


class ResNetRetrievalSystem:
    """
    A retrieval system that uses ResNet-based image embeddings for similarity-based retrieval.
    """

    def __init__(self, dataset: Dataset):
        """
        Initializes the retrieval system with a dataset.

        Args:
            dataset (Dataset): The dataset to use for retrieval.
        """
        self.dataset = dataset
        self.resnet_embeddings = dataset.resnet_embeddings

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def get_retrieval(self, query_song: Song, N: int) -> List[Song]:
        """
        Retrieves the top N most similar songs based on ResNet embeddings.

        Args:
            query_song (Song): The song used as the query.
            N (int): The number of songs to retrieve.

        Returns:
            List[Song]: A list of the top N most similar Song objects.
        """
        query_id = query_song.song_id
        if query_id not in self.resnet_embeddings:
            raise ValueError(f"No ResNet embedding found for song ID: {query_id}")

        query_vec = self.resnet_embeddings[query_id]
        similarities = []

        for song in self.dataset.get_all_songs():
            if song.song_id == query_id:
                continue

            cand_vec = self.resnet_embeddings[song.song_id]
            sim = self.cosine_similarity(query_vec, cand_vec)
            similarities.append((song, sim))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top N
        return [item[0] for item in similarities[:N]]

    def generate_retrieval_results(self, N: int) -> dict:
        """
        Generates retrieval results for all songs in the dataset based on ResNet similarity.

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
