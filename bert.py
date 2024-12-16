# bert_retrieval_system.py
import numpy as np
from typing import List
from Music4All import Dataset, Song


class BertRetrievalSystem:
    """
    A retrieval system that uses BERT-based lyric embeddings for similarity-based retrieval.
    """

    def __init__(self, dataset: Dataset):
        """
        Initializes the retrieval system with a dataset.

        Args:
            dataset (Dataset): The dataset to use for retrieval.
        """
        self.dataset = dataset
        self.bert_embeddings = dataset.bert_embeddings

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def get_retrieval(self, query_song: Song, N: int) -> List[Song]:
        """
        Retrieves the top N most similar songs based on BERT embeddings.

        Args:
            query_song (Song): The song used as the query.
            N (int): The number of songs to retrieve.

        Returns:
            List[Song]: A list of the top N most similar Song objects.
        """
        query_id = query_song.song_id
        if query_id not in self.bert_embeddings:
            raise ValueError(f"No BERT embedding found for song ID: {query_id}")

        query_vec = self.bert_embeddings[query_id]
        similarities = []

        for song in self.dataset.get_all_songs():
            if song.song_id == query_id:
                continue
            if song.song_id not in self.bert_embeddings:
                raise ValueError(f"No BERT embedding found for song ID: {song.song_id}")
            cand_vec = self.bert_embeddings[song.song_id]
            sim = self.cosine_similarity(query_vec, cand_vec)
            similarities.append((song, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top N
        return [item[0] for item in similarities[:N]]

    def generate_retrieval_results(self, N: int) -> dict:
        """
        Generates retrieval results for all songs in the dataset based on BERT similarity.

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
