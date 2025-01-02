import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity


class MFCCRetrievalSystem:
    """
    A retrieval system that uses MFCC features for similarity-based retrieval.
    """

    def __init__(self, dataset, similarity_cache_path_merged: str = 'precomputed/mfcc_similarities.pkl', similarity_cache_path_bow: str = 'precomputed/mfcc_similarities_bow.pkl', similarity_cache_path_stat: str = 'precomputed/mfcc_similarities_stat.pkl'):
        """
        Initializes the retrieval system using the dataset's preprocessed MFCC embeddings.

        Args:
            dataset: The dataset containing song metadata and MFCC embeddings.
            similarity_cache_path (str): Path to the file where precomputed similarities will be stored.
        """
        self.mfcc_embeddings_merged = dataset.mfcc_embeddings_merged
        self.mfcc_embeddings_bow = dataset.mfcc_embeddings_bow
        self.mfcc_embeddings_stat = dataset.mfcc_embeddings_stat
        self.song_dict = {s.song_id: s for s in dataset.get_all_songs()}
        self.similarity_cache_path_merged = similarity_cache_path_merged
        self.similarity_cache_path_bow = similarity_cache_path_bow
        self.similarity_cache_path_stat = similarity_cache_path_stat
        self.similarities_merged = self.load_or_compute_similarities_merged()
        self.similarities_bow = self.load_or_compute_similarities_bow()
        self.similarities_stat = self.load_or_compute_similarities_stat()

    def load_or_compute_similarities_merged(self) -> dict:
        """
        Loads precomputed similarities from a file if it exists, otherwise computes and saves them.

        Returns:
            dict: A dictionary containing precomputed similarities for all song pairs.
        """
        if os.path.exists(self.similarity_cache_path_merged):
            with open(self.similarity_cache_path_merged, 'rb') as file:
                return pickle.load(file)

        similarities = {}
        song_ids = list(self.mfcc_embeddings_merged.keys())

        for i, query_id in enumerate(song_ids):
            query_features = self.mfcc_embeddings_merged[query_id]
            similarities_with_songs = []

            for j, song_id in enumerate(song_ids):
                if query_id == song_id:
                    continue
                similarity = cosine_similarity([query_features], [self.mfcc_embeddings_merged[song_id]])[0][0]
                similarities_with_songs.append((song_id, similarity))

            # Sort by similarity descending and keep only the top 100
            similarities_with_songs.sort(key=lambda x: x[1], reverse=True)
            similarities[query_id] = {song_id: similarity for song_id, similarity in similarities_with_songs[:100]}

            if i % 100 == 0:
                print(f'Processed {i} songs, merged')

        with open(self.similarity_cache_path_merged, 'wb') as file:
            pickle.dump(similarities, file)

        return similarities

    def load_or_compute_similarities_bow(self) -> dict:
        """
        Loads precomputed similarities from a file if it exists, otherwise computes and saves them.

        Returns:
            dict: A dictionary containing precomputed similarities for all song pairs.
        """
        if os.path.exists(self.similarity_cache_path_bow):
            with open(self.similarity_cache_path_bow, 'rb') as file:
                return pickle.load(file)

        similarities = {}
        song_ids = list(self.mfcc_embeddings_bow.keys())

        for i, query_id in enumerate(song_ids):
            query_features = self.mfcc_embeddings_bow[query_id]
            similarities_with_songs = []

            for j, song_id in enumerate(song_ids):
                if query_id == song_id:
                    continue
                similarity = cosine_similarity([query_features], [self.mfcc_embeddings_bow[song_id]])[0][0]
                similarities_with_songs.append((song_id, similarity))

            # Sort by similarity descending and keep only the top 100
            similarities_with_songs.sort(key=lambda x: x[1], reverse=True)
            similarities[query_id] = {song_id: similarity for song_id, similarity in similarities_with_songs[:100]}

            if i % 100 == 0:
                print(f'Processed {i} songs, merged')

        with open(self.similarity_cache_path_bow, 'wb') as file:
            pickle.dump(similarities, file)

        return similarities

    def load_or_compute_similarities_stat(self) -> dict:
        """
        Loads precomputed similarities from a file if it exists, otherwise computes and saves them.

        Returns:
            dict: A dictionary containing precomputed similarities for all song pairs.
        """
        if os.path.exists(self.similarity_cache_path_stat):
            with open(self.similarity_cache_path_stat, 'rb') as file:
                return pickle.load(file)

        similarities = {}
        song_ids = list(self.mfcc_embeddings_stat.keys())

        for i, query_id in enumerate(song_ids):
            query_features = self.mfcc_embeddings_stat[query_id]
            similarities_with_songs = []

            for j, song_id in enumerate(song_ids):
                if query_id == song_id:
                    continue
                similarity = cosine_similarity([query_features], [self.mfcc_embeddings_stat[song_id]])[0][0]
                similarities_with_songs.append((song_id, similarity))

            # Sort by similarity descending and keep only the top 100
            similarities_with_songs.sort(key=lambda x: x[1], reverse=True)
            similarities[query_id] = {song_id: similarity for song_id, similarity in similarities_with_songs[:100]}

            if i % 100 == 0:
                print(f'Processed {i} songs, merged')

        with open(self.similarity_cache_path_stat, 'wb') as file:
            pickle.dump(similarities, file)

        return similarities

    def recommend_similar_songs_merged(self, query_song, k: int = 5) -> list:
        """
        Recommends the top-k most similar songs for a given query song based on precomputed cosine similarity.

        Args:
            query_song (Song): The query song.
            k (int, optional): Number of songs to retrieve. Defaults to 5.

        Returns:
            list: A list of the top-k most similar Song objects.
        """
        query_id = query_song.song_id
        if query_id not in self.similarities_merged:
            raise ValueError(f"No precomputed similarities found for song ID: {query_id}")

        similarities_with_songs = []
        for song_id, similarity in self.similarities_merged[query_id].items():
            song = self.song_dict.get(song_id)
            if song:
                similarities_with_songs.append((song, similarity))

        # Sort by similarity descending
        similarities_with_songs.sort(key=lambda x: x[1], reverse=True)

        return [item[0] for item in similarities_with_songs[:k]]

    def recommend_similar_songs_bow(self, query_song, k: int = 5) -> list:
        """
        Recommends the top-k most similar songs for a given query song based on precomputed cosine similarity.

        Args:
            query_song (Song): The query song.
            k (int, optional): Number of songs to retrieve. Defaults to 5.

        Returns:
            list: A list of the top-k most similar Song objects.
        """
        query_id = query_song.song_id
        if query_id not in self.similarities_bow:
            raise ValueError(f"No precomputed similarities found for song ID: {query_id}")

        similarities_with_songs = []
        for song_id, similarity in self.similarities_bow[query_id].items():
            song = self.song_dict.get(song_id)
            if song:
                similarities_with_songs.append((song, similarity))

        # Sort by similarity descending
        similarities_with_songs.sort(key=lambda x: x[1], reverse=True)

        return [item[0] for item in similarities_with_songs[:k]]

    def recommend_similar_songs_stat(self, query_song, k: int = 5) -> list:
        """
        Recommends the top-k most similar songs for a given query song based on precomputed cosine similarity.

        Args:
            query_song (Song): The query song.
            k (int, optional): Number of songs to retrieve. Defaults to 5.

        Returns:
            list: A list of the top-k most similar Song objects.
        """
        query_id = query_song.song_id
        if query_id not in self.similarities_stat:
            raise ValueError(f"No precomputed similarities found for song ID: {query_id}")

        similarities_with_songs = []
        for song_id, similarity in self.similarities_stat[query_id].items():
            song = self.song_dict.get(song_id)
            if song:
                similarities_with_songs.append((song, similarity))

        # Sort by similarity descending
        similarities_with_songs.sort(key=lambda x: x[1], reverse=True)

        return [item[0] for item in similarities_with_songs[:k]]
