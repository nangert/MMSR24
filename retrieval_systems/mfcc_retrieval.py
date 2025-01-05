import os
import pickle
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import mahalanobis
import numpy as np
from scipy.spatial.distance import cosine


class MFCCRetrievalSystem:
    """
    A retrieval system that uses MFCC features for similarity-based retrieval.
    """

    def __init__(self, dataset, similarity_cache_path_bow='precomputed/mfcc_similarities_bow.pkl', similarity_cache_path_stat='precomputed/mfcc_similarities_stat.pkl', similarity_cache_path_bow_cos='precomputed/mfcc_similarities_bow_cos.pkl', similarity_cache_path_stat_cos='precomputed/mfcc_similarities_stat_cos.pkl'):
        """
        Initializes the retrieval system using the dataset's preprocessed MFCC embeddings.

        Args:
            dataset: The dataset containing song metadata and MFCC embeddings.
            similarity_cache_path (str): Path to the file where precomputed similarities will be stored.
        """
        self.mfcc_embeddings_bow = dataset.mfcc_embeddings_bow
        self.mfcc_embeddings_stat = dataset.mfcc_embeddings_stat
        self.song_dict = {s.song_id: s for s in dataset.get_all_songs()}
        self.similarity_cache_path_bow = similarity_cache_path_bow
        self.similarity_cache_path_bow_cos = similarity_cache_path_bow_cos
        self.similarity_cache_path_stat = similarity_cache_path_stat
        self.similarity_cache_path_stat_cos = similarity_cache_path_stat_cos
        self.similarities_bow = self.load_similarities_bow()
        self.similarities_bow_cos = self.load_similarities_bow_cos()
        self.similarities_stat = self.load_similarities_stat()
        self.similarities_stat_cos = self.load_similarities_stat_cos()

    def load_similarities_bow(self) -> dict:
        if os.path.exists(self.similarity_cache_path_bow):
            with open(self.similarity_cache_path_bow, 'rb') as file:
                return pickle.load(file)
        return {}

    def load_similarities_bow_cos(self) -> dict:
        if os.path.exists(self.similarity_cache_path_bow_cos):
            with open(self.similarity_cache_path_bow_cos, 'rb') as file:
                return pickle.load(file)
        return {}

    def load_similarities_stat(self) -> dict:
        if os.path.exists(self.similarity_cache_path_stat):
            with open(self.similarity_cache_path_stat, 'rb') as file:
                return pickle.load(file)
        return {}

    def load_similarities_stat_cos(self) -> dict:
        if os.path.exists(self.similarity_cache_path_stat_cos):
            with open(self.similarity_cache_path_stat_cos, 'rb') as file:
                return pickle.load(file)
        return {}

    def compute_bow_similarity(self, query_id, song_id) -> float:
        query_features = self.mfcc_embeddings_bow[query_id]
        target_features = self.mfcc_embeddings_bow[song_id]
        return -wasserstein_distance(query_features, target_features)

    def compute_stat_similarity(self, query_id, song_id) -> float:
        def reconstruct_covariance_matrix(cov_data):
            n = 13
            cov_matrix = np.zeros((n, n))
            upper_tri_indices = np.triu_indices(n)
            cov_matrix[upper_tri_indices] = cov_data
            cov_matrix[(upper_tri_indices[1], upper_tri_indices[0])] = cov_data
            return cov_matrix

        query_features = self.mfcc_embeddings_stat[query_id][:13]
        target_features = self.mfcc_embeddings_stat[song_id][:13]
        query_cov = reconstruct_covariance_matrix(self.mfcc_embeddings_stat[query_id][13:])
        target_cov = reconstruct_covariance_matrix(self.mfcc_embeddings_stat[song_id][13:])
        try:
            inv_cov = np.linalg.inv((query_cov + target_cov) / 2)
            distance = mahalanobis(query_features, target_features, inv_cov)
            return -distance
        except np.linalg.LinAlgError:
            return float('-inf')

    def compute_recommendations_bow(self, query_song, k: int = 5) -> list:
        query_id = query_song.song_id
        song_ids = self.mfcc_embeddings_bow.keys()
        similarities = {
            song_id: self.compute_bow_similarity(query_id, song_id)
            for song_id in song_ids if song_id != query_id
        }
        sorted_songs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]
        return [self.song_dict[song_id] for song_id, _ in sorted_songs]

    def compute_recommendations_bow_cos(self, query_song, k: int = 5) -> list:
        query_id = query_song.song_id
        query_features = self.mfcc_embeddings_bow[query_id]
        song_ids = self.mfcc_embeddings_bow.keys()

        similarities = {
            song_id: 1 - cosine(query_features, self.mfcc_embeddings_bow[song_id])
            for song_id in song_ids if song_id != query_id
        }

        # Sort by similarity descending
        sorted_songs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]

        return [self.song_dict[song_id] for song_id, _ in sorted_songs]

    def compute_recommendations_stat(self, query_song, k: int = 5) -> list:
        query_id = query_song.song_id
        song_ids = self.mfcc_embeddings_stat.keys()
        similarities = {
            song_id: self.compute_stat_similarity(query_id, song_id)
            for song_id in song_ids if song_id != query_id
        }
        sorted_songs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]
        return [self.song_dict[song_id] for song_id, _ in sorted_songs]

    def compute_recommendations_stat_cos(self, query_song, k: int = 5) -> list:
        query_id = query_song.song_id
        query_features = self.mfcc_embeddings_stat[query_id][:13]  # Use mean features for cosine similarity
        song_ids = self.mfcc_embeddings_stat.keys()

        similarities = {
            song_id: 1 - cosine(query_features, self.mfcc_embeddings_stat[song_id][:13])
            for song_id in song_ids if song_id != query_id
        }

        # Sort by similarity descending
        sorted_songs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]

        return [self.song_dict[song_id] for song_id, _ in sorted_songs]

    def recommend_similar_songs_bow(self, query_song, k: int = 5) -> list:
        query_id = query_song.song_id
        if query_id in self.similarities_bow:
            similarities = self.similarities_bow[query_id]
        else:
            return self.compute_recommendations_bow(query_song, k)
        sorted_songs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]
        return [self.song_dict[song_id] for song_id, _ in sorted_songs]

    def recommend_similar_songs_bow_cos(self, query_song, k: int = 5) -> list:
        query_id = query_song.song_id
        if query_id in self.similarities_bow_cos:
            similarities = self.similarities_bow_cos[query_id]
        else:
            return self.compute_recommendations_bow_cos(query_song, k)
        sorted_songs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]
        return [self.song_dict[song_id] for song_id, _ in sorted_songs]

    def recommend_similar_songs_stat(self, query_song, k: int = 5) -> list:
        query_id = query_song.song_id
        if query_id in self.similarities_stat:
            similarities = self.similarities_stat[query_id]
        else:
            return self.compute_recommendations_stat(query_song, k)
        sorted_songs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]
        return [self.song_dict[song_id] for song_id, _ in sorted_songs]

    def recommend_similar_songs_stat_cos(self, query_song, k: int = 5) -> list:
        query_id = query_song.song_id
        if query_id in self.similarities_stat_cos:
            similarities = self.similarities_stat_cos[query_id]
        else:
            return self.compute_recommendations_stat_cos(query_song, k)
        sorted_songs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]
        return [self.song_dict[song_id] for song_id, _ in sorted_songs]
