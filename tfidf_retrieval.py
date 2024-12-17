import os
import pandas as pd
import numpy as np
from typing import List, Dict
from accuracy_metrics import Metrics

class TFIDFRetrievalSystem:
    def __init__(self, tfidf_file_path: str, relevance_labels_path: str):
        self.tfidf_embeddings = {}
        self.relevance_data = {}
        self.load_tfidf_embeddings(tfidf_file_path)
        self.load_relevance_data(relevance_labels_path)

    def load_tfidf_embeddings(self, tfidf_file_path: str):
        """Load TF-IDF embeddings."""
        df_tfidf = pd.read_csv(tfidf_file_path, sep='\t')
        for _, row in df_tfidf.iterrows():
            song_id = row['id']
            vector = row.iloc[1:].values.astype(np.float32)
            self.tfidf_embeddings[song_id] = vector / np.linalg.norm(vector)

    def load_relevance_data(self, relevance_labels_path: str):
        """Load relevance labels."""
        df = pd.read_csv(relevance_labels_path, sep='\t')
        for _, row in df.iterrows():
            self.relevance_data[row['query_id']] = row['relevant_songs'].split(',')

    def retrieve(self, query_id: str, N: int) -> List[str]:
        """Retrieve top N similar songs."""
        if query_id not in self.tfidf_embeddings:
            return []
        query_vec = self.tfidf_embeddings[query_id]
        similarities = []

        for song_id, vector in self.tfidf_embeddings.items():
            if song_id != query_id:
                sim = np.dot(query_vec, vector) / (np.linalg.norm(vector))
                similarities.append((song_id, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [song_id for song_id, _ in similarities[:N]]

    def calculate_metrics(self, query_id: str, N: int) -> Dict[str, float]:
        """Calculate MRR, Precision@k, Recall@k, and NDCG@k."""
        metrics = Metrics()
        retrieved = self.retrieve(query_id, N)
        relevant = set(self.relevance_data.get(query_id, []))
        result_songs = [{'song_id': song_id} for song_id in retrieved]

        result = metrics.calculate_metrics({'song_id': query_id}, result_songs, len(relevant), relevant, N)
        return result
