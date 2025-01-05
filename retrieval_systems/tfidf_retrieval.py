import pandas as pd
import numpy as np
from typing import List

class TFIDFRetrievalSystem:
    def __init__(self, dataset, tfidf_file_path: str):
        self.tfidf_embeddings = {}
        self.relevance_data = {}
        self.load_tfidf_embeddings(tfidf_file_path)
        self.song_dict = {s.song_id: s for s in dataset.get_all_songs()}

    def load_tfidf_embeddings(self, tfidf_file_path: str):
        """Load TF-IDF embeddings."""
        df_tfidf = pd.read_csv(tfidf_file_path, sep='\t')
        for _, row in df_tfidf.iterrows():
            song_id = row['id']
            vector = row.iloc[1:].values.astype(np.float32)
            self.tfidf_embeddings[song_id] = vector / np.linalg.norm(vector)

    def retrieve(self, query_id: str, N: int) -> list:
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

        top_similarities = similarities[:N]
        top_songs = []
        for song_id, sim in top_similarities:
            song = self.song_dict.get(song_id)
            if song:  # Only add if the song exists in the dictionary
                top_songs.append((song, sim))

        return [item[0] for item in top_songs[:N]]

