import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import random


class MFCCRetrievalSystem:
    """
    A retrieval system that uses MFCC features for similarity-based retrieval.
    """

    def __init__(self, bow_path: str, stats_path: str, dataset):
        """
        Initializes the retrieval system by loading MFCC feature files and merging them into a single DataFrame.

        Args:
            bow_path (str): Path to the MFCC BoW file.
            stats_path (str): Path to the MFCC stats file.
            dataset: The dataset containing song metadata.
        """
        self.df_bow = self.load_tsv(bow_path)
        self.df_stats = self.load_tsv(stats_path)
        self.df = pd.merge(self.df_bow, self.df_stats, on='id')
        self.dataset = dataset
        self.normalize_features()
        self.song_dict = {s.song_id: s for s in self.dataset.get_all_songs()}

    @staticmethod
    def load_tsv(file_path: str) -> pd.DataFrame:
        """
        Loads a TSV file into a Pandas DataFrame.

        Args:
            file_path (str): Path to the TSV file.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        return pd.read_csv(file_path, sep='\t')

    def normalize_features(self) -> None:
        """
        Normalizes the features of the merged DataFrame using MinMaxScaler.
        """
        scaler = MinMaxScaler()
        self.df.iloc[:, 1:] = scaler.fit_transform(self.df.iloc[:, 1:])

    def recommend_similar_songs(self, query_song, k: int = 5) -> list:
        """
        Recommends the top-k most similar songs for a given song ID based on cosine similarity.

        Args:
            query_song (Song): The song used as the query.
            k (int, optional): The number of songs to retrieve. Defaults to 5.

        Returns:
            list: A list of the top-k most similar songs.
        """
        query_id = query_song.song_id
        query_row = self.df[self.df['id'] == query_id]
        if query_row.empty:
            raise ValueError(f"No song found for song ID: {query_id}")

        query_features = query_row.iloc[:, 1:].values

        song_features = self.df.iloc[:, 1:].values
        similarities = cosine_similarity(query_features, song_features)[0]

        similarities_with_songs = []
        for idx, song_row in self.df.iterrows():
            if song_row['id'] == query_id:
                continue
            similarity = similarities[idx]
            # Retrieve song using the pre-built dictionary
            song = self.song_dict.get(song_row['id'])
            if song:
                similarities_with_songs.append((song, similarity))

        # Sort by similarity
        similarities_with_songs.sort(key=lambda x: x[1], reverse=True)

        # Return top N songs
        return [item[0] for item in similarities_with_songs[:k]]

    def generate_retrieval_results(self, N: int) -> dict:
        """
        Generates retrieval results for all songs in the dataset.

        Args:
            N (int): The number of songs to retrieve for each query.

        Returns:
            dict: A dictionary where each key is a query song ID and the value contains the query and retrieved songs.
        """
        retrieval_results = {}
        for query_song in self.dataset.get_all_songs():
            retrieved_songs = self.recommend_similar_songs(query_song, N)
            retrieval_results[query_song.song_id] = {
                'query': query_song.to_dict(),
                'retrieved': [song.to_dict() for song in retrieved_songs]
            }
        return retrieval_results