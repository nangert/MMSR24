# baseline_system.py
import csv
import ast
from typing import List, Dict
import numpy as np


class Song:
    """
    Represents a song with its associated metadata.
    """

    def __init__(self, song_id: str, artist: str, song_title: str, album_name: str, genres: List[str], url: str, spotify_id: str):
        """
        Initializes a Song instance.

        Args:
            song_id (str): The unique identifier for the song.
            artist (str): The artist of the song.
            song_title (str): The title of the song.
            album_name (str): The name of the album.
            genres (List[str]): A list of genres associated with the song.
        """
        self.song_id = song_id
        self.artist = artist
        self.song_title = song_title
        self.album_name = album_name
        self.genres = genres
        self.url = url
        self.spotify_id = spotify_id

    def to_dict(self) -> Dict[str, any]:
        """
        Converts the Song object into a dictionary.

        Returns:
            Dict[str, any]: A dictionary representation of the Song object.
        """
        return {
            'song_id': self.song_id,
            'artist': self.artist,
            'song_title': self.song_title,
            'album_name': self.album_name,
            'genres': self.genres,
            'url': self.url,
            'spotify_id': self.spotify_id,
        }


class Dataset:
    """
    Loads and stores song data from TSV files.
    """

    def __init__(self, info_file_path: str, genres_file_path: str, url_dataset_path: str, metadata_dataset_path: str, bert_embeddings_path: str):
        """
        Initializes the Dataset by loading song information, genres, and BERT embeddings.

        Args:
            info_file_path (str): Path to the TSV file containing song information.
            genres_file_path (str): Path to the TSV file containing song genres.
            url_dataset_path (str): Path to the TSV file containing song URLs.
            metadata_dataset_path (str): Path to the TSV file containing song metadata (e.g., Spotify IDs).
            bert_embeddings_path (str): Path to the TSV file containing precomputed BERT embeddings.
        """
        self.songs = []
        self.bert_embeddings = {}
        self.load_dataset(info_file_path, genres_file_path, url_dataset_path, metadata_dataset_path, bert_embeddings_path)

    def load_dataset(self, info_file_path: str, genres_file_path: str, url_dataset_path: str, metadata_dataset_path: str, bert_embeddings_path: str):
        # Load genres, URLs, metadata using a unified helper
        genres_dict = self._load_dict_from_tsv(
            file_path=genres_file_path,
            key_col='id',
            value_col='genre',
            transform=lambda val: ast.literal_eval(val)
        )

        url_dict = self._load_dict_from_tsv(
            file_path=url_dataset_path,
            key_col='id',
            value_col='url'
        )

        metadata_dict = self._load_dict_from_tsv(
            file_path=metadata_dataset_path,
            key_col='id',
            value_col='spotify_id'
        )

        # Load BERT embeddings separately due to different format
        self.bert_embeddings = {}
        with open(bert_embeddings_path, 'r', encoding='utf-8') as f:
            header = f.readline()  # Skip header
            for line in f:
                parts = line.strip().split('\t')
                track_id = parts[0]
                vector_values = list(map(float, parts[1:]))
                self.bert_embeddings[track_id] = np.array(vector_values, dtype=np.float32)

        # Load main song information
        self.songs = self._load_song_info(info_file_path, genres_dict, url_dict, metadata_dict)

    @staticmethod
    def _load_dict_from_tsv(file_path: str, key_col: str, value_col: str, transform=None) -> Dict[str, any]:
        """
        A generic helper to load a TSV file into a dictionary.

        Args:
            file_path (str): Path to the TSV file.
            key_col (str): The name of the column to use as dictionary keys.
            value_col (str): The name of the column to use as dictionary values.
            transform (callable, optional): A function to transform the value before storing.

        Returns:
            Dict[str, any]: A dictionary mapping from key_col values to transformed value_col values.
        """
        result_dict = {}
        with open(file_path, 'r', encoding='utf-8') as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t')
            for row in reader:
                key = row[key_col]
                val = row[value_col]
                if transform:
                    val = transform(val)
                result_dict[key] = val
        return result_dict

    @staticmethod
    def _load_song_info(info_file_path: str, genres_dict: Dict[str, List[str]], url_dict: Dict[str, str], metadata_dict: Dict[str, str]) -> List[Song]:
        songs = []
        with open(info_file_path, 'r', encoding='utf-8') as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t')
            for row in reader:
                song_id = row['id']
                song = Song(
                    song_id=song_id,
                    artist=row.get('artist', ''),
                    song_title=row.get('song', ''),
                    album_name=row.get('album_name', ''),
                    genres=genres_dict.get(song_id, []),
                    url=url_dict.get(song_id, ''),
                    spotify_id=metadata_dict.get(song_id, '')
                )
                songs.append(song)
        return songs

    def get_all_songs(self) -> List[Song]:
        """
        Retrieves all songs in the dataset.

        Returns:
            List[Song]: A list of all Song objects.
        """
        return self.songs

    @staticmethod
    def load_genre_weights(tags_path: str, genres_path: str) -> Dict[str, set]:
        """
        Match genres with corresponding weights for each song.

        Args:
            tags_path (str): Path to id_tags_dict.tsv (all tags with weights).
            genres_path (str): Path to id_genres_mmsr.tsv (filtered genres).

        Returns:
            Dict[str, Dict[str, int]]: A mapping of song IDs to genres and weights.
        """
        tag_weights = {}
        with open(tags_path, 'r') as tags_file:
            next(tags_file)  # Skip header
            for line in tags_file:
                song_id, tag_weight_str = line.strip().split('\t')
                # Convert the string representation of the dictionary to a real dictionary
                tag_weights[song_id] = ast.literal_eval(tag_weight_str)

        top_genre_weights = {}
        # Parse genres and find top-weight genres for each song
        with open(genres_path, 'r') as genres_file:
            next(genres_file)  # Skip header
            for line in genres_file:
                song_id, genres_str = line.strip().split('\t')
                genres_list = ast.literal_eval(genres_str)  # Convert string to list of genres

                if song_id in tag_weights:
                    # Filter weights to only include those in the genres list
                    relevant_weights = {genre: tag_weights[song_id][genre] for genre in genres_list if genre in tag_weights[song_id]}

                    if relevant_weights:
                        # Find the maximum weight and filter genres with that weight
                        max_weight = max(relevant_weights.values())
                        top_genre_weights[song_id] = {genre for genre, weight in relevant_weights.items() if weight == max_weight}

        return top_genre_weights

    @staticmethod
    def get_total_relevant(query_song: dict, top_genre_weights: Dict[str, set]) -> int:
        """
        Determine the total number of relevant items for a query song.

        Args:
            query_song (dict): The query song.
            top_genre_weights (Dict[str, set]): Mapping of song IDs to their top genres.

        Returns:
            int: Total number of relevant items.
        """
        query_song_id = query_song.get('song_id', '')
        query_genres = top_genre_weights.get(query_song_id, set())  # Get top genres for the query song

        if not query_genres:
            return 0  # No relevant genres, return 0

        total_relevant = 0
        for song_id, genres in top_genre_weights.items():
            if song_id != query_song_id and query_genres & genres:  # Intersection of top genres
                total_relevant += 1

        return total_relevant