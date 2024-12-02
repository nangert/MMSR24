# baseline_system.py

import random
import csv
import json
import ast
from typing import List, Dict


class Song:
    """
    Represents a song with its associated metadata.
    """

    def __init__(self, song_id: str, artist: str, song_title: str, album_name: str, genres: List[str]):
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
            'genres': self.genres
        }


class Dataset:
    """
    Loads and stores song data from TSV files.
    """

    def __init__(self, info_file_path: str, genres_file_path: str):
        """
        Initializes the Dataset by loading song information and genres.

        Args:
            info_file_path (str): Path to the TSV file containing song information.
            genres_file_path (str): Path to the TSV file containing song genres.
        """
        self.songs = self.load_dataset(info_file_path, genres_file_path)

    def load_dataset(self, info_file_path: str, genres_file_path: str) -> List[Song]:
        """
        Loads the dataset from TSV files and merges song information with genres.

        Args:
            info_file_path (str): Path to the song information TSV file.
            genres_file_path (str): Path to the song genres TSV file.

        Returns:
            List[Song]: A list of Song objects with merged information.
        """
        # Load genres data into a dictionary
        genres_dict: Dict[str, List[str]] = {}
        with open(genres_file_path, 'r', encoding='utf-8') as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t')
            for row in reader:
                song_id = row['id']
                genres_str = row['genre']
                # Parse the genre string into a list
                genres_list = ast.literal_eval(genres_str)
                genres_dict[song_id] = genres_list

        # Load song information and merge with genres
        songs: List[Song] = []
        with open(info_file_path, 'r', encoding='utf-8') as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t')
            for row in reader:
                song_id = row['id']
                artist = row['artist']
                song_title = row['song']
                album_name = row['album_name']
                genres = genres_dict.get(song_id, [])  # Default to empty list if not found

                song = Song(
                    song_id=song_id,
                    artist=artist,
                    song_title=song_title,
                    album_name=album_name,
                    genres=genres
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


class RetrievalSystem:
    """
    A simple retrieval system that randomly selects songs from the dataset.
    """

    def __init__(self, dataset: Dataset):
        """
        Initializes the RetrievalSystem with a dataset.

        Args:
            dataset (Dataset): The dataset to use for retrieval.
        """
        self.dataset = dataset

    def get_random_retrieval(self, query_song: Song, N: int) -> List[Song]:
        """
        Retrieves N random songs from the dataset excluding the query song.

        Args:
            query_song (Song): The song used as the query.
            N (int): The number of songs to retrieve.

        Returns:
            List[Song]: A list of N randomly selected Song objects.
        """
        possible_songs = [song for song in self.dataset.get_all_songs() if song.song_id != query_song.song_id]
        return random.sample(possible_songs, N)

    def generate_retrieval_results(self, N: int) -> Dict[str, Dict[str, any]]:
        """
        Generates random retrieval results for all songs in the dataset.

        Args:
            N (int): The number of songs to retrieve for each query.

        Returns:
            Dict[str, Dict[str, any]]: A dictionary where each key is a query song ID and the value contains the query and retrieved songs.
        """
        retrieval_results: Dict[str, Dict[str, any]] = {}
        for query_song in self.dataset.get_all_songs():
            retrieved_songs = self.get_random_retrieval(query_song, N)
            retrieval_results[query_song.song_id] = {
                'query': query_song.to_dict(),
                'retrieved': [song.to_dict() for song in retrieved_songs]
            }
        return retrieval_results


def main():
    """
    Main function to execute the baseline retrieval system.
    """
    # Parameters
    info_dataset_path = 'dataset/id_information_mmsr.tsv'  # Path to your song information TSV file
    genres_dataset_path = 'dataset/id_genres_mmsr.tsv'     # Path to your genres TSV file
    retrieval_results_path = 'results/retrieval_results.json'
    N = 10  # Number of songs to retrieve

    # Load dataset
    dataset = Dataset(info_dataset_path, genres_dataset_path)

    # Initialize retrieval system
    retrieval_system = RetrievalSystem(dataset)

    # Generate retrieval results
    retrieval_results = retrieval_system.generate_retrieval_results(N)

    # Save retrieval results
    with open(retrieval_results_path, 'w', encoding='utf-8') as f:
        json.dump(retrieval_results, f, ensure_ascii=False, indent=4)

    print(f"Retrieval results saved to {retrieval_results_path}")

if __name__ == '__main__':
    main()
