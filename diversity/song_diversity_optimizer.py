import json
import pandas as pd
from collections import defaultdict
from typing import List

class SongDiversityOptimizer:
    def __init__(self, filepath):
        """
        Initialize the optimizer with tag data from a file.
        :param filepath: Path to the dataset file (e.g., ./dataset/id_tags_dict.tsv)
        """
        self.song_tags = self.load_tags(filepath)

    def load_tags(self, filepath):
        """
        Load the tag data from a TSV file into a dictionary.
        :param filepath: Path to the dataset file
        :return: Dictionary where keys are song IDs and values are dictionaries of tags and weights
        """
        song_tags = {}
        df = pd.read_csv(filepath, sep='\t', header=None, names=['id', 'tags'])
        for _, row in df.iterrows():
            song_id = row['id']
            if pd.isna(row['tags']) or not row['tags'].strip():
                continue  # Skip rows with missing or empty tags
            try:
                tags = json.loads(row['tags'].replace("'", '"'))  # Replace single quotes with double quotes
                song_tags[song_id] = tags
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON for song_id {song_id}: {e}")
                continue
        return song_tags

    def greedy_optimize_diversity(self, songs: List, n: int):
        """
        Optimize diversity by greedily selecting songs with minimal tag overlap.
        :param songs: List of Song objects to consider for diversity optimization
        :param n: Number of songs to select
        :return: List of selected Song objects
        """
        if len(songs) <= n:
            return songs  # If fewer songs are retrieved than required, return all of them

        selected_songs = []
        selected_tags = defaultdict(int)

        for _ in range(n):
            best_song = None
            best_diversity_score = float('-inf')

            for song in songs:
                if song in selected_songs:
                    continue

                tags = self.song_tags.get(song.song_id, {})
                diversity_score = sum((1 / (1 + selected_tags[tag])) * weight for tag, weight in tags.items())

                if diversity_score > best_diversity_score:
                    best_diversity_score = diversity_score
                    best_song = song

            if best_song:
                selected_songs.append(best_song)
                songs.remove(best_song)  # Remove the selected song from the pool
                for tag, weight in self.song_tags.get(best_song.song_id, {}).items():
                    selected_tags[tag] += weight

        return selected_songs

    def calculate_diversity_score(self, songs: List['Song'], n: int = None) -> float:
        """
        Calculate the diversity score for a given list of songs.
        :param songs: List of Song objects
        :param n: Number of songs to consider (optional, defaults to the full list)
        :return: A float representing the diversity score of the list
        """
        if not songs:
            return 0.0

        if n is not None:
            songs = songs[:n]  # Limit the list to the first n songs

        tag_counts = defaultdict(int)
        total_score = 0.0

        for song in songs:
            tags = self.song_tags.get(song.song_id, {})
            for tag, weight in tags.items():
                tag_counts[tag] += weight

        for tag, count in tag_counts.items():
            total_score += 1 / (1 + count)

        return total_score
