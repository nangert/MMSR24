import json
import pandas as pd
from typing import List
import numpy as np
from sklearn.cluster import KMeans

class SongDiversityOptimizer:
    def __init__(self, filepath):
        """
        Initialize the optimizer with tag data from a file.
        :param filepath: Path to the dataset file (e.g., ./dataset/id_tags_dict.tsv)
        """
        self.song_tags = self.load_tags(filepath)

        self.all_tags = set()
        for tags_dict in self.song_tags.values():
            self.all_tags.update(tags_dict.keys())
        self.all_tags = sorted(list(self.all_tags))

    def load_tags(self, filepath: str) -> dict:
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
                # Replace single quotes with double quotes to parse JSON
                tags = json.loads(row['tags'].replace("'", '"'))
                song_tags[song_id] = tags
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON for song_id {song_id}: {e}")
                continue
        return song_tags

    def get_tag_dict(self, song) -> dict:
        """
        Given a song object, return its tag->weight dictionary.
        We assume 'song.song_id' is the key in self.song_tags.
        Adjust as needed if your 'song' structure differs.
        """
        return self.song_tags.get(song.song_id, {})

    def weighted_jaccard_similarity(self, song_a, song_b) -> float:
        """
        Calculate weighted Jaccard similarity between two songs, based on their tag dictionaries.
        """
        tags_a = self.get_tag_dict(song_a)
        tags_b = self.get_tag_dict(song_b)

        if not tags_a and not tags_b:
            return 0.0

        # Sum of min(...) of the two vectors
        sum_min = 0.0
        # Sum of max(...) of the two vectors
        sum_max = 0.0

        # Get the union of all tags that appear in either song
        all_tags = set(tags_a.keys()) | set(tags_b.keys())

        for tag in all_tags:
            w_a = tags_a.get(tag, 0)
            w_b = tags_b.get(tag, 0)
            sum_min += min(w_a, w_b)
            sum_max += max(w_a, w_b)

        if sum_max == 0:
            return 0.0

        return sum_min / sum_max

    def distance(self, song_a, song_b) -> float:
        """
        Define a distance measure as 1 - weighted_jaccard_similarity.
        """
        return 1.0 - self.weighted_jaccard_similarity(song_a, song_b)

    def calculate_diversity_score(self, songs: List) -> float:
        """
        Calculate the total pairwise distance among all songs in the list.
        The higher the score, the more diverse the set.
        """

        total_distance = 0.0
        for i in range(len(songs)):
            for j in range(i + 1, len(songs)):
                total_distance += self.distance(songs[i], songs[j])
        return total_distance

    def greedy_optimize_diversity(self, retrieved_songs: List, n: int) -> List:
        """
        From the larger list 'retrieved_songs' of length adapted_n,
        pick 'n' songs that greedily maximize pairwise diversity.
        """
        if n >= len(retrieved_songs):
            # If we already have fewer than or equal to n songs, just return them
            return retrieved_songs

        # Convert the list to a mutable set of candidates
        candidates = set(retrieved_songs)
        chosen = []

        while len(chosen) < n and candidates:
            best_song = None
            best_increase = -1.0

            for song in candidates:
                # Temporarily evaluate adding 'song' to chosen
                new_diversity = self.calculate_diversity_score(chosen + [song])
                # Current diversity of chosen
                current_diversity = self.calculate_diversity_score(chosen)
                increase = new_diversity - current_diversity

                if increase > best_increase:
                    best_increase = increase
                    best_song = song

            # Move the best candidate from the pool into chosen set
            chosen.append(best_song)
            candidates.remove(best_song)

        return chosen

    def semi_greedy_optimize_diversity(self, retrieved_songs: List, n: int) -> List:
        """
        From the ordered list 'retrieved_songs' of length adapted_n,
        pick 'n' songs by alternating:
          - one pick from the top of the ordered list
          - one greedy pick that maximizes diversity
        """
        # If retrieved_songs is shorter or equal to n, just return all
        if n >= len(retrieved_songs):
            return retrieved_songs

        # We'll treat `retrieved_songs` as a queue we pop from the front
        # for the "top item" picks, but for the "greedy" picks, we need to
        # search among the remaining items.
        candidates = list(retrieved_songs)  # ensure it's mutable
        chosen = []

        while len(chosen) < n and candidates:
            # Step 1: Pick from the top of the ordered list
            # (i.e., pop the first element in the candidate list)
            top_song = candidates.pop(0)
            chosen.append(top_song)
            if len(chosen) == n:
                break

            # Step 2: If there's space left, do a greedy diversity pick
            if candidates and len(chosen) < n:
                best_song = None
                best_increase = -1.0
                current_diversity = self.calculate_diversity_score(chosen)

                for song in candidates:
                    new_diversity = self.calculate_diversity_score(chosen + [song])
                    increase = new_diversity - current_diversity
                    if increase > best_increase:
                        best_increase = increase
                        best_song = song

                if best_song:
                    chosen.append(best_song)
                    candidates.remove(best_song)

        return chosen

    def get_song_tag_vector(self, song) -> np.ndarray:
        """
        Convert a song's tags into a vector in the global tag space.
        If a song has no tags or missing tags, those dimensions remain 0.
        """
        vector = np.zeros(len(self.all_tags))
        if song.song_id in self.song_tags:
            tag_dict = self.song_tags[song.song_id]
            for i, tag_name in enumerate(self.all_tags):
                if tag_name in tag_dict:
                    vector[i] = tag_dict[tag_name]
        return vector

    def cluster_optimize_diversity_tags(self, retrieved_songs: List, n: int) -> List:
        """
        1) Takes a list of candidate songs (already retrieved by relevance).
        2) Clusters them into n clusters based on their tag vectors.
        3) Returns exactly n songs (one from each cluster).
        """
        if n >= len(retrieved_songs):
            return retrieved_songs

        # Build a matrix X for k-means
        X = []
        for s in retrieved_songs:
            X.append(self.get_song_tag_vector(s))
        X = np.array(X)

        n_clusters = min(n, len(retrieved_songs))
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        kmeans.fit(X)
        labels = kmeans.labels_

        # Group
        clusters = {i: [] for i in range(n_clusters)}
        for idx, label in enumerate(labels):
            clusters[label].append(retrieved_songs[idx])

        chosen = []
        # Because retrieved_songs are presumably sorted by relevance,
        # we pick the earliest item in each cluster as "best" from that cluster.
        # Alternatively, if you had a 'relevance_score', you'd pick max.
        for cluster_idx in range(n_clusters):
            songs_in_cluster = clusters[cluster_idx]
            if songs_in_cluster:
                # pick the earliest in the original list
                best_song = None
                best_index = float('inf')
                for s in songs_in_cluster:
                    idx_in_original = retrieved_songs.index(s)
                    if idx_in_original < best_index:
                        best_song = s
                        best_index = idx_in_original
                chosen.append(best_song)

        # If for some reason we have fewer than n (empty cluster?), fill up
        if len(chosen) < n:
            leftover = [s for s in retrieved_songs if s not in chosen]
            chosen += leftover[: n - len(chosen)]

        return chosen