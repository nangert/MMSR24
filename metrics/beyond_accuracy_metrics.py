import numpy as np
from typing import List, Dict


class BeyondAccuracyMetrics:
    """
    A class to calculate beyond-accuracy metrics for information retrieval systems.
    Metrics include:
    - Coverage
    - Diversity (e.g., intra-list diversity)
    - Novelty
    - Serendipity
    - Unexpectedness
    """

    @staticmethod
    def coverage(retrieved_songs: List[Dict], total_songs: int) -> float:
        """
        Calculate the coverage of retrieved items in the recommendation list.

        Args:
            retrieved_songs (List[Dict]): List of retrieved song dictionaries.
            total_songs (int): Total number of songs in the catalog.

        Returns:
            float: Coverage value.
        """
        retrieved_ids = {song['song_id'] for song in retrieved_songs}
        return len(retrieved_ids) / total_songs

    @staticmethod
    def diversity(retrieved_songs: List[Dict]) -> float:
        """
        Calculate intra-list diversity based on genres.

        Args:
            retrieved_songs (List[Dict]): List of retrieved song dictionaries.

        Returns:
            float: Diversity value.
        """
        genres_list = [set(song['genres']) for song in retrieved_songs]
        pairwise_diversities = []

        for i, genres_1 in enumerate(genres_list):
            for j, genres_2 in enumerate(genres_list):
                if i < j:
                    jaccard_distance = 1 - len(genres_1 & genres_2) / len(genres_1 | genres_2)
                    pairwise_diversities.append(jaccard_distance)

        return np.mean(pairwise_diversities) if pairwise_diversities else 0.0

    @staticmethod
    def novelty(retrieved_songs: List[Dict], catalog_popularity: Dict[str, float]) -> float:
        """
        Calculate novelty as the average inverse popularity of retrieved items.

        Args:
            retrieved_songs (List[Dict]): List of retrieved song dictionaries.
            catalog_popularity (Dict[str, float]): Dictionary mapping song IDs to their popularity (e.g., play counts).

        Returns:
            float: Novelty value.
        """
        inversed_popularities = []
        for song in retrieved_songs:
            popularity = catalog_popularity.get(song['song_id'], 1)  # Default to 1 if not found
            if popularity > 0:  # Avoid division by zero
                inversed_popularities.append(1.0 / popularity)
            else:
                inversed_popularities.append(0.0)

        return np.mean(inversed_popularities) if inversed_popularities else 0.0

    @staticmethod
    def serendipity(retrieved_songs: List[Dict], user_history: List[Dict], catalog: List[Dict]) -> float:
        """
        Calculate serendipity based on how unexpected retrieved items are compared to the user's history.

        Args:
            retrieved_songs (List[Dict]): List of retrieved song dictionaries.
            user_history (List[Dict]): List of songs previously interacted with by the user.
            catalog (List[Dict]): Full catalog of songs.

        Returns:
            float: Serendipity value.
        """
        user_genres = {genre for song in user_history for genre in song['genres']}
        catalog_genres = {genre for song in catalog for genre in song['genres']}
        unexpectedness_scores = []

        for song in retrieved_songs:
            song_genres = set(song['genres'])
            overlap_with_user = len(song_genres & user_genres) / len(song_genres | user_genres)
            overlap_with_catalog = len(song_genres & catalog_genres) / len(song_genres | catalog_genres)

            unexpectedness = max(0, overlap_with_catalog - overlap_with_user)
            unexpectedness_scores.append(unexpectedness)

        return np.mean(unexpectedness_scores) if unexpectedness_scores else 0.0

    @staticmethod
    def unexpectedness(retrieved_songs: List[Dict], user_history: List[Dict]) -> float:
        """
        Calculate unexpectedness as the average dissimilarity between retrieved items and user history.

        Args:
            retrieved_songs (List[Dict]): List of retrieved song dictionaries.
            user_history (List[Dict]): List of songs previously interacted with by the user.

        Returns:
            float: Unexpectedness value.
        """
        user_genres = {genre for song in user_history for genre in song['genres']}
        unexpectedness_scores = []

        for song in retrieved_songs:
            song_genres = set(song['genres'])
            dissimilarity = 1 - len(song_genres & user_genres) / len(song_genres | user_genres)
            unexpectedness_scores.append(dissimilarity)

        return np.mean(unexpectedness_scores) if unexpectedness_scores else 0.0

# Example Usage
if __name__ == "__main__":
    # Sample data
    retrieved = [
        {"song_id": "1", "genres": ["pop", "dance"]},
        {"song_id": "2", "genres": ["rock", "indie"]},
        {"song_id": "3", "genres": ["classical", "instrumental"]}
    ]

    history = [
        {"song_id": "10", "genres": ["pop", "dance"]},
        {"song_id": "11", "genres": ["rock"]}
    ]

    catalog = [
        {"song_id": "1", "genres": ["pop", "dance"]},
        {"song_id": "2", "genres": ["rock", "indie"]},
        {"song_id": "3", "genres": ["classical", "instrumental"]},
        {"song_id": "4", "genres": ["jazz"]}
    ]

    popularity = {
        "1": 100,
        "2": 200,
        "3": 10,
        "4": 50
    }

    metrics = BeyondAccuracyMetrics()
    print("Coverage:", metrics.coverage(retrieved, len(catalog)))
    print("Diversity:", metrics.diversity(retrieved))
    print("Novelty:", metrics.novelty(retrieved, popularity))
    print("Serendipity:", metrics.serendipity(retrieved, history, catalog))
    print("Unexpectedness:", metrics.unexpectedness(retrieved, history))
