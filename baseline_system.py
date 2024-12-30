# baseline_system.py

import random
import csv
import json
import ast
from typing import List, Dict

from Music4All import Dataset, Song


class BaselineRetrievalSystem:
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

    def get_retrieval(self, query_song: Song, N: int) -> List[Song]:
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
            retrieved_songs = self.get_retrieval(query_song, N)
            retrieval_results[query_song.song_id] = {
                'query': query_song.to_dict(),
                'retrieved': [song.to_dict() for song in retrieved_songs]
            }
        return retrieval_results


def main():
    """
    Main function to test the baseline retrieval system.
    """
    """
    # Parameters
    info_dataset_path = 'dataset/id_information_mmsr.tsv'  # Path to your song information TSV file
    genres_dataset_path = 'dataset/id_genres_mmsr.tsv'     # Path to your genres TSV file
    retrieval_results_path = 'results/retrieval_results.json'
    N = 10  # Number of songs to retrieve

    # Load dataset
    dataset = Dataset(info_dataset_path, genres_dataset_path)

    # Initialize retrieval system
    retrieval_system = BaselineRetrievalSystem(dataset)

    # Generate retrieval results
    retrieval_results = retrieval_system.generate_retrieval_results(N)

    # Save retrieval results
    with open(retrieval_results_path, 'w', encoding='utf-8') as f:
        json.dump(retrieval_results, f, ensure_ascii=False, indent=4)

    print(f"Retrieval results saved to {retrieval_results_path}")
    """


if __name__ == '__main__':
    main()
