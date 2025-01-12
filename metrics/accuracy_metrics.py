import json

import numpy as np
from sklearn.metrics import ndcg_score
from typing import List, Dict

class Metrics:
    """
    A class containing static methods to compute various accuracy metrics.
    """

    @staticmethod
    def precision_at_k(relevant_labels: np.ndarray, k: int) -> float:
        """
        Compute Precision@K

        Args:
            relevant_labels (np.ndarray): An array of binary relevance labels (1 for relevant, 0 for not relevant).
            k (int): The rank position up to which to calculate the metric.

        Returns:
            float: The Precision@K value.
        """
        relevant_at_k = relevant_labels[:k]
        precision = np.sum(relevant_at_k) / k
        return precision

    @staticmethod
    def recall_at_k(relevant_labels: np.ndarray, total_relevant: int, k: int) -> float:
        """
        Compute Recall@K

        Args:
            relevant_labels (np.ndarray): An array of binary relevance labels (1 for relevant, 0 for not relevant).
            total_relevant (int): Total number of relevant items in the dataset.
            k (int): The rank position up to which to calculate the metric.

        Returns:
            float: The Recall@K value.
        """
        if total_relevant == 0:
            return 0.0
        relevant_at_k = relevant_labels[:k]
        recall = np.sum(relevant_at_k) / total_relevant
        return recall

    @staticmethod
    def ndcg_at_k(relevant_labels: np.ndarray, k: int) -> float:
        """
        Compute NDCG@K using sklearn's ndcg_score.

        Args:
            relevant_labels (np.ndarray): An array of binary relevance labels (1 for relevant, 0 for not relevant).
            k (int): The rank position up to which to calculate the metric.

        Returns:
            float: The NDCG@K value.
        """
        predicted_scores = np.arange(len(relevant_labels), 0, -1)
        ndcg = ndcg_score([relevant_labels], [predicted_scores], k=k)
        return ndcg

    @staticmethod
    def mrr(relevant_labels: np.ndarray) -> float:
        """
        Compute Mean Reciprocal Rank (MRR).

        Args:
            relevant_labels (np.ndarray): An array of binary relevance labels (1 for relevant, 0 for not relevant).

        Returns:
            float: The MRR value.
        """
        relevant_indices = np.where(relevant_labels == 1)[0]
        if len(relevant_indices) == 0:
            return 0.0
        else:
            mrr = 1.0 / (relevant_indices[0] + 1)  # +1 because indices start from 0
            return mrr

    def calculate_metrics(self, query_song: Dict, result_songs: List[Dict], total_relevant: int, query_genres: set, k: int) -> Dict[str, float]:
        """
        Calculate the retrieval metrics for a query song and a list of result songs.

        Args:
            query_song (Dict): The query song metadata.
            result_songs (List[Dict]): The list of retrieved songs metadata.
            total_relevant (int): Total number of relevant items in the dataset.
            query_genres (set): The set of genres for the query song.
            k (int): The rank position up to which to calculate the metrics.

        Returns:
            Dict[str, float]: A dictionary containing the computed metrics.
        """

        # Create a binary relevance list for the result songs
        relevant_labels: List[int] = []
        for song in result_songs:
            song_genres = set(song.get('genres', []))

            is_relevant = int(len(query_genres & song_genres) / len(query_genres | song_genres) > 0.1)
            relevant_labels.append(is_relevant)

        relevant_labels_array = np.array(relevant_labels)

        precision = self.precision_at_k(relevant_labels_array, k)
        recall = self.recall_at_k(relevant_labels_array, total_relevant, k)
        ndcg = self.ndcg_at_k(relevant_labels_array, k)
        mrr = self.mrr(relevant_labels_array)

        return {
            "precision_at_k": np.round(precision, 4),
            "recall_at_k": np.round(recall, 4),
            "ndcg_at_k": np.round(ndcg, 4),
            "mrr": np.round(mrr, 4)
        }


def main():
    """
    Main function to compute and print average accuracy metrics.
    """
    # Parameters
    retrieval_results_path = '../results/old/retrieval_results.json'
    N = 10  # Number of retrieved items to consider in metrics

    # Load retrieval results
    with open(retrieval_results_path, 'r', encoding='utf-8') as f:
        retrieval_results: Dict[str, Dict[str, any]] = json.load(f)

    # Initialize metrics
    total_precision = 0.0
    total_recall = 0.0
    total_ndcg = 0.0
    total_mrr = 0.0
    num_queries = len(retrieval_results)

    for query_id, data in retrieval_results.items():
        query_song = data['query']
        retrieved_songs = data['retrieved']

        # Determine relevant items based on genre match
        query_genres = set(query_song['genres'])
        relevant_labels: List[int] = []

        for song in retrieved_songs:
            song_genres = set(song['genres'])
            # TODO: find a better way to determine relevance
            #  They used last year: maybe variant of jaccard (relevant if jaccard index on genre > 0.5)
            is_relevant = int(bool(query_genres & song_genres))  # 1 if any common genre
            relevant_labels.append(is_relevant)

        relevant_labels_array = np.array(relevant_labels)

        # Compute metrics
        precision = Metrics.precision_at_k(relevant_labels_array, N)
        recall = Metrics.recall_at_k(relevant_labels_array, N)
        ndcg = Metrics.ndcg_at_k(relevant_labels_array, N)
        mrr = Metrics.mrr(relevant_labels_array)

        total_precision += precision
        total_recall += recall
        total_ndcg += ndcg
        total_mrr += mrr

    # Average metrics over all queries
    avg_precision = np.round(total_precision / num_queries, 4)
    avg_recall = np.round(total_recall / num_queries, 4)
    avg_ndcg = np.round(total_ndcg / num_queries, 4)
    avg_mrr = np.round(total_mrr / num_queries, 4)

    # Print the averaged metrics
    print(f"Average Precision@{N}: {avg_precision:.4f}")
    print(f"Average Recall@{N}: {avg_recall:.4f}")
    print(f"Average NDCG@{N}: {avg_ndcg:.4f}")
    print(f"Average MRR: {avg_mrr:.4f}")


if __name__ == '__main__':
    main()
