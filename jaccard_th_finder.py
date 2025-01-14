import numpy as np
import pandas as pd
import random
import os

from typing import List, Dict
from Music4All import Dataset
from metrics.accuracy_metrics import Metrics
from retrieval_systems.baseline_system import BaselineRetrievalSystem
from retrieval_systems.embedding_system import EmbeddingRetrievalSystem
import matplotlib.pyplot as plt


def detect_elbow(x, y):
    """
    Detects the elbow point in a curve using the triangle method.

    Args:
        x (array-like): X values (e.g., thresholds).
        y (array-like): Y values (e.g., metric values).

    Returns:
        float: The X value corresponding to the elbow point.
    """
    # Convert to NumPy arrays to allow negative indexing
    x = np.array(x)
    y = np.array(y)

    # Normalize x and y
    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))
    y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))

    # Compute distances to the line connecting the first and last points
    start = np.array([x_norm[0], y_norm[0]])
    end = np.array([x_norm[-1], y_norm[-1]])
    line_vec = end - start
    line_vec /= np.linalg.norm(line_vec)

    distances = []
    for i in range(len(x_norm)):
        point = np.array([x_norm[i], y_norm[i]])
        proj_len = np.dot(point - start, line_vec)
        proj_point = start + proj_len * line_vec
        distances.append(np.linalg.norm(point - proj_point))

    max_distance_idx = np.argmax(distances)
    return x[max_distance_idx]


def run_baseline_multiple_times(
        dataset: Dataset,
        query_songs: List,
        thresholds: List[float],
        top_k: int = 10,
        runs: int = 5
) -> pd.DataFrame:
    """
    Runs the baseline system multiple times and aggregates the results (mean and std).

    Args:
        dataset (Dataset): The dataset to evaluate.
        query_songs (List): List of query songs.
        thresholds (List[float]): Thresholds to sweep.
        top_k (int): Top-K results to consider.
        runs (int): Number of times to run the baseline.

    Returns:
        pd.DataFrame: Aggregated results with mean and std for each threshold.
    """
    metrics_instance = Metrics()

    all_results = []

    for run in range(runs):
        print(f"Running baseline system iteration {run + 1}/{runs}...")
        baseline_system = BaselineRetrievalSystem(dataset)
        run_results = []

        for thresh in thresholds:
            for query_song in query_songs:
                retrieved_songs = baseline_system.get_retrieval(query_song, top_k)
                total_relevant = dataset.get_total_relevant(
                    query_song.to_dict(),
                    dataset.load_genre_weights('dataset/id_tags_dict.tsv', 'dataset/id_genres_mmsr.tsv')
                )
                query_genres = set(query_song.genres)

                relevant_labels = []
                for candidate in retrieved_songs:
                    candidate_genres = set(candidate.genres)
                    union_g = query_genres | candidate_genres
                    intersect_g = query_genres & candidate_genres
                    if len(union_g) == 0:
                        jaccard = 0.0
                    else:
                        jaccard = len(intersect_g) / len(union_g)

                    is_relevant = int(jaccard >= thresh)
                    relevant_labels.append(is_relevant)

                relevant_labels_array = np.array(relevant_labels)

                # Calculate metrics
                precision = metrics_instance.precision_at_k(relevant_labels_array, top_k)
                recall = metrics_instance.recall_at_k(relevant_labels_array, total_relevant, top_k)
                ndcg = metrics_instance.ndcg_at_k(relevant_labels_array, top_k)
                mrr = metrics_instance.mrr(relevant_labels_array)

                run_results.append({
                    "run": run,
                    "query_id": query_song.song_id,
                    "threshold": thresh,
                    "precision": precision,
                    "recall": recall,
                    "ndcg": ndcg,
                    "mrr": mrr
                })

        all_results.extend(run_results)

    results_df = pd.DataFrame(all_results)

    # Aggregate results (mean and std)
    agg_results = results_df.groupby("threshold").agg(
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        ndcg_mean=("ndcg", "mean"),
        ndcg_std=("ndcg", "std"),
        mrr_mean=("mrr", "mean"),
        mrr_std=("mrr", "std")
    ).reset_index()

    # Find the elbow point
    agg_results["elbow"] = detect_elbow(agg_results["threshold"], agg_results["ndcg_mean"])

    return agg_results

def main():
    # Load dataset
    dataset = Dataset(
        'dataset/id_information_mmsr.tsv',
        'dataset/id_genres_mmsr.tsv',
        'dataset/id_url_mmsr.tsv',
        'dataset/id_metadata_mmsr.tsv',
        'dataset/id_lyrics_bert_mmsr.tsv',
        'dataset/id_resnet_mmsr.tsv',
        'dataset/id_vgg19_mmsr.tsv',
        'dataset/id_mfcc_bow_mmsr.tsv',
        'dataset/id_mfcc_stats_mmsr.tsv',
        'dataset/id_lyrics_word2vec_mmsr.tsv'
    )

    # Sample queries
    all_songs = dataset.get_all_songs()
    random.seed(100)
    query_songs = random.sample(all_songs, 10)  # e.g. 10 queries

    # Define thresholds to sweep
    thresholds = [0] + np.round(np.arange(0.0, 1.05, 0.05), 2).tolist()

    # Run the baseline multiple times
    agg_results = run_baseline_multiple_times(dataset, query_songs, thresholds, top_k=10, runs=3)

    # Save results
    os.makedirs("analysis_output", exist_ok=True)
    agg_results.to_csv("analysis_output/baseline_aggregated_results.csv", index=False)

    print("=== Baseline Aggregated Results ===")
    print(agg_results)

    color_palette = ["#0173B2", "#DE8F05", "#029E73", "#D55E00"]  # Blue, Orange, Green, Red

    # Visualize results
    plt.figure(figsize=(12, 6))
    for idx, (metric, color) in enumerate(zip(["precision", "recall", "ndcg", "mrr"], color_palette)):
        plt.errorbar(
            agg_results["threshold"],
            agg_results[f"{metric}_mean"],
            label=f"{metric.title()} (mean)",
            capsize=3,
            color=color
        )

    plt.xlabel("Threshold")
    plt.ylabel("Metric Value")
    plt.title("Baseline Metrics vs. Threshold")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
