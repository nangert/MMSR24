import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from retrieval_systems.baseline_system import BaselineRetrievalSystem
from retrieval_systems.embedding_system import EmbeddingRetrievalSystem
from retrieval_systems.mfcc_retrieval import MFCCRetrievalSystem
from retrieval_systems.lambdarank_system import LambdaRankRetrievalSystem
from retrieval_systems.tfidf_retrieval import TFIDFRetrievalSystem
from retrieval_systems.early_fusion import EarlyFusionRetrievalSystem
from retrieval_systems.late_fusion import LateFusionRetrievalSystem
from Music4All import Dataset
from metrics.accuracy_metrics import Metrics
from metrics.beyond_accuracy_metrics import BeyondAccuracyMetrics

# Load the dataset and initialize retrieval systems
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

# Initialize retrieval systems
baseline_system = BaselineRetrievalSystem(dataset)
bert_system = EmbeddingRetrievalSystem(dataset, dataset.bert_embeddings, "BERT")
resnet_system = EmbeddingRetrievalSystem(dataset, dataset.resnet_embeddings, "ResNet")
vgg19_system = EmbeddingRetrievalSystem(dataset, dataset.vgg19_embeddings, "VGG19")
mfcc_system = MFCCRetrievalSystem(dataset)
tfidf_system = TFIDFRetrievalSystem(dataset, 'dataset/id_lyrics_tf-idf_mmsr.tsv')
lambdarank_system = LambdaRankRetrievalSystem(dataset, 'dataset/lambdarank_model.pth', dataset.lambdarank_feature_dim)
early_fusion_system = EarlyFusionRetrievalSystem(dataset, dataset.word2vec_embeddings, dataset.resnet_embeddings, dataset.mfcc_embeddings_stat, 'dataset/svm_model.pkl')
late_fusion_system = LateFusionRetrievalSystem(dataset, dataset.word2vec_embeddings, dataset.resnet_embeddings, dataset.mfcc_embeddings_stat, 'dataset/late_fusion_model.pkl')


metrics_instance = Metrics()

# Configuration
NUM_REQUESTS = 10  # Number of queries to evaluate
TOP_K = 10  # Top-K results to evaluate
retrieval_systems = {
    "Early Fusion": early_fusion_system,
    "Late Fusion": late_fusion_system,
    "Baseline": baseline_system,
    "BERT": bert_system,
    "ResNet": resnet_system,
    "VGG19": vgg19_system,
    "MFCC stat cos": mfcc_system,
    "MFCC bow cos": mfcc_system,
    "MFCC stat": mfcc_system,
    "MFCC bow": mfcc_system,
    "TFIDF": tfidf_system,
    "LambdaRank": lambdarank_system,
}
results = []

# Generate and evaluate retrieval results
for system_name, system in retrieval_systems.items():
    print(f"Evaluating {system_name}...")

    random.seed(100)
    query_songs = random.sample(dataset.get_all_songs(), NUM_REQUESTS)

    for query_song in query_songs:
        # Use the appropriate retrieval method based on the system
        if system_name == "MFCC stat cos":
            retrieved_songs = system.recommend_similar_songs_stat_cos(query_song, TOP_K)
        elif system_name == "MFCC bow cos":
            retrieved_songs = system.recommend_similar_songs_bow_cos(query_song, TOP_K)
        elif system_name == "MFCC stat":
            retrieved_songs = system.recommend_similar_songs_stat(query_song, TOP_K)
        elif system_name == "MFCC bow":
            retrieved_songs = system.recommend_similar_songs_bow(query_song, TOP_K)
        elif system_name == "TFIDF":
            retrieved_songs = system.retrieve(query_song.song_id, TOP_K)
        else:
            retrieved_songs = system.get_retrieval(query_song, TOP_K)
        
        # Compute relevance metrics
        total_relevant = dataset.get_total_relevant(query_song.to_dict(), dataset.load_genre_weights(
            'dataset/id_tags_dict.tsv', 'dataset/id_genres_mmsr.tsv'))
        query_genres = set(query_song.genres)

        # Calculate accuracy metrics
        metrics = metrics_instance.calculate_metrics(
            query_song.to_dict(),
            [song.to_dict() for song in retrieved_songs],
            total_relevant,
            query_genres,
            TOP_K,
        )

        # Compute beyond-accuracy metrics
        catalog_popularity = {
            song.song_id: song.popularity
            for song in dataset.get_all_songs()
        }

        catalog_dicts = [s.to_dict() for s in dataset.get_all_songs()]
        retrieved_songs_dicts = [s.to_dict() for s in retrieved_songs]
        user_history_dicts = [query_song.to_dict()]

        beyond_metrics = {
            "diversity": BeyondAccuracyMetrics.diversity(retrieved_songs_dicts),
            "novelty": BeyondAccuracyMetrics.novelty(retrieved_songs_dicts, catalog_popularity),
            "coverage": BeyondAccuracyMetrics.coverage(retrieved_songs_dicts, len(catalog_dicts)),
            "serendipity": BeyondAccuracyMetrics.serendipity(
                retrieved_songs_dicts,
                user_history_dicts,
                catalog_dicts
            )
        }

        results.append({
            "system": system_name,
            "query_id": query_song.song_id,
            **metrics,
            **beyond_metrics,
        })

# Store metrics in a DataFrame
df_results = pd.DataFrame(results)
print("\nEvaluation Results are stored in evaluation_results.csv")
df_results.to_csv(os.path.join('results', 'evaluation_results.csv'), index=False)


# Function to annotate bars with values
def annotate_bars(ax):
    for bar in ax.patches:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # x-coordinate (center of the bar)
            value,  # y-coordinate (height of the bar)
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="black"
        )


# Visualization of Accuracy Metrics
for metric in ["precision_at_k", "recall_at_k", "ndcg_at_k", "mrr"]:
    plt.figure()
    grouped = df_results.groupby("system")[metric].mean().reset_index()
    ax = plt.bar(grouped["system"], grouped[metric])
    plt.title(f"Average {metric.upper()} by System")
    plt.ylabel(metric.upper())
    plt.xlabel("Retrieval System")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    annotate_bars(plt.gca())  # Annotate bars
    # Save the plot
    plt.savefig(f"results/{metric}_by_system.png")
    plt.show()

# Visualization of Beyond-Accuracy Metrics
for metric in ["diversity", "novelty", "coverage", "serendipity"]:
    plt.figure()
    grouped = df_results.groupby("system")[metric].mean().reset_index()
    ax = plt.bar(grouped["system"], grouped[metric])
    plt.title(f"Average {metric.capitalize()} by System")
    plt.ylabel(metric.capitalize())
    plt.xlabel("Retrieval System")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    annotate_bars(plt.gca())  # Annotate bars
    # Save the plot
    plt.savefig(f"results/{metric}_by_system.png")
    plt.show()

