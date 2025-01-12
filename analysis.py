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
from Music4All import Dataset, Song
from metrics.accuracy_metrics import Metrics
from metrics.beyond_accuracy_metrics import BeyondAccuracyMetrics
from diversity.song_diversity_optimizer import SongDiversityOptimizer  # <-- import your diversity optimizer

# Load the dataset
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
lambdarank_system = LambdaRankRetrievalSystem(dataset, 'models/lambdarank_model.pth', dataset.lambdarank_feature_dim)

metrics_instance = Metrics()
diversity_optimizer = SongDiversityOptimizer('dataset/id_tags_dict.tsv')  # or whichever path you use

# Configuration
NUM_REQUESTS = 10  # Number of queries to evaluate
TOP_K = 10  # Top-K results to evaluate
retrieval_systems = {
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

# ----------------------------------------------------------
# 1) Determine our set of query songs ONE TIME for all runs.
# ----------------------------------------------------------
random.seed(100)
all_songs = dataset.get_all_songs()
query_songs = random.sample(all_songs, NUM_REQUESTS)

# ----------------------------------------------------------
# 2) For each system, run retrieval *twice*:
#    - no diversity (original)
#    - with diversity (retrieve 5*K, then optimize)
# ----------------------------------------------------------
for system_name, system in retrieval_systems.items():
    print(f"\n========================================")
    print(f"Evaluating system: {system_name}")
    print(f"========================================\n")

    for query_song in query_songs:

        # 2.1) Retrieve top-K *without* diversity
        if system_name == "MFCC stat cos":
            retrieved_songs_original = system.recommend_similar_songs_stat_cos(query_song, TOP_K)
        elif system_name == "MFCC bow cos":
            retrieved_songs_original = system.recommend_similar_songs_bow_cos(query_song, TOP_K)
        elif system_name == "MFCC stat":
            retrieved_songs_original = system.recommend_similar_songs_stat(query_song, TOP_K)
        elif system_name == "MFCC bow":
            retrieved_songs_original = system.recommend_similar_songs_bow(query_song, TOP_K)
        elif system_name == "TFIDF":
            retrieved_songs_original = system.retrieve(query_song.song_id, TOP_K)
        else:
            retrieved_songs_original = system.get_retrieval(query_song, TOP_K)

        # 2.2) Retrieve top-5*K for diversity, then reduce to K
        adapted_k = TOP_K * 5
        if system_name == "MFCC stat cos":
            retrieved_songs_div = system.recommend_similar_songs_stat_cos(query_song, adapted_k)
        elif system_name == "MFCC bow cos":
            retrieved_songs_div = system.recommend_similar_songs_bow_cos(query_song, adapted_k)
        elif system_name == "MFCC stat":
            retrieved_songs_div = system.recommend_similar_songs_stat(query_song, adapted_k)
        elif system_name == "MFCC bow":
            retrieved_songs_div = system.recommend_similar_songs_bow(query_song, adapted_k)
        elif system_name == "TFIDF":
            retrieved_songs_div = system.retrieve(query_song.song_id, adapted_k)
        else:
            retrieved_songs_div = system.get_retrieval(query_song, adapted_k)

        # Apply diversity optimization
        retrieved_songs_div = diversity_optimizer.greedy_optimize_diversity(retrieved_songs_div, TOP_K)

        # ----------------------------------------------------------
        # 3) Compute relevance metrics for both sets of results.
        # ----------------------------------------------------------
        total_relevant = dataset.get_total_relevant(
            query_song.to_dict(),
            dataset.load_genre_weights('dataset/id_tags_dict.tsv', 'dataset/id_genres_mmsr.tsv')
        )
        query_genres = set(query_song.genres)

        # For "original" retrieval
        metrics_original = metrics_instance.calculate_metrics(
            query_song.to_dict(),
            [song.to_dict() for song in retrieved_songs_original],
            total_relevant,
            query_genres,
            TOP_K,
        )

        # For "diversified" retrieval
        metrics_div = metrics_instance.calculate_metrics(
            query_song.to_dict(),
            [song.to_dict() for song in retrieved_songs_div],
            total_relevant,
            query_genres,
            TOP_K,
        )

        # ----------------------------------------------------------
        # 4) Compute beyond-accuracy metrics (catalog-level).
        # ----------------------------------------------------------
        catalog_popularity = {song.song_id: song.popularity for song in all_songs}
        catalog_dicts = [s.to_dict() for s in all_songs]

        # Original
        retrieved_songs_dicts_original = [s.to_dict() for s in retrieved_songs_original]
        beyond_original = {
            "diversity": BeyondAccuracyMetrics.diversity(retrieved_songs_dicts_original),
            "novelty": BeyondAccuracyMetrics.novelty(retrieved_songs_dicts_original, catalog_popularity),
            "coverage": BeyondAccuracyMetrics.coverage(retrieved_songs_dicts_original, len(catalog_dicts)),
            "serendipity": BeyondAccuracyMetrics.serendipity(
                retrieved_songs_dicts_original,
                [query_song.to_dict()],
                catalog_dicts
            )
        }

        # Diversified
        retrieved_songs_dicts_div = [s.to_dict() for s in retrieved_songs_div]
        beyond_div = {
            "diversity": BeyondAccuracyMetrics.diversity(retrieved_songs_dicts_div),
            "novelty": BeyondAccuracyMetrics.novelty(retrieved_songs_dicts_div, catalog_popularity),
            "coverage": BeyondAccuracyMetrics.coverage(retrieved_songs_dicts_div, len(catalog_dicts)),
            "serendipity": BeyondAccuracyMetrics.serendipity(
                retrieved_songs_dicts_div,
                [query_song.to_dict()],
                catalog_dicts
            )
        }

        # ----------------------------------------------------------
        # 5) Store two rows in `results`:
        #    - <SystemName> (original)
        #    - <SystemName> + "_div" (with diversity)
        # ----------------------------------------------------------
        results.append({
            "system": system_name,
            "query_id": query_song.song_id,
            **metrics_original,
            **beyond_original
        })
        results.append({
            "system": system_name + "_div",
            "query_id": query_song.song_id,
            **metrics_div,
            **beyond_div
        })

# ------------------------------------------------------------------
# 6) Results DataFrame
# ------------------------------------------------------------------
df_results = pd.DataFrame(results)
print("\nEvaluation Results are stored in evaluation_results.csv")
df_results.to_csv(os.path.join('results', 'evaluation_results.csv'), index=False)


# ------------------------------------------------------------------
# 7) Visualization of Accuracy Metrics (original + _div)
# ------------------------------------------------------------------
def annotate_bars(ax):
    for bar in ax.patches:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
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
    plt.bar(grouped["system"], grouped[metric])
    plt.title(f"Average {metric.upper()} by System")
    plt.ylabel(metric.upper())
    plt.xlabel("Retrieval System")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    annotate_bars(plt.gca())
    plt.savefig(f"results/{metric}_by_system.png")
    plt.show()

# Visualization of Beyond-Accuracy Metrics
for metric in ["diversity", "novelty", "coverage", "serendipity"]:
    plt.figure()
    grouped = df_results.groupby("system")[metric].mean().reset_index()
    plt.bar(grouped["system"], grouped[metric])
    plt.title(f"Average {metric.capitalize()} by System")
    plt.ylabel(metric.capitalize())
    plt.xlabel("Retrieval System")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    annotate_bars(plt.gca())
    plt.savefig(f"results/{metric}_by_system.png")
    plt.show()
