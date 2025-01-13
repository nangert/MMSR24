import os
import random
import pandas as pd

from plot_results import make_plots
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

# Import our updated SongDiversityOptimizer with semi & cluster methods
from diversity.song_diversity_optimizer import SongDiversityOptimizer

# -------------------------------------------------------------------
# TOGGLE THIS FLAG TO ENABLE / DISABLE DIVERSITY OPTIMIZATION
# -------------------------------------------------------------------
USE_DIVERSITY_OPTIMIZATION = False

# 1) Load the dataset
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

# 2) Initialize retrieval systems
baseline_system = BaselineRetrievalSystem(dataset)
bert_system = EmbeddingRetrievalSystem(dataset, dataset.bert_embeddings, "BERT")
resnet_system = EmbeddingRetrievalSystem(dataset, dataset.resnet_embeddings, "ResNet")
vgg19_system = EmbeddingRetrievalSystem(dataset, dataset.vgg19_embeddings, "VGG19")
mfcc_system = MFCCRetrievalSystem(dataset)
tfidf_system = TFIDFRetrievalSystem(dataset, 'dataset/id_lyrics_tf-idf_mmsr.tsv')
lambdarank_system = LambdaRankRetrievalSystem(
    dataset,
    'models/lambdarank_model.pth',
    dataset.lambdarank_feature_dim
)
early_fusion_system = EarlyFusionRetrievalSystem(
    dataset,
    dataset.word2vec_embeddings,
    dataset.resnet_embeddings,
    dataset.mfcc_embeddings_stat,
    'models/early_fusion_model.pkl'
)
late_fusion_system = LateFusionRetrievalSystem(
    dataset,
    dataset.word2vec_embeddings,
    dataset.resnet_embeddings,
    dataset.mfcc_embeddings_stat,
    'models/late_fusion_model.pkl'
)

metrics_instance = Metrics()
diversity_optimizer = SongDiversityOptimizer('dataset/id_tags_dict.tsv')

# 3) Config
NUM_REQUESTS = 10  # Number of queries to evaluate
TOP_K = 10          # Top-K results
retrieval_systems = {
    "Early Fusion": early_fusion_system,
    "Late Fusion": late_fusion_system,
    "Baseline": baseline_system,
    "BERT": bert_system,
    "ResNet": resnet_system,
    "VGG19": vgg19_system,
    "MFCC bow": mfcc_system,
    "TFIDF": tfidf_system,
    "LambdaRank": lambdarank_system,
}
results = []

# 4) One-time sample of queries
random.seed(100)
all_songs = dataset.get_all_songs()
query_songs = random.sample(all_songs, NUM_REQUESTS)

# 5) Evaluate each system. If USE_DIVERSITY_OPTIMIZATION=True, do all 4 approaches.
for system_name, system in retrieval_systems.items():
    print(f"\n========================================")
    print(f"Evaluating system: {system_name}")
    print(f"========================================\n")

    for query_song in query_songs:
        # (A) Always do the original retrieval (K)
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

        # Compute metrics for the original retrieval
        total_relevant = dataset.get_total_relevant(
            query_song.to_dict(),
            dataset.load_genre_weights('dataset/id_tags_dict.tsv', 'dataset/id_genres_mmsr.tsv')
        )
        query_genres = set(query_song.genres)

        metrics_original = metrics_instance.calculate_metrics(
            query_song.to_dict(),
            [s.to_dict() for s in retrieved_songs_original],
            total_relevant,
            query_genres,
            TOP_K
        )

        catalog_popularity = {sng.song_id: sng.popularity for sng in all_songs}
        catalog_dicts = [sng.to_dict() for sng in all_songs]
        retrieved_dicts_original = [s.to_dict() for s in retrieved_songs_original]
        beyond_original = {
            "diversity": BeyondAccuracyMetrics.diversity(retrieved_dicts_original),
            "novelty": BeyondAccuracyMetrics.novelty(retrieved_dicts_original, catalog_popularity),
            "coverage": BeyondAccuracyMetrics.coverage(retrieved_dicts_original, len(catalog_dicts)),
            "serendipity": BeyondAccuracyMetrics.serendipity(
                retrieved_dicts_original, [query_song.to_dict()], catalog_dicts
            )
        }

        # Add row for "no diversity"
        results.append({
            "system": system_name,
            "query_id": query_song.song_id,
            **metrics_original,
            **beyond_original
        })

        # (B) If we want diversity, retrieve top-(5*K) and apply all 3 methods
        if USE_DIVERSITY_OPTIMIZATION:
            adapted_k = TOP_K * 5
            if system_name == "MFCC stat cos":
                retrieved_songs_5k = system.recommend_similar_songs_stat_cos(query_song, adapted_k)
            elif system_name == "MFCC bow cos":
                retrieved_songs_5k = system.recommend_similar_songs_bow_cos(query_song, adapted_k)
            elif system_name == "MFCC stat":
                retrieved_songs_5k = system.recommend_similar_songs_stat(query_song, adapted_k)
            elif system_name == "MFCC bow":
                retrieved_songs_5k = system.recommend_similar_songs_bow(query_song, adapted_k)
            elif system_name == "TFIDF":
                retrieved_songs_5k = system.retrieve(query_song.song_id, adapted_k)
            else:
                retrieved_songs_5k = system.get_retrieval(query_song, adapted_k)

            # a) Greedy
            retrieved_div_greedy = diversity_optimizer.greedy_optimize_diversity(retrieved_songs_5k, TOP_K)
            metrics_div_greedy = metrics_instance.calculate_metrics(
                query_song.to_dict(),
                [s.to_dict() for s in retrieved_div_greedy],
                total_relevant,
                query_genres,
                TOP_K
            )
            retrieved_dicts_greedy = [s.to_dict() for s in retrieved_div_greedy]
            beyond_div_greedy = {
                "diversity": BeyondAccuracyMetrics.diversity(retrieved_dicts_greedy),
                "novelty": BeyondAccuracyMetrics.novelty(retrieved_dicts_greedy, catalog_popularity),
                "coverage": BeyondAccuracyMetrics.coverage(retrieved_dicts_greedy, len(catalog_dicts)),
                "serendipity": BeyondAccuracyMetrics.serendipity(
                    retrieved_dicts_greedy, [query_song.to_dict()], catalog_dicts
                )
            }
            results.append({
                "system": system_name + "_div_greedy",
                "query_id": query_song.song_id,
                **metrics_div_greedy,
                **beyond_div_greedy
            })

            # b) Semi-Greedy
            retrieved_div_semi = diversity_optimizer.semi_greedy_optimize_diversity(retrieved_songs_5k, TOP_K)
            metrics_div_semi = metrics_instance.calculate_metrics(
                query_song.to_dict(),
                [s.to_dict() for s in retrieved_div_semi],
                total_relevant,
                query_genres,
                TOP_K
            )
            retrieved_dicts_semi = [s.to_dict() for s in retrieved_div_semi]
            beyond_div_semi = {
                "diversity": BeyondAccuracyMetrics.diversity(retrieved_dicts_semi),
                "novelty": BeyondAccuracyMetrics.novelty(retrieved_dicts_semi, catalog_popularity),
                "coverage": BeyondAccuracyMetrics.coverage(retrieved_dicts_semi, len(catalog_dicts)),
                "serendipity": BeyondAccuracyMetrics.serendipity(
                    retrieved_dicts_semi, [query_song.to_dict()], catalog_dicts
                )
            }
            results.append({
                "system": system_name + "_div_semi",
                "query_id": query_song.song_id,
                **metrics_div_semi,
                **beyond_div_semi
            })

            # c) Cluster
            retrieved_div_cluster = diversity_optimizer.cluster_optimize_diversity_tags(retrieved_songs_5k, TOP_K)
            metrics_div_cluster = metrics_instance.calculate_metrics(
                query_song.to_dict(),
                [s.to_dict() for s in retrieved_div_cluster],
                total_relevant,
                query_genres,
                TOP_K
            )
            retrieved_dicts_cluster = [s.to_dict() for s in retrieved_div_cluster]
            beyond_div_cluster = {
                "diversity": BeyondAccuracyMetrics.diversity(retrieved_dicts_cluster),
                "novelty": BeyondAccuracyMetrics.novelty(retrieved_dicts_cluster, catalog_popularity),
                "coverage": BeyondAccuracyMetrics.coverage(retrieved_dicts_cluster, len(catalog_dicts)),
                "serendipity": BeyondAccuracyMetrics.serendipity(
                    retrieved_dicts_cluster, [query_song.to_dict()], catalog_dicts
                )
            }
            results.append({
                "system": system_name + "_div_cluster",
                "query_id": query_song.song_id,
                **metrics_div_cluster,
                **beyond_div_cluster
            })

# 9) Build dataframe & write out
df_results = pd.DataFrame(results)
os.makedirs("results/test3", exist_ok=True)
output_csv = os.path.join("results/test3", "evaluation_results.csv")
df_results.to_csv(output_csv, index=False)

make_plots(df_results, output_folder="results/test2")