import pandas as pd
import matplotlib.pyplot as plt
from retrieval_systems.baseline_system import BaselineRetrievalSystem
from retrieval_systems.embedding_system import EmbeddingRetrievalSystem
from retrieval_systems.mfcc_retrieval import MFCCRetrievalSystem
from retrieval_systems.lambdamart_system import LambdaMARTRetrievalSystem
from retrieval_systems.tfidf_retrieval import TFIDFRetrievalSystem
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
    'dataset/id_lambdamart_feats.tsv',
)

# Initialize retrieval systems
baseline_system = BaselineRetrievalSystem(dataset)
bert_system = EmbeddingRetrievalSystem(dataset, dataset.bert_embeddings, "BERT")
resnet_system = EmbeddingRetrievalSystem(dataset, dataset.resnet_embeddings, "ResNet")
vgg19_system = EmbeddingRetrievalSystem(dataset, dataset.vgg19_embeddings, "VGG19")
mfcc_system = MFCCRetrievalSystem(dataset)
tfidf_system = TFIDFRetrievalSystem(dataset, 'dataset/id_lyrics_tf-idf_mmsr.tsv')
lambdamart_system = LambdaMARTRetrievalSystem(dataset, 'lambdamart.pkl', dataset.lambdamart_feature_dim)

metrics_instance = Metrics()

# Configuration
NUM_REQUESTS = 10  # Number of queries to evaluate
TOP_K = 10  # Top-K results to evaluate
retrieval_systems = {
    "Baseline": baseline_system,
    "BERT": bert_system,
    "ResNet": resnet_system,
    "VGG19": vgg19_system,
    "MFCC": mfcc_system,
    "TFIDF": tfidf_system,
    "LambdaMART": lambdamart_system,
}
results = []

# Generate and evaluate retrieval results
for system_name, system in retrieval_systems.items():
    print(f"Evaluating {system_name}...")
    for idx, query_song in enumerate(dataset.get_all_songs()[:NUM_REQUESTS]):
        # Use the appropriate retrieval method based on the system
        if system_name == "MFCC":
            retrieved_songs = system.recommend_similar_songs_stat_cos(query_song, TOP_K)
        elif system_name == "TFIDF":
            retrieved_songs = system.retrieve(query_song.song_id, TOP_K)
        elif system_name == "LambdaMART":
            retrieved_songs = system.get_retrieval(query_song, TOP_K)
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
        catalog = dataset.get_all_songs()  # Use the full dataset for coverage calculation
        user_history = query_genres  # Using genres as the user's history
        beyond_metrics = {
            "diversity": BeyondAccuracyMetrics.diversity(retrieved_songs),
            "novelty": BeyondAccuracyMetrics.novelty(retrieved_songs, user_history),
            "coverage": BeyondAccuracyMetrics.coverage(retrieved_songs, catalog),
            "serendipity": BeyondAccuracyMetrics.serendipity(retrieved_songs, user_history),
        }

        results.append({
            "system": system_name,
            "query_id": query_song.song_id,
            **metrics,
            **beyond_metrics,
        })

# Store metrics in a DataFrame
df_results = pd.DataFrame(results)
print("\nEvaluation Results:")
print(df_results)

# Visualization of Accuracy Metrics
for metric in ["precision_at_k", "recall_at_k", "ndcg_at_k", "mrr"]:
    plt.figure()
    df_results.groupby("system")[metric].mean().plot(kind="bar")
    plt.title(f"Average {metric.upper()} by System")
    plt.ylabel(metric.upper())
    plt.xlabel("Retrieval System")
    plt.show()

# Visualization of Beyond-Accuracy Metrics
for metric in ["diversity", "novelty", "coverage", "serendipity"]:
    plt.figure()
    df_results.groupby("system")[metric].mean().plot(kind="bar")
    plt.title(f"Average {metric.capitalize()} by System")
    plt.ylabel(metric.capitalize())
    plt.xlabel("Retrieval System")
    plt.show()
