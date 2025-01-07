import pandas as pd
import matplotlib.pyplot as plt
from baseline_system import BaselineRetrievalSystem
from embedding_system import EmbeddingRetrievalSystem
from mfcc_retrieval import MFCCRetrievalSystem
from accuracy_metrics import Metrics
from Music4All import Dataset

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
    'dataset/id_mfcc_stats_mmsr.tsv'
)

baseline_system = BaselineRetrievalSystem(dataset)
bert_system = EmbeddingRetrievalSystem(dataset, dataset.bert_embeddings, "BERT")
mfcc_system = MFCCRetrievalSystem(dataset)
metrics_instance = Metrics()

# Configuration
NUM_REQUESTS = 10  # Number of requests for evaluation
TOP_K = 10  # Number of top results to consider
retrieval_systems = {
    "Baseline": baseline_system,
    "BERT": bert_system,
    "MFCC": mfcc_system
}
results = []

# Generate and evaluate retrieval results
for system_name, system in retrieval_systems.items():
    print(f"Evaluating {system_name}...")
    for idx, query_song in enumerate(dataset.get_all_songs()[:NUM_REQUESTS]):
        if system_name == "MFCC":
            retrieved_songs = system.recommend_similar_songs(query_song, TOP_K)
        else:
            retrieved_songs = system.get_retrieval(query_song, TOP_K)
        
        total_relevant = dataset.get_total_relevant(query_song.to_dict(), dataset.load_genre_weights(
            'dataset/id_tags_dict.tsv', 'dataset/id_genres_mmsr.tsv'))
        query_genres = set(query_song.genres)
        
        # Calculate metrics
        metrics = metrics_instance.calculate_metrics(
            query_song.to_dict(), 
            [song.to_dict() for song in retrieved_songs],
            total_relevant, 
            query_genres, 
            TOP_K
        )
        
        results.append({
            "system": system_name,
            "query_id": query_song.song_id,
            **metrics
        })

# Store metrics in a DataFrame
df_results = pd.DataFrame(results)
print("\nEvaluation Results:")
print(df_results)

# Visualization
for metric in ["precision_at_k", "recall_at_k", "ndcg_at_k", "mrr"]:
    plt.figure()
    df_results.groupby("system")[metric].mean().plot(kind="bar")
    plt.title(f"Average {metric.upper()} by System")
    plt.ylabel(metric.upper())
    plt.xlabel("Retrieval System")
    plt.show()
