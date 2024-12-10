import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.sparse import csr_matrix

# Load all .tsv files from a folder and concatenate them
def load_tfidf_datasets(folder_path):
    """Load all .tsv files from a folder and concatenate them into one DataFrame."""
    dataframes = []
    for file in os.listdir(folder_path):
        if file.endswith(".tsv"):
            file_path = os.path.join(folder_path, file)
            print(f"Loading file: {file_path}")
            # Load only the 'id' and TF-IDF feature columns from each file
            df = pd.read_csv(file_path, sep="\t", usecols=['id'] + [col for col in pd.read_csv(file_path, sep="\t", nrows=1).columns if col != 'id'])
            dataframes.append(df)

    # Concatenate all the loaded dataframes into one
    full_data = pd.concat(dataframes, ignore_index=True)

    # Handle missing values (NaN)
    full_data = full_data.fillna(0)

    return full_data

# TF-IDF Retrieval System using sparse matrix
class TFIDFRetrievalSystem:
    def __init__(self, data):
        self.data = data
        # Convert features to a sparse matrix to save memory
        self.features = csr_matrix(data.drop(columns=['id']).values)

    def rank_songs(self, query_id, top_n=10):
        """Rank songs based on cosine similarity."""
        query_idx = self.data.index[self.data['id'] == query_id].tolist()
        if not query_idx:
            raise ValueError(f"Query ID {query_id} not found in dataset.")

        query_vector = self.features[query_idx[0]].reshape(1, -1)
        similarities = cosine_similarity(query_vector, self.features).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_n]
        return self.data.iloc[top_indices], similarities[top_indices]

# Evaluation Metrics
def evaluate_metrics(query_id, retrieved_songs, similarities, k=10):
    """Evaluate Precision@K, Recall@K, NDCG@K, and MRR."""
    # Assuming relevance based on the same genre as the query song
    relevant_genre = query_id.split('_')[0]  # Extract genre from query_id or set manually
    relevant = set(retrieved_songs[retrieved_songs['id'].str.contains(relevant_genre)]['id'])

    # Top-K indices
    retrieved_indices = retrieved_songs['id'][:k]

    # Precision@K
    precision = len(set(retrieved_indices) & relevant) / k

    # Recall@K
    recall = len(set(retrieved_indices) & relevant) / len(relevant) if relevant else 0

    # NDCG@K
    gains = [1 if song_id in relevant else 0 for song_id in retrieved_indices]
    dcg = sum(g / np.log2(idx + 2) for idx, g in enumerate(gains))
    idcg = sum(1 / np.log2(idx + 2) for idx in range(len(relevant)))
    ndcg = dcg / idcg if idcg > 0 else 0

    # MRR
    reciprocal_ranks = [1 / (idx + 1) for idx, song_id in enumerate(retrieved_indices) if song_id in relevant]
    mrr = reciprocal_ranks[0] if reciprocal_ranks else 0

    return {"Precision@10": precision, "Recall@10": recall, "NDCG@10": ndcg, "MRR": mrr}

# Main Script
if __name__ == "__main__":
    # Specify the folder containing .tsv files
    folder_path = r"C:\Users\ASUS\IdeaProjects\MMSR24\dataset"  # <-- Update this with your folder path

    # Load dataset from the folder
    data = load_tfidf_datasets(folder_path)

    # Display dataset info
    print("Dataset Loaded Successfully")
    print(f"Dataset shape: {data.shape}")
    print(f"Columns in dataset: {data.columns}")

    # Ensure the dataset contains the 'id' column
    if 'id' not in data.columns:
        raise ValueError("Dataset must contain an 'id' column")

    # Display the first few rows
    print("Sample data (first 5 rows):")
    print(data.head())

    # Initialize retrieval system
    retrieval_system = TFIDFRetrievalSystem(data)

    # Example query: Use the first song's ID (you can change it to any ID in the dataset)
    query_id = data.iloc[0]['id']
    print(f"Query Song ID: {query_id}")

    # Rank songs
    ranked_songs, similarities = retrieval_system.rank_songs(query_id, top_n=10)

    # Display results
    print("\nRanked Results:")
    for idx, (song, sim) in enumerate(zip(ranked_songs.itertuples(), similarities)):
        print(f"{idx + 1}. {song.id} (Similarity: {sim:.4f})")

    # Evaluate metrics
    metrics = evaluate_metrics(query_id, ranked_songs, similarities)
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
