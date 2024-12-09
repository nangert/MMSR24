import random
import json
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import the CORS library
from Music4All import Dataset, Song
from accuracy_metrics import Metrics
from baseline_system import BaselineRetrievalSystem
import numpy as np
from typing import Dict, List

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize dataset and retrieval system
info_dataset_path = 'dataset/id_information_mmsr.tsv'  # Path to your song information TSV file
genres_dataset_path = 'dataset/id_genres_mmsr.tsv'     # Path to your genres TSV file
url_dataset_path = 'dataset/id_url_mmsr.tsv'
metadata_dataset_path = 'dataset/id_metadata_mmsr.tsv'
dataset = Dataset(info_dataset_path, genres_dataset_path, url_dataset_path, metadata_dataset_path)
retrieval_system = BaselineRetrievalSystem(dataset)


@app.route('/calculate_metrics', methods=['POST'])
def calculate_metrics():

    try:
        # Parse the request body
        data = request.get_json()

        query_song = data.get('query_song', {})
        result_songs = data.get('result_songs', [])
        k = data.get('k', 10)

        if not query_song or not result_songs:
            return jsonify({"error": "Invalid or missing 'query_song' or 'result_songs'"}), 400

        # Extract the query genres
        query_genres = set(query_song.get('genres', []))

        # Create a binary relevance list for the result songs
        relevant_labels: List[int] = []
        for song in result_songs:
            song_genres = set(song.get('genres', []))
            is_relevant = int(bool(query_genres & song_genres))  # 1 if there is any genre overlap
            relevant_labels.append(is_relevant)

        relevant_labels_array = np.array(relevant_labels)

        # Calculate metrics
        precision = Metrics.precision_at_k(relevant_labels_array, k)
        recall = Metrics.recall_at_k(relevant_labels_array, k)
        ndcg = Metrics.ndcg_at_k(relevant_labels_array, k)
        mrr = Metrics.mrr(relevant_labels_array)

        # Return all metrics as a JSON response
        return jsonify({
            "precision_at_k": precision,
            "recall_at_k": recall,
            "ndcg_at_k": ndcg,
            "mrr": mrr
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/songs', methods=['GET'])
def get_songs():
    try:
        # Get list of Song objects
        songs = dataset.get_all_songs()

        # Convert each Song object to a dictionary
        songs_dict = [song.to_dict() for song in songs]

        return jsonify(songs_dict)  # Return the list of song dictionaries

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/retrieve', methods=['POST'])
def retrieve_songs():
    try:
        data = request.get_json()
        query_song_id = data.get('query_song_id')
        N = data.get('N', 10)
        query_song = next((song for song in dataset.get_all_songs() if song.song_id == query_song_id), None)
        if not query_song:
            return jsonify({"error": "Query song not found"}), 404

        retrieved_songs = retrieval_system.get_random_retrieval(query_song, N)
        response = {
            'query_song': query_song.to_dict(),
            'result_songs': [song.to_dict() for song in retrieved_songs]
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/generate_retrieval_results', methods=['GET'])
def generate_retrieval_results():
    try:
        data = request.get_json()
        N = data.get('N', 10)
        retrieval_results = retrieval_system.generate_retrieval_results(N)
        retrieval_results_path = 'results/retrieval_results.json'
        with open(retrieval_results_path, 'w', encoding='utf-8') as f:
            json.dump(retrieval_results, f, ensure_ascii=False, indent=4)

        return jsonify({"message": f"Retrieval results saved to {retrieval_results_path}"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    app.run(debug=True)
