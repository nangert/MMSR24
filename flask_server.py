import random
import json
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import the CORS library
from Music4All import Dataset, Song
from accuracy_metrics import Metrics
from baseline_system import BaselineRetrievalSystem
from bert import BertRetrievalSystem
import numpy as np
from typing import Dict, List

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize dataset and retrieval system
info_dataset_path = 'dataset/id_information_mmsr.tsv'  # Path to your song information TSV file
genres_dataset_path = 'dataset/id_genres_mmsr.tsv'     # Path to your genres TSV file
url_dataset_path = 'dataset/id_url_mmsr.tsv'
metadata_dataset_path = 'dataset/id_metadata_mmsr.tsv'
bert_embeddings_path = 'dataset/id_lyrics_bert_mmsr.tsv'
dataset = Dataset(info_dataset_path, genres_dataset_path, url_dataset_path, metadata_dataset_path, bert_embeddings_path)

# TODO: Add frontend code to give the user the ability to select the retrieval system
# retrieval_system = BaselineRetrievalSystem(dataset)
retrieval_system = BertRetrievalSystem(dataset)


@app.route('/calculate_metrics', methods=['POST'])
def calculate_metrics():

    try:
        # Parse the request body
        data = request.get_json()

        query_song = data.get('query_song', '')
        result_songs = data.get('result_songs', [])
        k = data.get('k', 10)

        if not query_song or not result_songs:
            return jsonify({"error": "Invalid or missing 'query_song' or 'result_songs'"}), 400

        # Load genre weights and compute total relevant items
        top_genre_weights = dataset.load_genre_weights('dataset/id_tags_dict.tsv', 'dataset/id_genres_mmsr.tsv')

        query_song_id = query_song.get('song_id', '')
        # Extract song ID and compute genres with the highest weight
        query_genres = set(top_genre_weights.get(query_song_id, {}))

        if not query_genres:
            return jsonify({"error": f"No genres found for query song: {query_song}"}), 404

        # Replace genres in result_songs with their top genres
        result_songs_filtered_genre = []
        for song in result_songs:
            song_id = song.get('song_id', '')
            top_genres = top_genre_weights.get(song_id, set())
            if top_genres:
                result_songs_filtered_genre.append({
                    "song_id": song_id,
                    "genres": list(top_genres)
                })

        # Determine total relevant items
        total_relevant = dataset.get_total_relevant(query_song, top_genre_weights)

        # Calculate metrics
        metrics_instance = Metrics()
        result = metrics_instance.calculate_metrics(query_song, result_songs_filtered_genre, total_relevant, query_genres, k)

        # Return all metrics as a JSON response
        return jsonify(result)

    except Exception as e:
        print(e)
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

        retrieved_songs = retrieval_system.get_retrieval(query_song, N)
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
