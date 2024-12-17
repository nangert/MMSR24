import json
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import the CORS library

from Music4All import Dataset, Song
from accuracy_metrics import Metrics
from baseline_system import BaselineRetrievalSystem
from bert import BertRetrievalSystem
from mfcc_retrieval import MFCCRetrievalSystem
from tfidf_retrieval import TFIDFRetrievalSystem  # Import the TF-IDF retrieval system

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize dataset and retrieval system
info_dataset_path = 'dataset/id_information_mmsr.tsv'  # Path to your song information TSV file
genres_dataset_path = 'dataset/id_genres_mmsr.tsv'     # Path to your genres TSV file
url_dataset_path = 'dataset/id_url_mmsr.tsv'
metadata_dataset_path = 'dataset/id_metadata_mmsr.tsv'
bert_embeddings_path = 'dataset/id_lyrics_bert_mmsr.tsv'
tfidf_file_path = 'dataset/id_lyrics_tf-idf_mmsr.tsv'
relevance_labels_path = 'dataset/relevance_labels.tsv'

# Initialize dataset and retrieval systems
dataset = Dataset(info_dataset_path, genres_dataset_path, url_dataset_path, metadata_dataset_path, bert_embeddings_path)
baseline_retrieval_system = BaselineRetrievalSystem(dataset)
bert_retrieval_system = BertRetrievalSystem(dataset)
mfcc_retrieval_system = MFCCRetrievalSystem('dataset/id_mfcc_bow_mmsr.tsv', 'dataset/id_mfcc_stats_mmsr.tsv', dataset)
tfidf_retrieval_system = TFIDFRetrievalSystem(tfidf_file_path, relevance_labels_path)  # TF-IDF retrieval system


@app.route('/songs', methods=['GET'])
def get_songs():
    try:
        songs = dataset.get_all_songs()
        songs_dict = [song.to_dict() for song in songs]
        return jsonify(songs_dict)  # Return the list of song dictionaries
    except Exception as e:
        print(f"Error in /songs: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/retrieve', methods=['POST'])
def retrieve_songs():
    try:
        data = request.get_json()
        query_song_id = data.get('query_song_id')
        N = data.get('N', 10)
        model = data.get('model')

        # Get the query song object
        query_song = next((song for song in dataset.get_all_songs() if song.song_id == query_song_id), None)
        if not query_song:
            return jsonify({"error": "Query song not found"}), 404

        if model == 'Baseline':
            retrieved_songs = baseline_retrieval_system.get_retrieval(query_song, N)
        elif model == 'TfIdf':
            retrieved_song_ids = tfidf_retrieval_system.retrieve(query_song_id, N)
            retrieved_songs = [song for song in dataset.get_all_songs() if song.song_id in retrieved_song_ids]
            metrics = tfidf_retrieval_system.calculate_metrics(query_song_id, N)
        elif model == 'Bert':
            retrieved_songs = bert_retrieval_system.get_retrieval(query_song, N)
        elif model == 'MFCC':
            retrieved_songs = mfcc_retrieval_system.recommend_similar_songs(query_song, N)
        else:
            return jsonify({"error": "Invalid model specified"}), 400

        response = {
            'query_song': query_song.to_dict(),
            'result_songs': [song.to_dict() for song in retrieved_songs]
        }

        # Add metrics to response if using TF-IDF
        if model == 'TfIdf':
            response['metrics'] = metrics

        return jsonify(response)

    except Exception as e:
        print(f"Error in /retrieve: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/calculate_metrics', methods=['POST'])
def calculate_metrics():
    try:
        data = request.get_json()
        query_song = data.get('query_song', '')
        result_songs = data.get('result_songs', [])
        k = data.get('k', 10)

        if not query_song or not result_songs:
            return jsonify({"error": "Invalid or missing 'query_song' or 'result_songs'"}), 400

        top_genre_weights = dataset.load_genre_weights('dataset/id_tags_dict.tsv', 'dataset/id_genres_mmsr.tsv')

        query_song_id = query_song.get('song_id', '')
        query_genres = set(top_genre_weights.get(query_song_id, {}))

        if not query_genres:
            return jsonify({"error": f"No genres found for query song: {query_song}"}), 404

        result_songs_filtered_genre = []
        for song in result_songs:
            song_id = song.get('song_id', '')
            top_genres = top_genre_weights.get(song_id, set())
            if top_genres:
                result_songs_filtered_genre.append({
                    "song_id": song_id,
                    "genres": list(top_genres)
                })

        total_relevant = dataset.get_total_relevant(query_song, top_genre_weights)

        metrics_instance = Metrics()
        result = metrics_instance.calculate_metrics(query_song, result_songs_filtered_genre, total_relevant, query_genres, k)

        return jsonify(result)

    except Exception as e:
        print(f"Error in /calculate_metrics: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    app.run(debug=True)
