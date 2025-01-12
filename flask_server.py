# flask_server.py
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import the CORS library
from numpy import number

from Music4All import Dataset, Song
from diversity.song_diversity_optimizer import SongDiversityOptimizer
from flask_server_utilities import get_query_data
from metrics.accuracy_metrics import Metrics
from retrieval_systems.baseline_system import BaselineRetrievalSystem
from retrieval_systems.embedding_system import EmbeddingRetrievalSystem
from retrieval_systems.lambdarank_system import LambdaRankRetrievalSystem
from retrieval_systems.mfcc_retrieval import MFCCRetrievalSystem
from retrieval_systems.tfidf_retrieval import TFIDFRetrievalSystem
from retrieval_systems.early_fusion import EarlyFusionRetrievalSystem
from retrieval_systems.late_fusion import LateFusionRetrievalSystem

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize dataset and retrieval system
info_dataset_path = 'dataset/id_information_mmsr.tsv'  # Path to your song information TSV file
genres_dataset_path = 'dataset/id_genres_mmsr.tsv'  # Path to your genres TSV file
url_dataset_path = 'dataset/id_url_mmsr.tsv'
metadata_dataset_path = 'dataset/id_metadata_mmsr.tsv'
bert_embeddings_path = 'dataset/id_lyrics_bert_mmsr.tsv'
resnet_embeddings_path = 'dataset/id_resnet_mmsr.tsv'
vgg19_embeddings_path = 'dataset/id_vgg19_mmsr.tsv'
tfidf_embeddings_path = 'dataset/id_lyrics_tf-idf_mmsr.tsv'
tags_dataset_path = 'dataset/id_tags_dict.tsv'
word2vec_embeddings_path = 'dataset/id_lyrics_word2vec_mmsr.tsv'

bow_path = 'dataset/id_mfcc_bow_mmsr.tsv'
stats_path = 'dataset/id_mfcc_stats_mmsr.tsv'

dataset = Dataset(info_dataset_path, genres_dataset_path, url_dataset_path,
                  metadata_dataset_path, bert_embeddings_path, resnet_embeddings_path,
                  vgg19_embeddings_path, bow_path, stats_path, word2vec_embeddings_path)

diversity_optimizer = SongDiversityOptimizer(tags_dataset_path)

bert_retrieval_system = EmbeddingRetrievalSystem(dataset, dataset.bert_embeddings, "Bert")
resnet_retrieval_system = EmbeddingRetrievalSystem(dataset, dataset.resnet_embeddings, "ResNet")
vgg19_retrieval_system = EmbeddingRetrievalSystem(dataset, dataset.vgg19_embeddings, "VGG19")

baseline_retrieval_system = BaselineRetrievalSystem(dataset)
mfcc_retrieval_system = MFCCRetrievalSystem(dataset)
tfidf_retrieval_system = TFIDFRetrievalSystem(dataset, tfidf_embeddings_path)
lambdarank_model = 'models/lambdarank_model.pth'
lambdarank_retrieval_system = LambdaRankRetrievalSystem(dataset, lambdarank_model, dataset.lambdarank_feature_dim)
early_fusion_retrieval_system = EarlyFusionRetrievalSystem(dataset, dataset.bert_embeddings, dataset.resnet_embeddings,
                                                           dataset.mfcc_embeddings_stat, 'dataset/svm_model.pkl')
late_fusion_retrieval_system = LateFusionRetrievalSystem(dataset, dataset.bert_embeddings, dataset.resnet_embeddings,
                                                         dataset.mfcc_embeddings_stat, 'dataset/late_fusion_model.pkl')


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

@app.route('/retrieve/baseline', methods=['POST'])
def retrieve_baseline():
    query_song, n, diversity_optimization = get_query_data(request.get_json(), dataset)

    if not query_song:
        return jsonify({"error": "Query song not found"}), 404

    return retrieve_songs(query_song, n, diversity_optimization, model='Baseline')

@app.route('/retrieve/tfidf', methods=['POST'])
def retrieve_tfidf():
    query_song, n, diversity_optimization = get_query_data(request.get_json(), dataset)

    if not query_song:
        return jsonify({"error": "Query song not found"}), 404

    return retrieve_songs(query_song, n, diversity_optimization, model='TfIdf')

@app.route('/retrieve/bert', methods=['POST'])
def retrieve_bert():
    query_song, n, diversity_optimization = get_query_data(request.get_json(), dataset)

    if not query_song:
        return jsonify({"error": "Query song not found"}), 404

    return retrieve_songs(query_song, n, diversity_optimization, model='Bert')

@app.route('/retrieve/mfcc-bow', methods=['POST'])
def retrieve_mfcc_bow():
    query_song, n, diversity_optimization = get_query_data(request.get_json(), dataset)

    if not query_song:
        return jsonify({"error": "Query song not found"}), 404

    return retrieve_songs(query_song, n, diversity_optimization, model='MFCCBOW')

@app.route('/retrieve/mfcc-bow-cos', methods=['POST'])
def retrieve_mfcc_bow_cos():
    query_song, n, diversity_optimization = get_query_data(request.get_json(), dataset)

    if not query_song:
        return jsonify({"error": "Query song not found"}), 404

    return retrieve_songs(query_song, n, diversity_optimization, model='MFCCBOWCOS')

@app.route('/retrieve/mfcc-stat', methods=['POST'])
def retrieve_mfcc_stat():
    query_song, n, diversity_optimization = get_query_data(request.get_json(), dataset)

    if not query_song:
        return jsonify({"error": "Query song not found"}), 404

    return retrieve_songs(query_song, n, diversity_optimization, model='MFCCSTAT')

@app.route('/retrieve/mfcc-stat-cos', methods=['POST'])
def retrieve_mfcc_stat_cos():
    query_song, n, diversity_optimization = get_query_data(request.get_json(), dataset)

    if not query_song:
        return jsonify({"error": "Query song not found"}), 404

    return retrieve_songs(query_song, n, diversity_optimization, model='MFCCSTATCOS')

@app.route('/retrieve/resnet', methods=['POST'])
def retrieve_resnet():
    query_song, n, diversity_optimization = get_query_data(request.get_json(), dataset)

    if not query_song:
        return jsonify({"error": "Query song not found"}), 404

    return retrieve_songs(query_song, n, diversity_optimization, model='ResNet')

@app.route('/retrieve/vgg19', methods=['POST'])
def retrieve_vgg19():
    query_song, n, diversity_optimization = get_query_data(request.get_json(), dataset)

    if not query_song:
        return jsonify({"error": "Query song not found"}), 404

    return retrieve_songs(query_song, n, diversity_optimization, model='VGG19')

@app.route('/retrieve/lambda', methods=['POST'])
def retrieve_lamdba_mart():
    query_song, n, diversity_optimization = get_query_data(request.get_json(), dataset)

    if not query_song:
        return jsonify({"error": "Query song not found"}), 404

    return retrieve_songs(query_song, n, diversity_optimization, model='LambdaRank')

@app.route('/retrieve/early-fusion', methods=['POST'])
def retrieve_early_fusion():
    query_song, n = get_query_data(request.get_json(), dataset)

    if not query_song:
        return jsonify({"error": "Query song not found"}), 404

    return retrieve_songs(query_song, n, model='EarlyFusion')

@app.route('/retrieve/late-fusion', methods=['POST'])
def retrieve_late_fusion():
    query_song, n = get_query_data(request.get_json(), dataset)

    if not query_song:
        return jsonify({"error": "Query song not found"}), 404

    return retrieve_songs(query_song, n, model='LateFusion')

def retrieve_songs(query_song: Song, n: number, diversity_optimization: bool, model: str):
    if diversity_optimization:
        adapted_n = 5 * n
    else:
        adapted_n = n

    match model:
        case 'Baseline':
            print('baseline')
            retrieved_songs = baseline_retrieval_system.get_retrieval(query_song, adapted_n)
        case 'TfIdf':
            print('tfidf')
            retrieved_songs = tfidf_retrieval_system.retrieve(query_song.song_id, adapted_n)
        case 'Bert':
            print('bert')
            retrieved_songs = bert_retrieval_system.get_retrieval(query_song, adapted_n)
        case 'MFCCBOW':
            print('mfccbow')
            if n <= 100:
                retrieved_songs = mfcc_retrieval_system.recommend_similar_songs_bow(query_song, adapted_n)
            else:
                # Compute similarities on the fly
                retrieved_songs = mfcc_retrieval_system.compute_recommendations_bow(query_song, adapted_n)
        case 'MFCCBOWCOS':
            print('mfccbowcos')
            if n <= 100:
                retrieved_songs = mfcc_retrieval_system.recommend_similar_songs_bow_cos(query_song, adapted_n)
            else:
                # Compute similarities on the fly
                retrieved_songs = mfcc_retrieval_system.compute_recommendations_bow_cos(query_song, adapted_n)
        case 'MFCCSTAT':
            print('mfccstat')
            if n <= 100:
                retrieved_songs = mfcc_retrieval_system.recommend_similar_songs_stat(query_song, adapted_n)
            else:
                # Compute similarities on the fly
                retrieved_songs = mfcc_retrieval_system.compute_recommendations_stat(query_song, adapted_n)
        case 'MFCCSTATCOS':
            print('mfccstatcos')
            if n <= 100:
                retrieved_songs = mfcc_retrieval_system.recommend_similar_songs_stat_cos(query_song, adapted_n)
            else:
                # Compute similarities on the fly
                retrieved_songs = mfcc_retrieval_system.compute_recommendations_stat_cos(query_song, adapted_n)
        case 'ResNet':
            print('resnet')
            retrieved_songs = resnet_retrieval_system.get_retrieval(query_song, adapted_n)
        case 'VGG19':
            print('vgg19')
            retrieved_songs = vgg19_retrieval_system.get_retrieval(query_song, adapted_n)
        case 'LambdaRank':
            print('lambdarank')
            retrieved_songs = lambdarank_retrieval_system.get_retrieval(query_song, adapted_n)
        case 'EarlyFusion':
            print('earlyfusion')
            retrieved_songs = early_fusion_retrieval_system.get_retrieval(query_song, adapted_n)
        case 'LateFusion':
            print('latefusion')
            retrieved_songs = late_fusion_retrieval_system.get_retrieval(query_song, adapted_n)
        case _:
            print('default')
            retrieved_songs = bert_retrieval_system.get_retrieval(query_song, adapted_n)


    print('diversity before optimization')
    cut_list = retrieved_songs[:n]
    print(diversity_optimizer.calculate_diversity_score(cut_list))
    if diversity_optimization:
        retrieved_songs = diversity_optimizer.greedy_optimize_diversity(retrieved_songs, n)

    print('diversity after optimization')
    print(diversity_optimizer.calculate_diversity_score(retrieved_songs))

    response = {
        'query_song': query_song.to_dict(),
        'result_songs': [song.to_dict() for song in retrieved_songs]
    }
    return jsonify(response)


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    app.run(debug=True)
