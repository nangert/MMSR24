
def get_query_data(data, dataset):
    query_song_id = data.get('query_song_id')
    n = data.get('N', 10)
    divserity_optimization = data.get('diversity')

    query_song = next((song for song in dataset.get_all_songs() if song.song_id == query_song_id), None)

    return query_song, n, divserity_optimization

