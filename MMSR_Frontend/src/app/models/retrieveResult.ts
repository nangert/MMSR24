export interface RetrieveResult {
  query_song: Song
  result_songs: Song[]
}

export interface Song {
  album_name: string,
  artist: string,
  genres: string[],
  song_id: string,
  song_title: string,
  url: string,
  spotify_id: string
}
