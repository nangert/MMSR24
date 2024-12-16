import {inject, Injectable} from '@angular/core';
import {HttpClient} from '@angular/common/http';
import {Observable} from 'rxjs';
import {RetrieveResult, Song} from '../models/retrieveResult';
import {QueryMetrics} from "../models/retrieveModel";

@Injectable({
  providedIn: 'root'
})
export class RecommenderApiService {
  private baseUrl = 'http://127.0.0.1:5000';
  private http = inject(HttpClient)

  checkHealth(): Observable<any> {
    return this.http.get(`${this.baseUrl}/health`);
  }

  retrieveSongs(): Observable<Song[]> {
    return this.http.get<Song[]>(`${this.baseUrl}/songs`);
  }

  getRandomRecommendations(querySongId: string, N: number, model: string): Observable<RetrieveResult> {
    const url = `${this.baseUrl}/retrieve`;
    const body = {
      query_song_id: querySongId,
      N: N,
      model: model
    };
    return this.http.post<RetrieveResult>(url, body);
  }

  getQueryMetrics(body: RetrieveResult): Observable<QueryMetrics> {
    return this.http.post<QueryMetrics>(`${this.baseUrl}/calculate_metrics`, body)
  }

}
