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

  getBaselineRecommendations(querySongId: string, N: number, diversity: boolean): Observable<RetrieveResult> {
    const url = `${this.baseUrl}/retrieve/baseline`;
    return this.http.post<RetrieveResult>(url, {
      query_song_id: querySongId,
      N: N,
      diversity: diversity
    });
  }

  getTfIdfRecommendations(querySongId: string, N: number, diversity: boolean): Observable<RetrieveResult> {
    const url = `${this.baseUrl}/retrieve/tfidf`;
    return this.http.post<RetrieveResult>(url, {
      query_song_id: querySongId,
      N: N,
      diversity: diversity
    });
  }

  getBertRecommendations(querySongId: string, N: number, diversity: boolean): Observable<RetrieveResult> {
    const url = `${this.baseUrl}/retrieve/bert`;
    return this.http.post<RetrieveResult>(url, {
      query_song_id: querySongId,
      N: N,
      diversity: diversity
    });
  }

  getMFCCBOWRecommendations(querySongId: string, N: number, diversity: boolean): Observable<RetrieveResult> {
    const url = `${this.baseUrl}/retrieve/mfcc-bow`;
    return this.http.post<RetrieveResult>(url, {
      query_song_id: querySongId,
      N: N,
      diversity: diversity
    });
  }

  getMFCCBOWCosRecommendations(querySongId: string, N: number, diversity: boolean): Observable<RetrieveResult> {
    const url = `${this.baseUrl}/retrieve/mfcc-bow-cos`;
    return this.http.post<RetrieveResult>(url, {
      query_song_id: querySongId,
      N: N,
      diversity: diversity
    });
  }


  getMFCCSTATRecommendations(querySongId: string, N: number, diversity: boolean): Observable<RetrieveResult> {
    const url = `${this.baseUrl}/retrieve/mfcc-stat`;
    return this.http.post<RetrieveResult>(url, {
      query_song_id: querySongId,
      N: N,
      diversity: diversity
    });
  }

  getMFCCSTATCosRecommendations(querySongId: string, N: number, diversity: boolean): Observable<RetrieveResult> {
    const url = `${this.baseUrl}/retrieve/mfcc-stat-cos`;
    return this.http.post<RetrieveResult>(url, {
      query_song_id: querySongId,
      N: N,
      diversity: diversity
    });
  }

  getResNetRecommendations(querySongId: string, N: number, diversity: boolean): Observable<RetrieveResult> {
    const url = `${this.baseUrl}/retrieve/resnet`;
    return this.http.post<RetrieveResult>(url, {
      query_song_id: querySongId,
      N: N,
      diversity: diversity
    });
  }

  getVGG19Recommendations(querySongId: string, N: number, diversity: boolean): Observable<RetrieveResult> {
    const url = `${this.baseUrl}/retrieve/vgg19`;
    return this.http.post<RetrieveResult>(url, {
      query_song_id: querySongId,
      N: N,
      diversity: diversity
    });
  }

  getLamdaMARTRecommendations(querySongId: string, N: number, diversity: boolean): Observable<RetrieveResult> {
    const url = `${this.baseUrl}/retrieve/lambda`;
    return this.http.post<RetrieveResult>(url, {
      query_song_id: querySongId,
      N: N,
      diversity: diversity
    });
  }

  getEarlyFusionRecommendations(querySongId: string, N: number, diversity: boolean): Observable<RetrieveResult> {
    const url = `${this.baseUrl}/retrieve/early-fusion`;
    return this.http.post<RetrieveResult>(url, {
      query_song_id: querySongId,
      N: N,
      diversity: diversity
    });
  }

  getLateFusionRecommendations(querySongId: string, N: number, diversity: boolean): Observable<RetrieveResult> {
    const url = `${this.baseUrl}/retrieve/late-fusion`;
    return this.http.post<RetrieveResult>(url, {
      query_song_id: querySongId,
      N: N,
      diversity: diversity
    });
  }

  getQueryMetrics(body: RetrieveResult): Observable<QueryMetrics> {
    return this.http.post<QueryMetrics>(`${this.baseUrl}/calculate_metrics`, body)
  }

}
