import {inject, Injectable, Signal, signal, WritableSignal} from '@angular/core';
import {Observable, Subject, switchMap, tap} from 'rxjs';
import {RetrieveResult, Song} from '../models/retrieveResult';
import {RecommenderApiService} from './recommender-api.service';
import {toSignal} from '@angular/core/rxjs-interop';
import {QueryMetrics, RetrieveApiModel} from '../models/retrieveModel';

@Injectable({
  providedIn: 'root'
})
export class RecommenderService {
  private apiService = inject(RecommenderApiService)

  relevanceMeasure: any

  reloadSongs: Subject<void> = new Subject<void>();
  reloadSongs$ = this.reloadSongs.asObservable();
  isLoadingSongs: WritableSignal<boolean> = signal(false)

  getRandomRecommendations: Subject<RetrieveApiModel> = new Subject<RetrieveApiModel>();
  getRandomRecommendations$ = this.getRandomRecommendations.asObservable();
  isLoadingRecommendations: WritableSignal<boolean> = signal(false)

  isLoadingQueryMetrics: WritableSignal<boolean> = signal(false)

  songs$: Observable<Song[]> = this.reloadSongs$.pipe(
    tap(() => this.isLoadingSongs.set(true)),
    switchMap(() => {
      this.isLoadingSongs.set(false)
      return this.apiService.retrieveSongs()
    })
  )
  songs = toSignal(this.songs$, { initialValue: []})

  randomRecommendation$: Observable<RetrieveResult> = this.getRandomRecommendations$.pipe(
    tap(() => this.isLoadingRecommendations.set(true)),
    switchMap((model) => {
      this.isLoadingRecommendations.set(false)
      return this.apiService.getRandomRecommendations(model.songId, model.count, model.model)
    })
  )
  randomRecommendation: Signal<RetrieveResult | undefined> = toSignal(this.randomRecommendation$)

  queryMetrics$: Observable<QueryMetrics> = this.randomRecommendation$.pipe(
    tap(() => this.isLoadingQueryMetrics.set(true)),
    switchMap((body) => {
      if (this.relevanceMeasure) {
        body.relevanceSystem = this.relevanceMeasure
      }
      this.isLoadingQueryMetrics.set(false)
      return this.apiService.getQueryMetrics(body)
    })
  )
  queryMetrics = toSignal(this.queryMetrics$)



}
