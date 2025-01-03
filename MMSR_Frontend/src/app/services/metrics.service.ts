import {inject, Injectable, signal, WritableSignal} from '@angular/core';
import {Observable, of, switchMap, tap} from "rxjs";
import {QueryMetrics} from "../models/retrieveModel";
import {toSignal} from "@angular/core/rxjs-interop";
import {RecommenderApiService} from "./recommender-api.service";
import {RecommenderService} from "./recommender.service";

@Injectable({
  providedIn: 'root'
})
export class MetricsService {
  private apiService = inject(RecommenderApiService)
  private recommenderService = inject(RecommenderService)

  relevanceMeasure: any
  isLoadingQueryMetrics: WritableSignal<boolean> = signal(false)

  baselineMetrics$: Observable<QueryMetrics | undefined> = this.recommenderService.baselineRecommendations$.pipe(
    tap(() => this.isLoadingQueryMetrics.set(true)),
    switchMap((body) => {
      if (!body) return of(void 0)

      if (this.relevanceMeasure) {
        body.relevanceSystem = this.relevanceMeasure
      }
      this.isLoadingQueryMetrics.set(false)
      return this.apiService.getQueryMetrics(body)
    })
  )
  baselineMetrics = toSignal(this.baselineMetrics$)

  tfidfMetrics$: Observable<QueryMetrics | undefined> = this.recommenderService.tfIdfRecommendations$.pipe(
    tap(() => this.isLoadingQueryMetrics.set(true)),
    switchMap((body) => {
      if (!body) return of(void 0)

      if (this.relevanceMeasure) {
        body.relevanceSystem = this.relevanceMeasure
      }
      this.isLoadingQueryMetrics.set(false)
      return this.apiService.getQueryMetrics(body)
    })
  )
  tfidfMetrics = toSignal(this.tfidfMetrics$)

  bertMetrics$: Observable<QueryMetrics | undefined> = this.recommenderService.bertRecommendations$.pipe(
    tap(() => this.isLoadingQueryMetrics.set(true)),
    switchMap((body) => {
      if (!body) return of(void 0)

      if (this.relevanceMeasure) {
        body.relevanceSystem = this.relevanceMeasure
      }
      this.isLoadingQueryMetrics.set(false)
      return this.apiService.getQueryMetrics(body)
    })
  )
  bertMetrics = toSignal(this.bertMetrics$)

  mfccMetrics$: Observable<QueryMetrics | undefined> = this.recommenderService.mfccRecommendations$.pipe(
    tap(() => this.isLoadingQueryMetrics.set(true)),
    switchMap((body) => {
      if (!body) return of(void 0)

      if (this.relevanceMeasure) {
        body.relevanceSystem = this.relevanceMeasure
      }
      this.isLoadingQueryMetrics.set(false)
      return this.apiService.getQueryMetrics(body)
    })
  )
  mfccMetrics = toSignal(this.mfccMetrics$)

  mfccbowMetrics$: Observable<QueryMetrics | undefined> = this.recommenderService.mfccbowRecommendations$.pipe(
    tap(() => this.isLoadingQueryMetrics.set(true)),
    switchMap((body) => {
      if (!body) return of(void 0)

      if (this.relevanceMeasure) {
        body.relevanceSystem = this.relevanceMeasure
      }
      this.isLoadingQueryMetrics.set(false)
      return this.apiService.getQueryMetrics(body)
    })
  )
  mfccbowMetrics = toSignal(this.mfccbowMetrics$)

  mfccstatMetrics$: Observable<QueryMetrics | undefined> = this.recommenderService.mfccstatRecommendations$.pipe(
    tap(() => this.isLoadingQueryMetrics.set(true)),
    switchMap((body) => {
      if (!body) return of(void 0)

      if (this.relevanceMeasure) {
        body.relevanceSystem = this.relevanceMeasure
      }
      this.isLoadingQueryMetrics.set(false)
      return this.apiService.getQueryMetrics(body)
    })
  )
  mfccstatMetrics = toSignal(this.mfccstatMetrics$)

  resnetMetrics$: Observable<QueryMetrics | undefined> = this.recommenderService.resNetRecommendations$.pipe(
    tap(() => this.isLoadingQueryMetrics.set(true)),
    switchMap((body) => {
      if (!body) return of(void 0)

      if (this.relevanceMeasure) {
        body.relevanceSystem = this.relevanceMeasure
      }
      this.isLoadingQueryMetrics.set(false)
      return this.apiService.getQueryMetrics(body)
    })
  )
  resnetMetrics = toSignal(this.resnetMetrics$)

  vgg19Metrics$: Observable<QueryMetrics | undefined> = this.recommenderService.vgg19Recommendations$.pipe(
    tap(() => this.isLoadingQueryMetrics.set(true)),
    switchMap((body) => {
      if (!body) return of(void 0)

      if (this.relevanceMeasure) {
        body.relevanceSystem = this.relevanceMeasure
      }
      this.isLoadingQueryMetrics.set(false)
      return this.apiService.getQueryMetrics(body)
    })
  )
  vgg19Metrics = toSignal(this.vgg19Metrics$)
}
