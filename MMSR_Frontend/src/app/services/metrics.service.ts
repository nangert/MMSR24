import {inject, Injectable, signal, WritableSignal} from '@angular/core';
import {Observable, switchMap, tap} from "rxjs";
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

  baselineMetrics$: Observable<QueryMetrics> = this.recommenderService.baselineRecommendations$.pipe(
    tap(() => this.isLoadingQueryMetrics.set(true)),
    switchMap((body) => {
      if (this.relevanceMeasure) {
        body.relevanceSystem = this.relevanceMeasure
      }
      this.isLoadingQueryMetrics.set(false)
      return this.apiService.getQueryMetrics(body)
    })
  )
  baselineMetrics = toSignal(this.baselineMetrics$)

  tfidfMetrics$: Observable<QueryMetrics> = this.recommenderService.tfIdfRecommendations$.pipe(
    tap(() => this.isLoadingQueryMetrics.set(true)),
    switchMap((body) => {
      if (this.relevanceMeasure) {
        body.relevanceSystem = this.relevanceMeasure
      }
      this.isLoadingQueryMetrics.set(false)
      return this.apiService.getQueryMetrics(body)
    })
  )
  tfidfMetrics = toSignal(this.tfidfMetrics$)

  bertMetrics$: Observable<QueryMetrics> = this.recommenderService.bertRecommendations$.pipe(
    tap(() => this.isLoadingQueryMetrics.set(true)),
    switchMap((body) => {
      if (this.relevanceMeasure) {
        body.relevanceSystem = this.relevanceMeasure
      }
      this.isLoadingQueryMetrics.set(false)
      return this.apiService.getQueryMetrics(body)
    })
  )
  bertMetrics = toSignal(this.bertMetrics$)

  mfccMetrics$: Observable<QueryMetrics> = this.recommenderService.mfccRecommendations$.pipe(
    tap(() => this.isLoadingQueryMetrics.set(true)),
    switchMap((body) => {
      if (this.relevanceMeasure) {
        body.relevanceSystem = this.relevanceMeasure
      }
      this.isLoadingQueryMetrics.set(false)
      return this.apiService.getQueryMetrics(body)
    })
  )
  mfccMetrics = toSignal(this.mfccMetrics$)

  mfccbowMetrics$: Observable<QueryMetrics> = this.recommenderService.mfccbowRecommendations$.pipe(
    tap(() => this.isLoadingQueryMetrics.set(true)),
    switchMap((body) => {
      if (this.relevanceMeasure) {
        body.relevanceSystem = this.relevanceMeasure
      }
      this.isLoadingQueryMetrics.set(false)
      return this.apiService.getQueryMetrics(body)
    })
  )
  mfccbowMetrics = toSignal(this.mfccbowMetrics$)

  mfccstatMetrics$: Observable<QueryMetrics> = this.recommenderService.mfccstatRecommendations$.pipe(
    tap(() => this.isLoadingQueryMetrics.set(true)),
    switchMap((body) => {
      if (this.relevanceMeasure) {
        body.relevanceSystem = this.relevanceMeasure
      }
      this.isLoadingQueryMetrics.set(false)
      return this.apiService.getQueryMetrics(body)
    })
  )
  mfccstatMetrics = toSignal(this.mfccstatMetrics$)

  resnetMetrics$: Observable<QueryMetrics> = this.recommenderService.resNetRecommendations$.pipe(
    tap(() => this.isLoadingQueryMetrics.set(true)),
    switchMap((body) => {
      if (this.relevanceMeasure) {
        body.relevanceSystem = this.relevanceMeasure
      }
      this.isLoadingQueryMetrics.set(false)
      return this.apiService.getQueryMetrics(body)
    })
  )
  resnetMetrics = toSignal(this.resnetMetrics$)

  vgg19Metrics$: Observable<QueryMetrics> = this.recommenderService.vgg19Recommendations$.pipe(
    tap(() => this.isLoadingQueryMetrics.set(true)),
    switchMap((body) => {
      if (this.relevanceMeasure) {
        body.relevanceSystem = this.relevanceMeasure
      }
      this.isLoadingQueryMetrics.set(false)
      return this.apiService.getQueryMetrics(body)
    })
  )
  vgg19Metrics = toSignal(this.vgg19Metrics$)
}
