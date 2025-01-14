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

  isLoadingQueryMetrics: WritableSignal<boolean> = signal(false)

  baselineMetrics$: Observable<QueryMetrics | undefined> = this.recommenderService.baselineRecommendations$.pipe(
    tap(() => this.isLoadingQueryMetrics.set(true)),
    switchMap((body) => {
      if (!body) return of(void 0)

      this.isLoadingQueryMetrics.set(false)
      return this.apiService.getQueryMetrics(body)
    })
  )
  baselineMetrics = toSignal(this.baselineMetrics$)

  tfidfMetrics$: Observable<QueryMetrics | undefined> = this.recommenderService.tfIdfRecommendations$.pipe(
    tap(() => this.isLoadingQueryMetrics.set(true)),
    switchMap((body) => {
      if (!body) return of(void 0)

      this.isLoadingQueryMetrics.set(false)
      return this.apiService.getQueryMetrics(body)
    })
  )
  tfidfMetrics = toSignal(this.tfidfMetrics$)

  bertMetrics$: Observable<QueryMetrics | undefined> = this.recommenderService.bertRecommendations$.pipe(
    tap(() => this.isLoadingQueryMetrics.set(true)),
    switchMap((body) => {
      if (!body) return of(void 0)

      this.isLoadingQueryMetrics.set(false)
      return this.apiService.getQueryMetrics(body)
    })
  )
  bertMetrics = toSignal(this.bertMetrics$)

  mfccbowMetrics$: Observable<QueryMetrics | undefined> = this.recommenderService.mfccbowRecommendations$.pipe(
    tap(() => this.isLoadingQueryMetrics.set(true)),
    switchMap((body) => {
      if (!body) return of(void 0)

      this.isLoadingQueryMetrics.set(false)
      return this.apiService.getQueryMetrics(body)
    })
  )
  mfccbowMetrics = toSignal(this.mfccbowMetrics$)

  /*

  mfccbowCosMetrics$: Observable<QueryMetrics | undefined> = this.recommenderService.mfccbowCosRecommendations$.pipe(
    tap(() => this.isLoadingQueryMetrics.set(true)),
    switchMap((body) => {
      if (!body) return of(void 0)

      this.isLoadingQueryMetrics.set(false)
      return this.apiService.getQueryMetrics(body)
    })
  )
  mfccbowCosMetrics = toSignal(this.mfccbowCosMetrics$)
   */

  mfccstatMetrics$: Observable<QueryMetrics | undefined> = this.recommenderService.mfccstatRecommendations$.pipe(
    tap(() => this.isLoadingQueryMetrics.set(true)),
    switchMap((body) => {
      if (!body) return of(void 0)

      this.isLoadingQueryMetrics.set(false)
      return this.apiService.getQueryMetrics(body)
    })
  )
  mfccstatMetrics = toSignal(this.mfccstatMetrics$)

  /*

  mfccstatCosMetrics$: Observable<QueryMetrics | undefined> = this.recommenderService.mfccstatCosRecommendations$.pipe(
    tap(() => this.isLoadingQueryMetrics.set(true)),
    switchMap((body) => {
      if (!body) return of(void 0)

      this.isLoadingQueryMetrics.set(false)
      return this.apiService.getQueryMetrics(body)
    })
  )
  mfccstatCosMetrics = toSignal(this.mfccstatCosMetrics$)
   */

  resnetMetrics$: Observable<QueryMetrics | undefined> = this.recommenderService.resNetRecommendations$.pipe(
    tap(() => this.isLoadingQueryMetrics.set(true)),
    switchMap((body) => {
      if (!body) return of(void 0)

      this.isLoadingQueryMetrics.set(false)
      return this.apiService.getQueryMetrics(body)
    })
  )
  resnetMetrics = toSignal(this.resnetMetrics$)

  vgg19Metrics$: Observable<QueryMetrics | undefined> = this.recommenderService.vgg19Recommendations$.pipe(
    tap(() => this.isLoadingQueryMetrics.set(true)),
    switchMap((body) => {
      if (!body) return of(void 0)

      this.isLoadingQueryMetrics.set(false)
      return this.apiService.getQueryMetrics(body)
    })
  )
  vgg19Metrics = toSignal(this.vgg19Metrics$)

  lamdaMARTMetrics$: Observable<QueryMetrics | undefined> = this.recommenderService.lamdaMARTRecommendations$.pipe(
    tap(() => this.isLoadingQueryMetrics.set(true)),
    switchMap((body) => {
      if (!body) return of(void 0)

      this.isLoadingQueryMetrics.set(false)
      return this.apiService.getQueryMetrics(body)
    })
  )
  lamdaMARTMetrics = toSignal(this.lamdaMARTMetrics$)

  earlyFusionMetrics$: Observable<QueryMetrics | undefined> = this.recommenderService.earlyFusionRecommendations$.pipe(
    tap(() => this.isLoadingQueryMetrics.set(true)),
    switchMap((body) => {
      if (!body) return of(void 0)

      this.isLoadingQueryMetrics.set(false)
      return this.apiService.getQueryMetrics(body)
    })
  )
  earlyFusionMetrics = toSignal(this.earlyFusionMetrics$)

  lateFusionMetrics$: Observable<QueryMetrics | undefined> = this.recommenderService.lateFusionRecommendations$.pipe(
    tap(() => this.isLoadingQueryMetrics.set(true)),
    switchMap((body) => {
      if (!body) return of(void 0)

      this.isLoadingQueryMetrics.set(false)
      return this.apiService.getQueryMetrics(body)
    })
  )
  lateFusionMetrics = toSignal(this.lateFusionMetrics$)
}
