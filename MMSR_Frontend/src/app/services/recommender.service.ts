import {inject, Injectable, Signal, signal, WritableSignal} from '@angular/core';
import {Observable, of, Subject, switchMap, tap} from 'rxjs';
import {RetrieveResult} from '../models/retrieveResult';
import {RecommenderApiService} from './recommender-api.service';
import {toSignal} from '@angular/core/rxjs-interop';
import {RetrieveApiModel} from '../models/retrieveModel';

@Injectable({
  providedIn: 'root'
})
export class RecommenderService {
  private apiService = inject(RecommenderApiService)

  isLoadingRecommendations: WritableSignal<boolean> = signal(false)

  getBaselineRecommendations: Subject<RetrieveApiModel | void> = new Subject<RetrieveApiModel | void>();
  getBaselineRecommendations$ = this.getBaselineRecommendations.asObservable();

  baselineRecommendations$: Observable<RetrieveResult | undefined> = this.getBaselineRecommendations$.pipe(
    switchMap((model) => {
      if (!model) return of(void 0)

      this.isLoadingRecommendations.set(true)

      return this.apiService.getBaselineRecommendations(model.songId, model.count).pipe(
        tap(() => this.isLoadingRecommendations.set(false))
      )
    })
  )
  baselineRecommendations: Signal<RetrieveResult | undefined> = toSignal(this.baselineRecommendations$)

  getTfIdfRecommendations: Subject<RetrieveApiModel | void> = new Subject<RetrieveApiModel | void>();
  getTfIdfRecommendations$ = this.getTfIdfRecommendations.asObservable();

  tfIdfRecommendations$: Observable<RetrieveResult | undefined> = this.getTfIdfRecommendations$.pipe(
    switchMap((model) => {
      if (!model) return of(void 0)

      this.isLoadingRecommendations.set(true)

      return this.apiService.getTfIdfRecommendations(model.songId, model.count).pipe(
        tap(() => this.isLoadingRecommendations.set(false))
      )
    })
  )
  tfIdfRecommendations: Signal<RetrieveResult | undefined> = toSignal(this.tfIdfRecommendations$)

  getBertRecommendations: Subject<RetrieveApiModel | void> = new Subject<RetrieveApiModel | void>();
  getBertRecommendations$ = this.getBertRecommendations.asObservable();

  bertRecommendations$: Observable<RetrieveResult | undefined> = this.getBertRecommendations$.pipe(
    switchMap((model) => {
      if (!model) return of(void 0)

      this.isLoadingRecommendations.set(true)

      return this.apiService.getBertRecommendations(model.songId, model.count).pipe(
        tap(() => this.isLoadingRecommendations.set(false))
      )
    })
  )
  bertRecommendations: Signal<RetrieveResult | undefined> = toSignal(this.bertRecommendations$)

  getMFCCRecommendations: Subject<RetrieveApiModel | void> = new Subject<RetrieveApiModel | void>();
  getMFCCRecommendations$ = this.getMFCCRecommendations.asObservable();

  mfccRecommendations$: Observable<RetrieveResult | undefined> = this.getMFCCRecommendations$.pipe(
    switchMap((model) => {
      if (!model) return of(void 0)

      this.isLoadingRecommendations.set(true)

      return this.apiService.getMFCCRecommendations(model.songId, model.count).pipe(
        tap(() => this.isLoadingRecommendations.set(false))
      )
    })
  )
  mfccRecommendations: Signal<RetrieveResult | undefined> = toSignal(this.mfccRecommendations$)

  mfccbowRecommendations$: Observable<RetrieveResult | undefined> = this.getMFCCRecommendations$.pipe(
    switchMap((model) => {
      if (!model) return of(void 0)

      this.isLoadingRecommendations.set(true)

      return this.apiService.getMFCCBOWRecommendations(model.songId, model.count).pipe(
        tap(() => this.isLoadingRecommendations.set(false))
      )
    })
  )
  mfccbowRecommendations: Signal<RetrieveResult | undefined> = toSignal(this.mfccbowRecommendations$)

  mfccstatRecommendations$: Observable<RetrieveResult | undefined> = this.getMFCCRecommendations$.pipe(
    switchMap((model) => {
      if (!model) return of(void 0)

      this.isLoadingRecommendations.set(true)

      return this.apiService.getMFCCSTATRecommendations(model.songId, model.count).pipe(
        tap(() => this.isLoadingRecommendations.set(false))
      )
    })
  )
  mfccstatRecommendations: Signal<RetrieveResult | undefined> = toSignal(this.mfccstatRecommendations$)

  getResNetRecommendations: Subject<RetrieveApiModel | void> = new Subject<RetrieveApiModel | void>();
  getResNetRecommendations$ = this.getResNetRecommendations.asObservable();

  resNetRecommendations$: Observable<RetrieveResult | undefined> = this.getResNetRecommendations$.pipe(
    switchMap((model) => {
      if (!model) return of(void 0)

      this.isLoadingRecommendations.set(true)

      return this.apiService.getResNetRecommendations(model.songId, model.count).pipe(
        tap(() => this.isLoadingRecommendations.set(false))
      )
    })
  )
  resNetRecommendations: Signal<RetrieveResult | undefined> = toSignal(this.resNetRecommendations$)

  getVGG19Recommendations: Subject<RetrieveApiModel | void> = new Subject<RetrieveApiModel | void>();
  getVGG19Recommendations$ = this.getVGG19Recommendations.asObservable();

  vgg19Recommendations$: Observable<RetrieveResult | undefined> = this.getVGG19Recommendations$.pipe(
    switchMap((model) => {
      if (!model) return of(void 0)

      this.isLoadingRecommendations.set(true)

      return this.apiService.getVGG19Recommendations(model.songId, model.count).pipe(
        tap(() => this.isLoadingRecommendations.set(false))
      )
    })
  )
  vgg19Recommendations: Signal<RetrieveResult | undefined> = toSignal(this.vgg19Recommendations$)

}
