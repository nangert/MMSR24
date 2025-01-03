import {inject, Injectable, Signal, signal, WritableSignal} from '@angular/core';
import {Observable, Subject, switchMap, tap} from 'rxjs';
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

  getBaselineRecommendations: Subject<RetrieveApiModel> = new Subject<RetrieveApiModel>();
  getBaselineRecommendations$ = this.getBaselineRecommendations.asObservable();

  baselineRecommendations$: Observable<RetrieveResult> = this.getBaselineRecommendations$.pipe(
    tap(() => this.isLoadingRecommendations.set(true)),
    switchMap((model) => {
      return this.apiService.getBaselineRecommendations(model.songId, model.count).pipe(
        tap(() => this.isLoadingRecommendations.set(false))
      )
    })
  )
  baselineRecommendations: Signal<RetrieveResult | undefined> = toSignal(this.baselineRecommendations$)

  getTfIdfRecommendations: Subject<RetrieveApiModel> = new Subject<RetrieveApiModel>();
  getTfIdfRecommendations$ = this.getTfIdfRecommendations.asObservable();

  tfIdfRecommendations$: Observable<RetrieveResult> = this.getTfIdfRecommendations$.pipe(
    tap(() => this.isLoadingRecommendations.set(true)),
    switchMap((model) => {
      return this.apiService.getTfIdfRecommendations(model.songId, model.count).pipe(
        tap(() => this.isLoadingRecommendations.set(false))
      )
    })
  )
  tfIdfRecommendations: Signal<RetrieveResult | undefined> = toSignal(this.tfIdfRecommendations$)

  getBertRecommendations: Subject<RetrieveApiModel> = new Subject<RetrieveApiModel>();
  getBertRecommendations$ = this.getBertRecommendations.asObservable();

  bertRecommendations$: Observable<RetrieveResult> = this.getBertRecommendations$.pipe(
    tap(() => this.isLoadingRecommendations.set(true)),
    switchMap((model) => {
      return this.apiService.getBertRecommendations(model.songId, model.count).pipe(
        tap(() => this.isLoadingRecommendations.set(false))
      )
    })
  )
  bertRecommendations: Signal<RetrieveResult | undefined> = toSignal(this.bertRecommendations$)

  getMFCCRecommendations: Subject<RetrieveApiModel> = new Subject<RetrieveApiModel>();
  getMFCCRecommendations$ = this.getMFCCRecommendations.asObservable();

  mfccRecommendations$: Observable<RetrieveResult> = this.getMFCCRecommendations$.pipe(
    tap(() => this.isLoadingRecommendations.set(true)),
    switchMap((model) => {
      return this.apiService.getMFCCRecommendations(model.songId, model.count).pipe(
        tap(() => this.isLoadingRecommendations.set(false))
      )
    })
  )
  mfccRecommendations: Signal<RetrieveResult | undefined> = toSignal(this.mfccRecommendations$)

  mfccbowRecommendations$: Observable<RetrieveResult> = this.getMFCCRecommendations$.pipe(
    tap(() => this.isLoadingRecommendations.set(true)),
    switchMap((model) => {
      return this.apiService.getMFCCBOWRecommendations(model.songId, model.count).pipe(
        tap(() => this.isLoadingRecommendations.set(false))
      )
    })
  )
  mfccbowRecommendations: Signal<RetrieveResult | undefined> = toSignal(this.mfccbowRecommendations$)

  mfccstatRecommendations$: Observable<RetrieveResult> = this.getMFCCRecommendations$.pipe(
    tap(() => this.isLoadingRecommendations.set(true)),
    switchMap((model) => {
      return this.apiService.getMFCCSTATRecommendations(model.songId, model.count).pipe(
        tap(() => this.isLoadingRecommendations.set(false))
      )
    })
  )
  mfccstatRecommendations: Signal<RetrieveResult | undefined> = toSignal(this.mfccstatRecommendations$)

  getResNetRecommendations: Subject<RetrieveApiModel> = new Subject<RetrieveApiModel>();
  getResNetRecommendations$ = this.getResNetRecommendations.asObservable();

  resNetRecommendations$: Observable<RetrieveResult> = this.getResNetRecommendations$.pipe(
    tap(() => this.isLoadingRecommendations.set(true)),
    switchMap((model) => {
      return this.apiService.getResNetRecommendations(model.songId, model.count).pipe(
        tap(() => this.isLoadingRecommendations.set(false))
      )
    })
  )
  resNetRecommendations: Signal<RetrieveResult | undefined> = toSignal(this.resNetRecommendations$)

  getVGG19Recommendations: Subject<RetrieveApiModel> = new Subject<RetrieveApiModel>();
  getVGG19Recommendations$ = this.getVGG19Recommendations.asObservable();

  vgg19Recommendations$: Observable<RetrieveResult> = this.getVGG19Recommendations$.pipe(
    tap(() => this.isLoadingRecommendations.set(true)),
    switchMap((model) => {
      return this.apiService.getVGG19Recommendations(model.songId, model.count).pipe(
        tap(() => this.isLoadingRecommendations.set(false))
      )
    })
  )
  vgg19Recommendations: Signal<RetrieveResult | undefined> = toSignal(this.vgg19Recommendations$)

}
