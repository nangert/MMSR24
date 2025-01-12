import {computed, inject, Injectable, Signal, signal, WritableSignal} from '@angular/core';
import {Observable, of, shareReplay, Subject, switchMap, tap} from 'rxjs';
import {RetrieveResult, Song} from '../models/retrieveResult';
import {RecommenderApiService} from './recommender-api.service';
import {toObservable, toSignal} from '@angular/core/rxjs-interop';
import {RetrieveApiModel} from '../models/retrieveModel';

@Injectable({
  providedIn: 'root'
})
export class RecommenderService {
  private apiService = inject(RecommenderApiService)

  showMetrics: WritableSignal<boolean> = signal(false)

  retrievalResults = computed(() => {
    const list = []

    if (this.baselineRecommendations()) {
      list.push(this.baselineRecommendations());
    }
    if (this.tfIdfRecommendations()) {
      list.push(this.tfIdfRecommendations());
    }
    if (this.bertRecommendations()) {
      list.push(this.bertRecommendations());
    }
    if (this.mfccbowRecommendations()) {
      list.push(this.mfccbowRecommendations());
    }
    if (this.mfccbowCosRecommendations()) {
      list.push(this.mfccbowCosRecommendations());
    }
    if (this.mfccstatRecommendations()) {
      list.push(this.mfccstatRecommendations());
    }
    if (this.mfccstatCosRecommendations()) {
      list.push(this.mfccstatCosRecommendations());
    }
    if (this.resNetRecommendations()) {
      list.push(this.resNetRecommendations());
    }
    if (this.vgg19Recommendations()) {
      list.push(this.vgg19Recommendations());
    }
    if (this.lamdaMARTRecommendations()) {
      list.push(this.lamdaMARTRecommendations());
    }

    return list
  })

  retrievalResults$ = toObservable(this.retrievalResults)

  isLoadingRecommendations: WritableSignal<boolean> = signal(false)
  querySong: WritableSignal<Song | undefined> = signal(undefined)

  getBaselineRecommendations: Subject<RetrieveApiModel | void> = new Subject<RetrieveApiModel | void>();
  getBaselineRecommendations$ = this.getBaselineRecommendations.asObservable();

  baselineRecommendations$: Observable<RetrieveResult | undefined> = this.getBaselineRecommendations$.pipe(
    switchMap((model) => {
      if (!model) return of(void 0)

      this.isLoadingRecommendations.set(true)

      return this.apiService.getBaselineRecommendations(model.songId, model.count, model.diversity).pipe(
        tap((res) => {
          this.querySong.set(res.query_song)
          this.isLoadingRecommendations.set(false)
        })
      )
    }),
    shareReplay({
      bufferSize: 1,
      refCount: true
    })
  )
  baselineRecommendations: Signal<RetrieveResult | undefined> = toSignal(this.baselineRecommendations$)

  getTfIdfRecommendations: Subject<RetrieveApiModel | void> = new Subject<RetrieveApiModel | void>();
  getTfIdfRecommendations$ = this.getTfIdfRecommendations.asObservable();

  tfIdfRecommendations$: Observable<RetrieveResult | undefined> = this.getTfIdfRecommendations$.pipe(
    switchMap((model) => {
      if (!model) return of(void 0)

      this.isLoadingRecommendations.set(true)

      return this.apiService.getTfIdfRecommendations(model.songId, model.count, model.diversity).pipe(
        tap((res) => {
          this.querySong.set(res.query_song)
          this.isLoadingRecommendations.set(false)
        })
      )
    }),
    shareReplay({
      bufferSize: 1,
      refCount: true
    })
  )
  tfIdfRecommendations: Signal<RetrieveResult | undefined> = toSignal(this.tfIdfRecommendations$)

  getBertRecommendations: Subject<RetrieveApiModel | void> = new Subject<RetrieveApiModel | void>();
  getBertRecommendations$ = this.getBertRecommendations.asObservable();

  bertRecommendations$: Observable<RetrieveResult | undefined> = this.getBertRecommendations$.pipe(
    switchMap((model) => {
      if (!model) return of(void 0)

      this.isLoadingRecommendations.set(true)

      return this.apiService.getBertRecommendations(model.songId, model.count, model.diversity).pipe(
        tap((res) => {
          this.querySong.set(res.query_song)
          this.isLoadingRecommendations.set(false)
        })
      )
    }),
    shareReplay({
      bufferSize: 1,
      refCount: true
    })
  )
  bertRecommendations: Signal<RetrieveResult | undefined> = toSignal(this.bertRecommendations$)

  getMFCCRecommendations: Subject<RetrieveApiModel | void> = new Subject<RetrieveApiModel | void>();
  getMFCCRecommendations$ = this.getMFCCRecommendations.asObservable();

  mfccbowRecommendations$: Observable<RetrieveResult | undefined> = this.getMFCCRecommendations$.pipe(
    switchMap((model) => {
      if (!model) return of(void 0)

      this.isLoadingRecommendations.set(true)

      return this.apiService.getMFCCBOWRecommendations(model.songId, model.count, model.diversity).pipe(
        tap((res) => {
          this.querySong.set(res.query_song)
          this.isLoadingRecommendations.set(false)
        })
      )
    }),
    shareReplay({
      bufferSize: 1,
      refCount: true
    })
  )
  mfccbowRecommendations: Signal<RetrieveResult | undefined> = toSignal(this.mfccbowRecommendations$)

  mfccbowCosRecommendations$: Observable<RetrieveResult | undefined> = this.getMFCCRecommendations$.pipe(
    switchMap((model) => {
      if (!model) return of(void 0)

      this.isLoadingRecommendations.set(true)

      return this.apiService.getMFCCBOWCosRecommendations(model.songId, model.count, model.diversity).pipe(
        tap((res) => {
          this.querySong.set(res.query_song)
          this.isLoadingRecommendations.set(false)
        })
      )
    }),
    shareReplay({
      bufferSize: 1,
      refCount: true
    })
  )
  mfccbowCosRecommendations: Signal<RetrieveResult | undefined> = toSignal(this.mfccbowCosRecommendations$)

  mfccstatRecommendations$: Observable<RetrieveResult | undefined> = this.getMFCCRecommendations$.pipe(
    switchMap((model) => {
      if (!model) return of(void 0)

      this.isLoadingRecommendations.set(true)

      return this.apiService.getMFCCSTATRecommendations(model.songId, model.count, model.diversity).pipe(
        tap((res) => {
          this.querySong.set(res.query_song)
          this.isLoadingRecommendations.set(false)
        })
      )
    }),
    shareReplay({
      bufferSize: 1,
      refCount: true
    })
  )
  mfccstatRecommendations: Signal<RetrieveResult | undefined> = toSignal(this.mfccstatRecommendations$)

  mfccstatCosRecommendations$: Observable<RetrieveResult | undefined> = this.getMFCCRecommendations$.pipe(
    switchMap((model) => {
      if (!model) return of(void 0)

      this.isLoadingRecommendations.set(true)

      return this.apiService.getMFCCSTATCosRecommendations(model.songId, model.count, model.diversity).pipe(
        tap((res) => {
          this.querySong.set(res.query_song)
          this.isLoadingRecommendations.set(false)
        })
      )
    }),
    shareReplay({
      bufferSize: 1,
      refCount: true
    })
  )
  mfccstatCosRecommendations: Signal<RetrieveResult | undefined> = toSignal(this.mfccstatCosRecommendations$)

  getResNetRecommendations: Subject<RetrieveApiModel | void> = new Subject<RetrieveApiModel | void>();
  getResNetRecommendations$ = this.getResNetRecommendations.asObservable();

  resNetRecommendations$: Observable<RetrieveResult | undefined> = this.getResNetRecommendations$.pipe(
    switchMap((model) => {
      if (!model) return of(void 0)

      this.isLoadingRecommendations.set(true)

      return this.apiService.getResNetRecommendations(model.songId, model.count, model.diversity).pipe(
        tap((res) => {
          this.querySong.set(res.query_song)
          this.isLoadingRecommendations.set(false)
        })
      )
    }),
    shareReplay({
      bufferSize: 1,
      refCount: true
    })
  )
  resNetRecommendations: Signal<RetrieveResult | undefined> = toSignal(this.resNetRecommendations$)

  getVGG19Recommendations: Subject<RetrieveApiModel | void> = new Subject<RetrieveApiModel | void>();
  getVGG19Recommendations$ = this.getVGG19Recommendations.asObservable();

  vgg19Recommendations$: Observable<RetrieveResult | undefined> = this.getVGG19Recommendations$.pipe(
    switchMap((model) => {
      if (!model) return of(void 0)

      this.isLoadingRecommendations.set(true)

      return this.apiService.getVGG19Recommendations(model.songId, model.count, model.diversity).pipe(
        tap((res) => {
          this.querySong.set(res.query_song)
          this.isLoadingRecommendations.set(false)
        })
      )
    }),
    shareReplay({
      bufferSize: 1,
      refCount: true
    })
  )
  vgg19Recommendations: Signal<RetrieveResult | undefined> = toSignal(this.vgg19Recommendations$)

  getLamdaMARTRecommendations: Subject<RetrieveApiModel | void> = new Subject<RetrieveApiModel | void>();
  getLamdaMARTRecommendations$ = this.getLamdaMARTRecommendations.asObservable();

  lamdaMARTRecommendations$: Observable<RetrieveResult | undefined> = this.getLamdaMARTRecommendations$.pipe(
    switchMap((model) => {
      if (!model) return of(void 0)

      this.isLoadingRecommendations.set(true)

      return this.apiService.getLamdaMARTRecommendations(model.songId, model.count, model.diversity).pipe(
        tap((res) => {
          this.querySong.set(res.query_song)
          this.isLoadingRecommendations.set(false)
        })
      )
    }),
    shareReplay({
      bufferSize: 1,
      refCount: true
    })
  )
  lamdaMARTRecommendations: Signal<RetrieveResult | undefined> = toSignal(this.lamdaMARTRecommendations$)

  resetRecommendations(): void {
    this.querySong.set(undefined)
    this.getBaselineRecommendations.next(void 0)
    this.getTfIdfRecommendations.next(void 0)
    this.getBertRecommendations.next(void 0)
    this.getMFCCRecommendations.next(void 0)
    this.getResNetRecommendations.next(void 0)
    this.getVGG19Recommendations.next(void 0)
    this.getLamdaMARTRecommendations.next(void 0)
  }

}
