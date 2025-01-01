import {inject, Injectable, signal, WritableSignal} from '@angular/core';
import {catchError, Observable, of, Subject, switchMap, tap} from "rxjs";
import {Song} from "../models/retrieveResult";
import {toSignal} from "@angular/core/rxjs-interop";
import {RecommenderApiService} from "./recommender-api.service";

@Injectable({
  providedIn: 'root'
})
export class DataService {
  private apiService = inject(RecommenderApiService)

  reloadSongs: Subject<void> = new Subject<void>();
  reloadSongs$ = this.reloadSongs.asObservable();
  isLoadingSongs: WritableSignal<boolean> = signal(false)

  songs$: Observable<Song[]> = this.reloadSongs$.pipe(
    tap(() => this.isLoadingSongs.set(true)),
    switchMap(() => {
      return this.apiService.retrieveSongs().pipe(
        tap(() => this.isLoadingSongs.set(false)),
        catchError(() => {
          this.isLoadingSongs.set(false)
          return of([])
        })
      )
    })
  )
  songs = toSignal(this.songs$, { initialValue: []})
}
