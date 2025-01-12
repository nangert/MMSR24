import {Component, computed, effect, inject, Input, OnChanges, signal, Signal, WritableSignal} from '@angular/core';
import {AccordionModule} from "primeng/accordion";
import {PanelModule} from "primeng/panel";
import {QueryMetricsComponent} from "../query-metrics/query-metrics.component";
import {TagModule} from "primeng/tag";
import {RetrieveResult, Song} from "../../models/retrieveResult";
import {QueryMetrics} from "../../models/retrieveModel";
import {CardModule} from "primeng/card";
import {TableModule} from "primeng/table";
import {RecommenderService} from "../../services/recommender.service";
import {Button} from "primeng/button";
import {tap} from "rxjs";

@Component({
  selector: 'app-retrieval-result-model',
  standalone: true,
  imports: [
    AccordionModule,
    PanelModule,
    QueryMetricsComponent,
    TagModule,
    CardModule,
    TableModule,
    Button
  ],
  templateUrl: './retrieval-result-model.component.html',
  styleUrl: './retrieval-result-model.component.scss'
})
export class RetrievalResultModelComponent {
  @Input() retrievalResult!: RetrieveResult
  @Input() queryResult!: Signal<QueryMetrics | undefined>
  @Input() retrievalSystem!: string

  recommenderService = inject(RecommenderService)

  sharedSongs: WritableSignal<Set<string>> = signal(new Set())

  constructor() {
    effect(() => {
      const songMap = new Map<string, number>();

      this.recommenderService.retrievalResults().forEach(result => {
        if (!result) return;

        result.result_songs.forEach(song => {
          const key = this.getNormalizedSongKey(song);
          songMap.set(key, (songMap.get(key) || 0) + 1);
        });
      });

      const newShared = new Set(
        Array.from(songMap.entries())
          .filter(([_, count]) => count > 1)
          .map(([key]) => key)
      );

      this.sharedSongs.set(newShared)
    }, {allowSignalWrites: true});
  }

  checkIfGenreMatch(genre: any): boolean {
    const querySong = this.recommenderService.querySong()

    if (!querySong) return false

    return querySong.genres.includes(genre)
  }

  private getNormalizedSongKey(song: Song): string {
    return `${song.song_title?.trim().toLowerCase()}-${song.artist?.trim().toLowerCase()}`;
  }

  isShared(song: Song): boolean {
    return this.sharedSongs().has(this.getNormalizedSongKey(song));
  }
}
