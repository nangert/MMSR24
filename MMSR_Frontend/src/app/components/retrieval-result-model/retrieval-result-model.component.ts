import {Component, computed, effect, inject, Input, Signal} from '@angular/core';
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

  checkIfGenreMatch(genre: any): boolean {
    const querySong = this.recommenderService.querySong()

    if (!querySong) return false

    return querySong.genres.includes(genre)
  }




  sharedSongs: Set<string> = new Set();

  ngOnChanges(): void {
    // Collect all retrieved songs across systems
    const songMap = new Map<string, number>();

    this.recommenderService.retrievalResults().forEach(result => {
      if (!result) return

      result.result_songs.forEach(song => {
        const key = this.getSongKey(song);
        songMap.set(key, (songMap.get(key) || 0) + 1);
      });
    });

    // Identify shared songs
    this.sharedSongs = new Set(
      Array.from(songMap.entries())
        .filter(([_, count]) => count > 1) // Songs appearing in more than one system
        .map(([key]) => key)
    );
  }

  private getSongKey(song: Song): string {
    // Generate a unique key for each song, e.g., using title and artist
    return `${song.song_title}-${song.artist}`;
  }

  isShared(song: Song): boolean {
    return this.sharedSongs.has(this.getSongKey(song));
  }

}
