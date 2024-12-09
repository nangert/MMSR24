import {Component, inject, Input} from '@angular/core';
import {AccordionModule} from "primeng/accordion";
import {Song} from "../../models/retrieveResult";
import {RecommenderService} from "../../services/recommender.service";
import {DomSanitizer, SafeResourceUrl} from "@angular/platform-browser";
import {RecommenderApiService} from "../../services/recommender-api.service";
import {NgxLiteYoutubeModule} from "ngx-lite-video";
import {Button} from "primeng/button";

@Component({
  selector: 'app-retrieved-song',
  standalone: true,
  imports: [
    AccordionModule,
    NgxLiteYoutubeModule,
    Button
  ],
  templateUrl: './retrieved-song.component.html',
  styleUrl: './retrieved-song.component.scss',
})
export class RetrievedSongComponent {
  @Input() song!: Song;

  recommenderService = inject(RecommenderService)
  apiService = inject(RecommenderApiService)
  sanitizer = inject(DomSanitizer)

  getAccordionTitle(): string {
    return this.song.artist + ' - ' + this.song.song_title + ' (' + this.song.album_name + ')'
  }

  getSpotifyId(): string {
    return this.song?.spotify_id ?? 'no spotify id'
  }

  getSafeUrl(url: string): SafeResourceUrl {
    if (!url) return '';
    const videoId = this.extractVideoId(url);

    if (!videoId) return '';

    // Create the embed URL
    const embedUrl = `https://www.youtube.com/embed/${videoId}`;
    return this.sanitizer.bypassSecurityTrustResourceUrl(embedUrl);
  }

  extractVideoId(url: string): string | null {
    const match = url.match(/(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([^&]+)/)
      || url.match(/(?:https?:\/\/)?(?:www\.)?youtu\.be\/([^?&]+)/);
    return match ? match[1] : null;
  }

}
