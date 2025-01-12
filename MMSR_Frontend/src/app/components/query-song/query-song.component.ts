import {Component, inject} from '@angular/core';
import {CardModule} from "primeng/card";
import {TagModule} from "primeng/tag";
import {RecommenderService} from "../../services/recommender.service";

@Component({
  selector: 'app-query-song',
  standalone: true,
  imports: [
    CardModule,
    TagModule
  ],
  templateUrl: './query-song.component.html',
  styleUrl: './query-song.component.scss',
})
export class QuerySongComponent {
  recommenderService = inject(RecommenderService)

}
