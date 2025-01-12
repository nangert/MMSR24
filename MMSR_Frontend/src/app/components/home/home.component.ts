import {Component, inject, } from '@angular/core';
import {RecommenderService} from '../../services/recommender.service';
import { ReactiveFormsModule} from '@angular/forms';
import {FilterComponent} from '../filter/filter.component';
import {RetrievalResultsComponent} from '../retrieval-results/retrieval-results.component';
import {QuerySongComponent} from "../query-song/query-song.component";

@Component({
  selector: 'app-home',
  imports: [
    ReactiveFormsModule,
    FilterComponent,
    RetrievalResultsComponent,
    QuerySongComponent
  ],
  templateUrl: './home.component.html',
  styleUrl: './home.component.scss',
  standalone: true
})
export class HomeComponent   {
  recommenderService = inject(RecommenderService)
}

