import {Component, inject, } from '@angular/core';
import {RecommenderService} from '../../services/recommender.service';
import { ReactiveFormsModule} from '@angular/forms';
import {FilterComponent} from '../filter/filter.component';
import {RetrievalResultsComponent} from '../retrieval-results/retrieval-results.component';

@Component({
  selector: 'app-home',
  imports: [
    ReactiveFormsModule,
    FilterComponent,
    RetrievalResultsComponent
  ],
  templateUrl: './home.component.html',
  styleUrl: './home.component.scss',
  standalone: true
})
export class HomeComponent   {
  private recommenderService = inject(RecommenderService)
}
