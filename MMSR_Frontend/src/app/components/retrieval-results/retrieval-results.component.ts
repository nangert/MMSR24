import {Component, computed, inject} from '@angular/core';
import {RecommenderService} from '../../services/recommender.service';
import {AccordionModule} from "primeng/accordion";
import {Song} from "../../models/retrieveResult";
import {PanelModule} from "primeng/panel";
import {RetrievedSongComponent} from "../retrieved-song/retrieved-song.component";
import {Button} from "primeng/button";
import {QueryMetricsComponent} from "../query-metrics/query-metrics.component";

@Component({
  selector: 'app-retrieval-results',
  imports: [
    AccordionModule,
    PanelModule,
    RetrievedSongComponent,
    Button,
    QueryMetricsComponent
  ],
  templateUrl: './retrieval-results.component.html',
  styleUrl: './retrieval-results.component.scss',
  standalone: true
})
export class RetrievalResultsComponent {
  recommenderService = inject(RecommenderService)

  isLoading = computed(() => {
    return this.recommenderService.isLoadingSongs() || this.recommenderService.isLoadingRecommendations()
  })

  getQueryMetrics(): void {
    const body = this.recommenderService.randomRecommendation()
    if (body) {
      console.log(body)
      this.recommenderService.getQueryMetrics.next(body)
    }
  }


}
