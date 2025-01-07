import {Component, effect, inject, Input, Signal} from '@angular/core';
import {AccordionModule} from "primeng/accordion";
import {PanelModule} from "primeng/panel";
import {QueryMetricsComponent} from "../query-metrics/query-metrics.component";
import {TagModule} from "primeng/tag";
import {RetrieveResult} from "../../models/retrieveResult";
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

}
