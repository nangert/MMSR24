import {Component, inject, Input} from '@angular/core';
import {PanelModule} from "primeng/panel";
import {RecommenderService} from "../../services/recommender.service";
import {MetricsService} from "../../services/metrics.service";
import {QueryMetrics} from "../../models/retrieveModel";

@Component({
  selector: 'app-query-metrics',
  standalone: true,
  imports: [
    PanelModule
  ],
  templateUrl: './query-metrics.component.html',
  styleUrl: './query-metrics.component.scss'
})
export class QueryMetricsComponent {
  @Input() queryMetrics!: QueryMetrics

}
