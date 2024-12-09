import {Component, inject} from '@angular/core';
import {PanelModule} from "primeng/panel";
import {RecommenderService} from "../../services/recommender.service";

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
  recommenderService = inject(RecommenderService)

}
