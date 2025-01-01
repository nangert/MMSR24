import {Component, effect, Input, Signal} from '@angular/core';
import {AccordionModule} from "primeng/accordion";
import {PanelModule} from "primeng/panel";
import {QueryMetricsComponent} from "../query-metrics/query-metrics.component";
import {RetrievedSongComponent} from "../retrieved-song/retrieved-song.component";
import {TagModule} from "primeng/tag";
import {RetrieveResult} from "../../models/retrieveResult";
import {QueryMetrics} from "../../models/retrieveModel";

@Component({
  selector: 'app-retrieval-result-model',
  standalone: true,
  imports: [
    AccordionModule,
    PanelModule,
    QueryMetricsComponent,
    RetrievedSongComponent,
    TagModule
  ],
  templateUrl: './retrieval-result-model.component.html',
  styleUrl: './retrieval-result-model.component.scss'
})
export class RetrievalResultModelComponent {
  @Input() retrievalResult!: RetrieveResult
  @Input() queryResult!: Signal<QueryMetrics | undefined>

}
