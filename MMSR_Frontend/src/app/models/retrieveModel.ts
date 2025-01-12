import {FormControl} from '@angular/forms';

export interface RetrieveModel {
  songId: FormControl<string>,
  count: FormControl<number>,
  retrievalSystem: FormControl<any>,
  diversity: FormControl<boolean>,
}

export interface RetrieveApiModel {
  songId: string,
  count: number,
  diversity: boolean,
}

export interface QueryMetrics {
  mrr: string
  ndcg_at_k: string
  precision_at_k: string
  recall_at_k: string
}
