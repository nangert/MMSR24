import {Component, computed, inject, OnInit} from '@angular/core';
import {FormBuilder, FormGroup, FormsModule, ReactiveFormsModule} from '@angular/forms';
import {RetrieveApiModel, RetrieveModel} from '../../models/retrieveModel';
import {RecommenderService} from '../../services/recommender.service';
import {DropdownModule} from "primeng/dropdown";
import {InputGroupModule} from "primeng/inputgroup";
import {InputTextModule} from "primeng/inputtext";
import {FilterModel} from "../../models/filter.model";
import {toSignal} from "@angular/core/rxjs-interop";
import {RadioButtonModule} from "primeng/radiobutton";
import {MetricsService} from "../../services/metrics.service";
import {DataService} from "../../services/data.service";

@Component({
  selector: 'app-filter',
  imports: [
    FormsModule,
    ReactiveFormsModule,
    DropdownModule,
    InputGroupModule,
    InputTextModule,
    RadioButtonModule
  ],
  templateUrl: './filter.component.html',
  styleUrl: './filter.component.scss',
  standalone: true
})
export class FilterComponent implements OnInit{
  private formBuilder = inject(FormBuilder);
  recommenderService = inject(RecommenderService)
  metricService = inject(MetricsService)
  dataService = inject(DataService)

  selectedCategory: any = null;
  selectedRelevance: any = null;

  categories: any[] = [
    { name: 'Baseline', key: 'Baseline' },
    { name: 'Text based', key: 'TfIdf' },
    { name: 'BERT', key: 'Bert' },
    { name: 'MFCC', key: 'MFCC' },
    { name: 'ResNet', key: 'ResNet' },
    { name: 'VGG19', key: 'VGG19' }
  ];

  relevance: any[] = [
    { name: 'Top genre weights', key: 'Top' },
    { name: 'Any genre match', key: 'Any' }
  ];

  retrievalForm: FormGroup<RetrieveModel> = this.formBuilder.group({
    songId: '',
    count: 10,
    retrievalSystem: 'Baseline',
    relevanceSystem: 'Top'
  }) as FormGroup<RetrieveModel>;

  filterForm: FormGroup<FilterModel> = this.formBuilder.group({
    album_name: '',
    artist: '',
    genres: '',
    song_title: ''
  }) as FormGroup<FilterModel>;

  isLoading = computed(() => {
    return this.dataService.isLoadingSongs() || this.recommenderService.isLoadingRecommendations()
  })

  filterValuesChanged = toSignal(this.filterForm.valueChanges)
  mappedSongsToDropdown = computed(() => {
    return this.dataService.songs();
  })
  dropdownValues = computed(() => {
    this.filterValuesChanged()
    const filtered =  this.mappedSongsToDropdown().filter(song => {
      return song.artist.toLowerCase().includes(this.filterForm.controls.artist.value.toLowerCase()) &&
       song.album_name.toLowerCase().includes(this.filterForm.controls.album_name.value.toLowerCase()) &&
       song.song_title.toLowerCase().includes(this.filterForm.controls.song_title.value.toLowerCase())
    })

    return filtered.map(song => {
      return {
        name: song.song_title + ' by ' + song.artist,
        value: song.song_id
      };
    })
  })

  ngOnInit(): void {
    this.dataService.reloadSongs.next()
    this.selectedCategory = this.categories[0];
    this.selectedRelevance = this.relevance[0];
  }

  retrieveSongs(): void {
    const model: RetrieveApiModel = {
      songId: this.retrievalForm.controls.songId.value,
      count: this.retrievalForm.controls.count.value
    }

    this.metricService.relevanceMeasure = this.retrievalForm.controls.relevanceSystem.value.key

    if (!model.songId || !model.count) return

    switch (this.retrievalForm.controls.retrievalSystem.value.key) {
      case 'Baseline':
        this.recommenderService.getBaselineRecommendations.next(model)
        break
      case 'TfIdf':
        this.recommenderService.getTfIdfRecommendations.next(model)
        break
      case 'Bert':
        this.recommenderService.getBertRecommendations.next(model)
        break
      case 'MFCC':
        this.recommenderService.getMFCCRecommendations.next(model)
        break
      case 'ResNet':
        this.recommenderService.getResNetRecommendations.next(model)
        break
      case 'VGG19':
        this.recommenderService.getVGG19Recommendations.next(model)
        break
      default:
        this.recommenderService.getBaselineRecommendations.next(model)
    }

  }

  clearResults(): void {
    this.recommenderService.getBaselineRecommendations.next(void 0)
  }
}
