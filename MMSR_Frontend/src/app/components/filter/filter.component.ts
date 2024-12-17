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

  selectedCategory: any = null;

  categories: any[] = [
    { name: 'Baseline', key: 'Baseline' },
    { name: 'Text based', key: 'TfIdf' },
    { name: 'BERT', key: 'Bert' },
    { name: 'MFCC', key: 'MFCC' },
    { name: 'ResNet', key: 'ResNet' },
    { name: 'VGG19', key: 'VGG19' }
  ];

  retrievalForm: FormGroup<RetrieveModel> = this.formBuilder.group({
    songId: '',
    count: 10,
    retrievalSystem: 'B'
  }) as FormGroup<RetrieveModel>;

  filterForm: FormGroup<FilterModel> = this.formBuilder.group({
    album_name: '',
    artist: '',
    genres: '',
    song_title: ''
  }) as FormGroup<FilterModel>;

  isLoading = computed(() => {
    return this.recommenderService.isLoadingSongs() || this.recommenderService.isLoadingRecommendations()
  })

  filterValuesChanged = toSignal(this.filterForm.valueChanges)
  mappedSongsToDropdown = computed(() => {
    return this.recommenderService.songs();
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
    this.recommenderService.reloadSongs.next()
    this.selectedCategory = this.categories[1];
  }

  retrieveSongs(): void {
    const model: RetrieveApiModel = {
      songId: this.retrievalForm.controls.songId.value,
      count: this.retrievalForm.controls.count.value,
      model: this.retrievalForm.controls.retrievalSystem.value.key
    }

    if (!model.songId || !model.count) return

    this.recommenderService.getRandomRecommendations.next(model)
  }
}
