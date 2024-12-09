import {FormControl} from "@angular/forms";

export interface FilterModel {
  album_name: FormControl<string>,
  artist: FormControl<string>,
  genres: FormControl<string>,
  song_title: FormControl<string>,
}
