import { ComponentFixture, TestBed } from '@angular/core/testing';

import { QuerySongComponent } from './query-song.component';

describe('QuerySongComponent', () => {
  let component: QuerySongComponent;
  let fixture: ComponentFixture<QuerySongComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [QuerySongComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(QuerySongComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
