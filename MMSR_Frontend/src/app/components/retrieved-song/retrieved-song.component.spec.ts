import { ComponentFixture, TestBed } from '@angular/core/testing';

import { RetrievedSongComponent } from './retrieved-song.component';

describe('RetrievedSongComponent', () => {
  let component: RetrievedSongComponent;
  let fixture: ComponentFixture<RetrievedSongComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [RetrievedSongComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(RetrievedSongComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
