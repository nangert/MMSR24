import { ComponentFixture, TestBed } from '@angular/core/testing';

import { RetrievalResultsComponent } from './retrieval-results.component';

describe('RetrievalResultsComponent', () => {
  let component: RetrievalResultsComponent;
  let fixture: ComponentFixture<RetrievalResultsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [RetrievalResultsComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(RetrievalResultsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
