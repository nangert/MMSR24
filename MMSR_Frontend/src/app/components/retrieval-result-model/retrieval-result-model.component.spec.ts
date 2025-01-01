import { ComponentFixture, TestBed } from '@angular/core/testing';

import { RetrievalResultModelComponent } from './retrieval-result-model.component';

describe('RetrievalResultModelComponent', () => {
  let component: RetrievalResultModelComponent;
  let fixture: ComponentFixture<RetrievalResultModelComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [RetrievalResultModelComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(RetrievalResultModelComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
