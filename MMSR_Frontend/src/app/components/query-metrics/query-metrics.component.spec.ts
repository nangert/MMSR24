import { ComponentFixture, TestBed } from '@angular/core/testing';

import { QueryMetricsComponent } from './query-metrics.component';

describe('QueryMetricsComponent', () => {
  let component: QueryMetricsComponent;
  let fixture: ComponentFixture<QueryMetricsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [QueryMetricsComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(QueryMetricsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
