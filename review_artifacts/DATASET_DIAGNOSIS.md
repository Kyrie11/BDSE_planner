# Dataset Diagnosis

## Validation

- Loaded: 58418
- Full-interface action match: 0.965661
- B16 oracle decision sufficiency: 0.912013
- Runtime decision sufficiency: 0.749033
- Evidence sufficiency: 0.592157
- Safe candidate exists: 0.717296 (fails 0.75)
- Teacher candidate ADE p50/p90: 5.480/12.915

The evidence interface is viable on validation, but candidate-bank safety and trajectory coverage are already a ceiling for closed-loop performance.

## Partial test

- Loaded: 67042 (incomplete)
- Full-interface action match: 0.934280
- B16 oracle decision sufficiency: 0.839921
- Runtime decision sufficiency: 0.640733
- Safe candidate exists: 0.577862
- Teacher candidate ADE p50/p90: 4.956/17.141
- Route-distance p90 diagnostic: 17.550

Use this split only as a frozen-checkpoint stress test. Do not tune on it; rebuild and re-audit the complete test set for final publication results.
