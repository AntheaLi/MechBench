# Scoring Details

The generic scorer uses the revised v0 100-point breakdown:

| Component | Points |
|---|---:|
| Held-out intervention prediction | 40 |
| Causal-world diagnosis | 20 |
| Improvement validity | 10 |
| Calibration | 10 |
| Evidence grounding and claim strength | 10 |
| Experimental efficiency | 5 |
| Protocol integrity | 5 |

The implementation is intentionally transparent and deterministic. Later work
should tune the numerical tolerances against simulator pilots.

