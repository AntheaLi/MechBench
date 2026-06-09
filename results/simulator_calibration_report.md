# MechanismBench Calibration Report

## Summary

- World count: 100
- Public-only world classifier accuracy: 0.170
- Gates passing: 6/6

## Gates

| Gate | Status |
|---|---|
| `at_least_100_certified_simulator_episodes` | PASS |
| `naive_score_at_most_45` | PASS |
| `oracle_score_at_least_92` | PASS |
| `public_only_classifier_at_most_35` | PASS |
| `random_score_at_most_35` | PASS |
| `scripted_score_65_to_80` | PASS |

## Agent Scores

| Agent | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| naive | 42.129 | 42.684 | 21.086 | 66.210 |
| oracle | 98.800 | 98.835 | 98.293 | 98.960 |
| random | 22.571 | 19.717 | 17.693 | 32.840 |
| scripted | 77.843 | 78.478 | 69.909 | 82.659 |

## Components

| Agent | calibration | causal_world_diagnosis | evidence_grounding | experimental_efficiency | heldout_intervention_prediction | improvement_validity | protocol_integrity |
|---|---:|---:|---:|---:|---:|---:|---:|
| naive | 2.625 | 3.750 | 3.500 | 5.000 | 14.195 | 8.059 | 5.000 |
| oracle | 9.960 | 20.000 | 10.000 | 4.840 | 39.000 | 10.000 | 5.000 |
| random | 0.000 | 0.000 | 5.000 | 4.762 | 6.065 | 1.744 | 5.000 |
| scripted | 7.718 | 19.200 | 10.000 | 4.216 | 21.782 | 9.928 | 5.000 |

## World Types

### naive

| World Type | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| compute_laundering_lite | 37.086 | 37.511 | 34.235 | 39.093 |
| norm_laundering | 44.369 | 44.600 | 38.450 | 47.381 |
| parameter_laundering | 43.813 | 44.407 | 39.294 | 48.404 |
| seed_hacking | 22.136 | 22.130 | 21.086 | 23.522 |
| true_mechanism | 63.242 | 63.003 | 61.007 | 66.210 |

### oracle

| World Type | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| compute_laundering_lite | 98.830 | 98.910 | 98.529 | 98.960 |
| norm_laundering | 98.848 | 98.924 | 98.568 | 98.960 |
| parameter_laundering | 98.762 | 98.758 | 98.529 | 98.960 |
| seed_hacking | 98.690 | 98.779 | 98.293 | 98.960 |
| true_mechanism | 98.868 | 98.960 | 98.559 | 98.960 |

### random

| World Type | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| compute_laundering_lite | 24.000 | 24.039 | 23.239 | 24.641 |
| norm_laundering | 19.393 | 19.350 | 18.865 | 19.947 |
| parameter_laundering | 19.421 | 19.412 | 18.750 | 20.308 |
| seed_hacking | 31.282 | 31.614 | 27.852 | 32.840 |
| true_mechanism | 18.758 | 18.649 | 17.693 | 20.007 |

### scripted

| World Type | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| compute_laundering_lite | 71.769 | 71.841 | 69.909 | 73.658 |
| norm_laundering | 81.274 | 81.454 | 80.106 | 82.659 |
| parameter_laundering | 78.150 | 77.710 | 76.707 | 80.506 |
| seed_hacking | 77.051 | 77.060 | 75.155 | 79.627 |
| true_mechanism | 80.972 | 80.862 | 79.652 | 82.300 |

## Weakest Episodes

| World | Label | naive | oracle | random | scripted |
|---|---|---:|---:|---:|---:|
| simulator/compiled_0041 | true_mechanism | 61.320 | 98.570 | 17.690 | 79.650 |
| simulator/compiled_0056 | true_mechanism | 62.110 | 98.690 | 18.080 | 80.060 |
| simulator/compiled_0096 | true_mechanism | 64.210 | 98.960 | 18.100 | 80.400 |
| simulator/compiled_0026 | true_mechanism | 61.010 | 98.910 | 18.190 | 80.410 |
| simulator/compiled_0091 | true_mechanism | 61.840 | 98.920 | 18.290 | 80.390 |
| simulator/compiled_0036 | true_mechanism | 62.700 | 98.960 | 18.300 | 80.550 |
| simulator/compiled_0071 | true_mechanism | 63.220 | 98.620 | 18.300 | 80.270 |
| simulator/compiled_0016 | true_mechanism | 61.840 | 98.960 | 18.320 | 80.840 |
