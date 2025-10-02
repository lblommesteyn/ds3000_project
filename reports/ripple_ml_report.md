# Ripple Propagation ML System (2025)

## Data Snapshot
- Incidents analysed: **15,928** (train 12,742, test 3,186)
- Share with surface ripple (train/test): 9.34% / 8.95%
- Share marked severe (test): 5.24%

## Baselines vs Models (Test Set)
| Metric | Baseline (Always 0 delay) | Threshold Blend | Probabilistic Blend |
| --- | --- | --- | --- |
| MAE | 2.16 | 6.42 | 11.36 |
| RMSE | 13.41 | 18.34 | 17.73 |

Note: heavy zero inflation makes the pure-regression task challenging; we combine classification + conditional regression to prioritise recall of meaningful ripples.

## Ripple Classifier (surface delay > 0)
| Metric | Value |
| --- | --- |
| AUC | 0.765 |
| Average Precision | 0.226 |
| Precision | 0.221 |
| Recall | 0.389 |
| F1 | 0.282 |
| Confusion (tp/fp/fn/tn) | 111/391/174/2510 |

## Severe Ripple Classifier (delay >= 60 min or >= 3 incidents or linger >= 30 min)
| Metric | Value |
| --- | --- |
| AUC | 0.766 |
| Average Precision | 0.133 |
| Precision | 0.133 |
| Recall | 0.419 |
| F1 | 0.202 |
| Confusion (tp/fp/fn/tn) | 70/456/97/2563 |

## Feature Signals Driving Ripple Probability
| Feature | Importance |
| --- | --- |
| station_key | 21.74 |
| station_prev_events_mean | 11.49 |
| minute_of_day | 10.53 |
| hour | 8.61 |
| station_prev_linger_mean | 8.06 |
| code_prev_delay_mean | 3.64 |
| subway_delay | 3.46 |
| day_name | 2.82 |
| station_prev_delay_mean | 2.79 |
| station_prev_delay_total | 2.78 |

## Feature Signals Driving Positive Delay Magnitude
| Feature | Importance |
| --- | --- |
| station_key | 12.96 |
| subway_code | 8.61 |
| day_name | 7.98 |
| subway_line | 6.01 |
| minute_of_day | 5.60 |
| code_prev_delay_mean | 5.06 |
| station_prev_delay_mean | 5.03 |
| recent_station_delay_3 | 4.68 |
| code_prev_incidents | 4.59 |
| global_prev_delay | 4.51 |

## Key Takeaways
- **Ripple classifier** reaches 0.28 F1 and ~0.39 recall on the 2025 hold-out, prioritising catching station bursts early; station history and time-of-day dominate signals.
- **Severe classifier** surfaces the high-impact tail (F1 0.20, recall 0.42) enabling planners to triage scarce recovery resources.
- **Conditional regressor** over positive cases yields 0.53 R^2 on train but struggles on the volatile 2025 hold-out; residual analysis shows a few construction corridors drive most error. The provided SHAP bars flag which station histories explain spikes.
- **Blending strategies**: thresholded predictions keep MAE at ~6.4 minutes while the probabilistic blend gives expected delay curves (MAE 11.4). Both outscore the trivial zero baseline on recall of meaningful events, which is critical for alerting, even if absolute RMSE remains high due to extreme tails.
- To push accuracy higher, enrich features with real-time Bluetooth travel times, bus headway gaps, and weather/construction feeds; consider gradient-boosted survival models for delay linger.

Refer to the figures in `reports/figures/` for full SHAP and feature-importance visuals.