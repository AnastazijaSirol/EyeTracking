# Rezultati neovisne evaluacije

Kutne pogreške izražene su u stupnjevima. Off-center definicije koriste prag 0,3.

## Glavna usporedba

| Konfiguracija | Seed | Kutna pogreška (°) | Medijan (°) | P95 (°) | Off-center osno (°) | Off-center radijalno (°) | On-center osno (°) | FPS GPU model | FPS GPU ukupno | Ispitanici | Uzorci |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mtl_full | 42 | 3.17 ± 2.17 | 2.71 | 7.42 | 5.47 | 5.40 | 3.11 | 77.9 | 74.1 | 11 | 79964 |
| mtl_full | 123 | 3.37 ± 2.16 | 2.93 | 7.27 | 5.04 | 4.96 | 3.32 | — | — | 11 | 79964 |
| mtl_full | 2024 | 3.16 ± 2.14 | 2.79 | 7.44 | 5.06 | 5.08 | 3.11 | — | — | 11 | 79964 |
| mtl_wseg0 | 42 | 2.96 ± 2.44 | 2.34 | 7.86 | 5.15 | 5.12 | 2.89 | — | — | 11 | 79964 |
| mtl_wseg0 | 123 | 3.18 ± 2.32 | 2.60 | 7.62 | 5.11 | 5.05 | 3.13 | — | — | 11 | 79964 |
| mtl_wseg0 | 2024 | 2.60 ± 1.96 | 2.08 | 6.41 | 4.68 | 4.60 | 2.54 | — | — | 11 | 79964 |
| mtl_pupil | 42 | 3.05 ± 2.25 | 2.59 | 7.66 | 5.21 | 5.27 | 2.98 | — | — | 11 | 79964 |
| mtl_pupil | 123 | 3.24 ± 2.24 | 2.81 | 7.65 | 4.87 | 4.82 | 3.20 | — | — | 11 | 79964 |
| mtl_pupil | 2024 | 2.94 ± 2.01 | 2.38 | 6.96 | 4.61 | 4.53 | 2.89 | — | — | 11 | 79964 |
| gaze_only | 42 | 6.47 ± 4.89 | 5.43 | 16.53 | 23.37 | 21.77 | 5.99 | 99.3 | 84.4 | 11 | 79964 |
| gaze_only | 123 | 6.47 ± 4.88 | 5.33 | 16.67 | 23.31 | 21.69 | 5.98 | — | — | 11 | 79964 |
| gaze_only | 2024 | 6.49 ± 4.89 | 5.41 | 16.63 | 23.37 | 21.76 | 6.01 | — | — | 11 | 79964 |
| gaze_only | 7 | 6.30 ± 4.85 | 5.23 | 16.27 | 23.14 | 21.55 | 5.82 | — | — | 11 | 79964 |
| gaze_only | 13 | 6.51 ± 4.90 | 5.44 | 16.63 | 23.40 | 21.79 | 6.02 | — | — | 11 | 79964 |
| trivial_baseline/global | none | 6.18 ± 4.81 | 5.06 | 16.06 | 22.87 | 21.30 | 5.70 | — | — | 11 | 79964 |

## Prosjek preko seedova

| Konfiguracija | Srednja kutna pogreška preko seedova (°) |
| --- | --- |
| gaze_only | 6.45 ± 0.08 |
| mtl_full | 3.23 ± 0.12 |
| mtl_pupil | 3.08 ± 0.15 |
| mtl_wseg0 | 2.91 ± 0.29 |

## Per-subject trivijalni baseline

**Napomena:** Per-subject baseline je identičan globalnom (6,18°), jer su svi testni ispitanici subject-disjoint — model nema per-subject povijest za test ispitanike i koristi globalni fallback. Nije zasebno informativan i ne prijavljuje se kao zaseban rezultat u glavnoj usporedbi.

U final_summary.csv per-subject baseline je prisutan radi potpunosti (označen kao "cheating"), ali brojke su identične globalnom baselineu.
