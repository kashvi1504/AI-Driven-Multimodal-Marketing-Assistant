# Bulk Analysis benchmark

Machine: arm64 · macOS-15.5-arm64-arm-64bit · device: mps

| Rows | Rewrites | Time | Rows per second | Rows rewritten | Avg score gain | Candidates rejected for invented claims |
|---|---|---|---|---|---|---|
| 50 | off | 0.3 s | 182.0 | - | - | - |
| 100 | off | 0.4 s | 232.1 | - | - | - |
| 1,000 | off | 4.3 s | 230.3 | - | - | - |
| 5,000 | off | 23.5 s | 212.6 | - | - | - |
| 50 | on | 4.5 s | 11.2 | 11 | +21.6 | 4 |
| 100 | on | 7.9 s | 12.7 | 57 | +32.9 | 13 |
| 1,000 | on | 188.3 s | 5.3 | 703 | +27.8 | 154 |

With rewrites on, the 1,000-row run cost about 0.19 s per row, so 5,000 rows take about 16 min.

## Before the speed-ups (same machine, same rows)

| Rows | Rewrites | Time |
|---|---|---|
| 1,000 | off | 35.5 s |
| 5,000 | off | 167.9 s |
| 100 | on | 59.7 s |
| 1,000 | on | ~18 min (1.1 s per row, measured on 50 and 100 rows) |

What changed: the sentiment model and the rewriter run on the Apple-silicon GPU (MPS) when there is
one; sentiment is batched 32 at a time; rows that already score 70+ keep their copy instead of being
rewritten; identical rows are rewritten once; and candidates for 8 rows are generated in one model call
(6 candidates per row instead of 8). Every candidate still goes through the grounding check.

Limit: there is no hard model limit. Uploads are capped at 20,000 rows (`BULK_MAX_ROWS` in config.py)
only to protect memory; analysis alone finishes 20,000 rows in about 1.5 minutes.
