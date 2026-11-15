# Bulk Analysis benchmark

Machine: arm64 · macOS-15.5-arm64-arm-64bit · device: mps

| Rows | Rewrites | Time | Rows per second | Rows rewritten | Avg score gain | Candidates rejected for invented claims |
|---|---|---|---|---|---|---|
| 1,000 | off | 5.7 s | 177.0 | - | - | - |
| 100 | on | 100.4 s | 1.0 | 57 | +33.0 | 84 |

With rewrites on, each row costs about 1.00 s, so 1,000 rows take about 17 min and 5,000 rows about 84 min.

## With the Qwen2.5-1.5B rewriter (current default)

| Rows | Rewrites | Time | Notes |
|---|---|---|---|
| 100 | on, 6 candidates, batches of 8 | 242.6 s | first setting tried |
| 100 | on, 3 candidates, batches of 16 | 100.4 s | current setting; same average gain (+33.0) |
| 1,000 | off | 5.7 s | analysis only |

Qwen writes real rewrites (see rewrite_comparison.md) but costs about 1 s per weak row in bulk,
against about 0.2 s for flan-t5-base. Set LOCAL_REWRITE_MODEL=google/flan-t5-base for faster,
weaker bulk rewrites.
