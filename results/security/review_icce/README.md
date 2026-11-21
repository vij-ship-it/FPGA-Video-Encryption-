# results/security/review_icce: runs for the ICCE 2027 camera-ready (reviews of 20 Nov 2026)

| File | Produced by | Where it ran |
|---|---|---|
| `e5_size_match/<source>.json`, `.md`, `summary.*` | `experiments/e5_size_match.py` (4 starts, gaps 1 2 5 10 25, stride 2, 320x240, quality 75) via `software/system/kaggle/kaggle_runs.py` (`JOB = "review"`), branch `claude/project-thread-65c6k0` at commit cdd0eac (`run_log.json`: 9.3 min, every step PASS) | Kaggle, CPU (results received 21 Nov 2026) |
| `benchmark_suite_xiph/<sequence>.json`, `.md` | `experiments/benchmark_suite.py` on the six Xiph CIF sequences, same run | Kaggle |
| `xiph_manifest.md` | `software/scripts/get_xiph.py`: where each sequence came from, with its sha256 | Kaggle |
| `review_numbers/review_numbers.tex`, `.json` | `python experiments/review_numbers.py review_results.zip`: the camera-ready's macros and the values behind them | this repository |

Sources of the size-matching run: the synthetic scene (not pooled), the six Xiph sequences (akiyo, foreman,
news, hall_monitor, mobile, mother_daughter) and the Sintel and Tears of Steel trailers. The attacker kept the
last E2 frame and places each E5 segment at the E2 MCU with the most similar length and non-zero counts
(rank by length, or minimum-cost assignment); its view is scored against the new frame and compared with the
old E2 frame alone (`e2_old`), random placement and the true order (`oracle`, what E4 would show).

Nothing here is typed by hand: the files are the unzipped `review_results.zip` plus the output of
`review_numbers.py`.
