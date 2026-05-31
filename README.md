# CS2 Demo Analyser & Coach

A private Counter-Strike 2 demo analysis and coaching tool built in Python.

The project parses CS2 demos, extracts player performance metrics, compares them against a local benchmark pool, estimates event impact with a machine learning model, and generates an actionable post-match coaching report.

The goal is not only to show raw statistics, but to explain what happened, why it mattered, and which rounds should be reviewed first.

---

## Features

### Demo parsing and cache

- Parses CS2 `.dem` files with `awpy` / `demoparser2`.
- Stores parsed demos in a local `.cache/` directory.
- Uses SHA-256 based cache keys / match IDs for reproducibility.
- Supports re-running analysis from cached demos without re-parsing the original demo.

### Player-level report

The analyser generates a detailed player report with:

- K/D/A
- KPR and DPR
- ADR
- KAST
- Headshot percentage
- Opening duel stats
- Trade kills
- Untraded deaths
- Damage before death
- Death timing: early / mid / late
- Economy summary
- Clutch summary
- CT/T side breakdown

### Local benchmark system

The project builds a local benchmark pool from previously analysed matches.

Benchmarks can compare a selected player against contextual pools such as:

- map + side
- side fallback
- global fallback

The report displays benchmark context per metric, so the user can see exactly which pool was used.

Example:

```text
BENCHMARKS
Status: available
Context: mixed
Metrics evaluated: 7

Benchmark pools:
- adr: map+side, samples=50
- hs_percent: map+side, samples=50
- full_buy_win_rate: side, samples=127
- clutch_win_rate: unavailable, reason=not_enough_player_samples
```

Benchmark-based feedback can identify strengths and weaknesses such as:

- low headshot percentage compared with the local pool
- strong ADR / damage impact
- weak clutch conversion
- strong KAST / round presence
- weak opening duel conversion

### ML round impact model

The project includes an experimental machine learning layer that estimates round win probability from event-based snapshots.

Current ML flow:

```text
cached demos
-> round snapshot dataset
-> LightGBM round win probability model
-> before/after event predictions
-> win probability delta
-> player-level ML impact summary
```

The ML layer estimates how much a kill or death changed the selected player's team's chance of winning the round.

Example:

```text
Round 18 | CT | died to phzy with m4a1 | -38.0 pp
Round 10 | T | killed phzy with awp | +43.2 pp
```

### ML impact report section

The report includes an experimental ML impact section with:

- kill impact score
- death impact score
- net ML impact
- average kill impact
- average death impact
- best kills
- worst deaths
- low-impact kills
- excluded contexts such as teamkills and world deaths

### Evidence-based coaching tips

Feedback tips include concrete examples from the match instead of only generic advice.

Example:

```text
[WARNING] Too many untraded deaths
7/11 deaths were untraded (63.6%). You often die outside trade range or take isolated duels.
Examples:
- Round 18 | CT | died to phzy with m4a1 | mid | -38.0 pp
- Round 5 | T | died to HooXi with fiveseven | late | -31.3 pp
```

Supported evidence examples currently include:

- untraded deaths
- zero-damage deaths
- low-damage deaths
- low headshot percentage examples
- high-impact kills from ML impact
- costly deaths from ML impact

### VOD Review Priority

The report ends with a compact VOD review queue that selects the most important rounds to watch first.

Example:

```text
VOD REVIEW PRIORITY
1. Round 18 | CT | high | mistake
   Reasons: zero-damage death, -37.99 pp ML impact, untraded death
   Summary: died to phzy with m4a1 in a high-cost ML swing.

2. Round 5 | T | high | mistake
   Reasons: -31.25 pp ML impact, untraded death
   Summary: died to HooXi with fiveseven in a high-cost ML swing.
```

This makes the report useful for actual demo review sessions, not only stat inspection.

---

## Tech stack

- Python
- Polars
- pandas
- LightGBM
- awpy
- demoparser2
- colorlog
- pyarrow

---

## Project structure

```text
src/
├── main.py                         # CLI entrypoint
├── Parser.py                       # demo parsing, cache handling, match IDs
├── Analyser.py                     # main analysis orchestration
├── coach_metrics.py                # raw player metrics
├── report_builder.py               # text report, tips, VOD priority formatting
├── benchmarks.py                   # local benchmark storage and percentile evaluation
├── backfill_benchmarks.py          # builds benchmark pool from cached demos
├── parse_demos_to_cache.py         # batch parse demos into cache
├── sectors/
│   ├── overall.py                  # player selection and overall stats
│   ├── economy.py                  # economy-related stats
│   ├── clutch.py                   # clutch stats
│   ├── round_timeline.py           # timeline and impact events
│   ├── feedback.py                 # coaching tips and evidence examples
│   ├── ml_impact.py                # event-level ML delta calculation
│   └── player_ml_impact.py         # player-level ML impact summary
└── ml/
    ├── build_dataset.py            # builds round snapshot dataset
    ├── dataset.py                  # dataset construction helpers
    ├── features.py                 # feature extraction for ML snapshots
    ├── train_lgbm.py               # LightGBM training script
    ├── predict.py                  # round win probability prediction
    ├── evaluate_impact.py          # builds ML event impact parquet
    ├── evaluate_player_impact.py   # player-level ML impact CLI
    ├── audit_impact.py             # CT/T symmetry and impact sanity checks
    └── inspect_dataset.py          # dataset inspection utility
```

---

## Installation

Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r src/requirements.txt
```

If you use the ML training and prediction scripts, make sure `lightgbm` is installed in your environment.

---

## Basic usage

### 1. Add a CS2 demo

Place a CS2 demo file in the project, for example:

```text
Demos/demo.dem
```

### 2. Run the analyser

From the `src` directory:

```powershell
cd src
py main.py
```

The CLI will show available players and ask which player should be analysed.

### 3. Re-run from cache

After a demo has been parsed once, the analyser can re-run from the cached match using:

```text
last_cache_key.txt
.cache/
```

This avoids re-parsing the original `.dem` file every time.

---

## Benchmark workflow

Benchmarks are local and are built from analysed / cached demos.

To backfill benchmark samples from all cached matches:

```powershell
cd src
py backfill_benchmarks.py
```

To force rebuild the benchmark registry:

```powershell
py backfill_benchmarks.py --force
```

The benchmark system stores data in:

```text
src/benchmarks/local_benchmarks.json
src/benchmarks/analyzed_matches.json
```

---

## ML workflow

### Build the dataset

```powershell
cd src
py ml/build_dataset.py
```

Output:

```text
data/ml/round_snapshots.parquet
```

### Inspect the dataset

```powershell
py ml/inspect_dataset.py
```

### Train the LightGBM model

```powershell
py ml/train_lgbm.py --seed 42 --n-trials 0
```

Typical outputs:

```text
models/round_win_lgbm.txt
data/ml/round_win_lgbm_metrics.json
data/ml/round_win_lgbm_feature_importance.json
```

### Build ML event impact

```powershell
py ml/evaluate_impact.py
```

Outputs:

```text
data/ml/ml_event_impact.parquet
data/ml/top_positive_events.json
data/ml/top_negative_events.json
```

### Audit ML impact consistency

```powershell
py ml/audit_impact.py
```

The audit checks whether CT and T perspectives are approximately consistent for the same event.

### Evaluate one player

```powershell
py ml/evaluate_player_impact.py --steamid <STEAM_ID> --top-n 10
```

---

## Example report sections

### Benchmark-based feedback

```text
[CRITICAL] Headshot Consistency Is Well Below Benchmark
hs_percent 17.65 is around the 4.0 percentile in your map+side pool.

Examples:
- Round 14 | CT | killed jabbi with famas | non-HS | +5.4 pp
- Round 2 | T | killed jabbi with glock | non-HS | +4.7 pp
```

### ML impact

```text
ML IMPACT (EXPERIMENTAL)
Kills: 17
Deaths: 11
Net impact score: +0.719 impact score
Kill impact score: +2.749 impact score
Death impact score: -2.030 impact score
Average kill impact: +16.17 pp per event
Average death impact: -18.46 pp per event
```

### VOD review priority

```text
VOD REVIEW PRIORITY
1. Round 18 | CT | high | mistake
   Reasons: zero-damage death, -37.99 pp ML impact, untraded death
   Summary: died to phzy with m4a1 in a high-cost ML swing.

2. Round 5 | T | high | mistake
   Reasons: -31.25 pp ML impact, untraded death
   Summary: died to HooXi with fiveseven in a high-cost ML swing.
```

---

## Current status

This project is currently an advanced MVP / experimental private coaching tool.

Working end-to-end:

- demo parsing and cache
- player-level report
- CT/T side breakdown
- economy and clutch analysis
- local benchmark comparison
- benchmark-based feedback
- ML event impact
- player-level ML impact
- evidence examples for tips
- VOD review priority section

Still experimental:

- ML impact interpretation
- benchmark quality depends on the size and quality of the local benchmark pool
- report output is currently CLI/text-first
- web/API layer is not implemented yet

---

## Roadmap

Planned improvements:

- structured JSON report export
- FastAPI backend
- web dashboard
- match history browser
- richer round timeline viewer
- better ML features, including equipment and HP state
- player-specific trend tracking across matches
- more robust VOD review indexing
- optional frontend visualizations for impact swings and benchmark percentiles

---

## Notes

The benchmark system is local by design. Percentiles describe how a player compares to the current local benchmark pool, not to the entire global CS2 player population.

The ML impact model estimates changes in round win probability. It should be treated as an assistive signal for review, not as an absolute truth.

---

## Repository description

Private CS2 demo analysis and coaching tool with local benchmarks, ML-based round impact, evidence-backed feedback, and VOD review priorities.

---

## Suggested GitHub topics

```text
counter-strike-2
cs2
cs2-demo
demo-analysis
esports-analytics
game-analytics
python
polars
machine-learning
lightgbm
awpy
demoparser2
sports-analytics
coaching-tool
vod-review
```

---

## License

No license has been specified yet.
