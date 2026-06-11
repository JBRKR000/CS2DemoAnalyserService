# CS2 Demo Coach

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Status](https://img.shields.io/badge/status-Coach%20MVP-green)
![Release](https://img.shields.io/badge/release-v1.0.0--coach--mvp-purple)
![ML](https://img.shields.io/badge/ML-LightGBM-orange)
![Game](https://img.shields.io/badge/game-CS2-yellow)

CS2 Demo Coach is a post-match analysis tool for Counter-Strike 2 demos. It takes a parsed demo, focuses on one selected player, and turns the match into a practical coaching report with stats, benchmarks, ML-based impact estimates, VOD review priorities, death-risk context, and suggested decision alternatives.

Current release: `v1.0.0-coach-mvp`

The idea is simple: instead of only showing numbers, the tool tries to answer the questions a player or coach would actually ask after a match:

```text
What went wrong?
Where did it happen?
Why did it matter?
Was the player already in danger?
What could have been done instead?
What should be practiced next?
```

A typical report flow looks like this:

```text
demo/cache
→ player stats
→ benchmark comparison
→ feedback tips
→ ML impact scoring
→ VOD review queue
→ death-risk context
→ decision suggestions
→ coach summary
→ structured JSON export
```

---

## Preview

### Coach Summary

```text
COACH SUMMARY
Main weakness:
- Headshot consistency is below the benchmark pool.

Best strength:
- Round survival/trade value is strong compared with the benchmark pool.

Top VOD focus:
- Round 18: zero-damage death, -37.99 pp ML impact, high death risk

Decision pattern:
- You often take fights where backing off would preserve round equity.

Practice focus:
- Run 10 minutes of head-height pathing and first-bullet discipline.
```

### VOD Review Priority

```text
VOD REVIEW PRIORITY
1. Round 18 | CT | high | mistake
   Reasons: zero-damage death, -37.99 pp ML impact, untraded death
   Risk before death: high, 18.6%, top_10_percent, enemy 475u, teammate 334u, hp 100, weapon m4a1
   Risk explanation: enemy close
   Summary: died to phzy with m4a1 in a high-cost ML swing.
```

### Decision Simulation

```text
DECISION SIMULATION (MVP)
1. Round 18 | CT
   Actual: died to phzy with m4a1 in a high-cost ML swing.
   Risk before death: high, 18.6% (enemy close) | score -1.08

   Better alternatives:
   - fall_back | score +0.75 | safer after zero-damage high-cost death
   - wait_for_trade | score +0.65 | keeps trade possibility alive
   - hold_angle | score +0.25 | reduces isolation risk
```

---

## What this project does

CS2 Demo Coach analyses one selected player from a CS2 demo and generates a report that is useful for review, not just stat tracking.

It currently covers:

- overall player stats
- CT/T side breakdown
- economy and clutch summaries
- local benchmark comparison
- percentile-based feedback
- evidence-backed examples
- experimental ML impact scoring
- VOD review priority ranking
- 5-second death-risk estimation
- compact explanations for risky situations
- MVP decision suggestions
- coach summary generation
- structured JSON export for future API or frontend use

The coaching angle is the important part. The report is meant to point to specific rounds, deaths, and decisions that are worth reviewing.

---

## Current status

The project is currently at:

```text
v1.0.0-coach-mvp
```

Working end-to-end:

- demo parsing and cache loading
- player selection
- player-level text report
- CT/T side breakdown
- impact, economy, and clutch analysis
- local benchmark pool
- benchmark-based feedback
- ML event impact
- VOD review priority
- 5-second death-risk estimation
- risk explanation layer
- risk-aware decision simulation
- Coach Summary v1
- structured report dictionary
- structured JSON report export
- structured report validation

Still experimental:

- ML impact interpretation
- decision simulation
- benchmark quality, since it depends on the size and quality of the local benchmark pool
- no web/API layer yet
- no long-term personalized player calibration yet

---

## A note on the ML layer

The ML layer should be treated as a **pro-baseline risk and impact model**.

In other words, it is useful for review and prioritization, but it is not claiming to be a perfect personalized decision engine. Right now it is best used to support:

- VOD prioritization
- risk-aware coaching
- round impact explanation
- post-match review
- structured report generation

The decision simulation is also still MVP-level. It uses signals such as risk, ML impact, untraded deaths, zero-damage deaths, and VOD context to suggest safer alternatives, including:

- `fall_back`
- `wait_for_trade`
- `hold_angle`
- `reposition`
- `play_time`

Treat those suggestions as coaching prompts, not absolute truth.

---

## Report sections

A generated text report includes:

```text
CS2 COACH REPORT
├── OVERALL
├── IMPACT
├── SIDE BREAKDOWN
├── ECONOMY
├── CLUTCH
├── BENCHMARKS
├── ML IMPACT
├── TOP TIPS
├── VOD REVIEW PRIORITY
├── DECISION SIMULATION
└── COACH SUMMARY
```

---

## Structured JSON report

Alongside the text report, the project exports a structured JSON report. This is mainly intended for a future API, dashboard, or frontend.

Top-level schema:

```json
{
  "meta": {},
  "player": {},
  "overall": {},
  "impact": {},
  "side_breakdown": {},
  "economy": {},
  "clutch": {},
  "benchmarks": {},
  "ml_impact": {},
  "tips": [],
  "vod_review_priority": [],
  "decision_simulation": [],
  "coach_summary": {}
}
```

Example:

```json
{
  "meta": {
    "schema_version": "1.0",
    "report_type": "cs2_coach_report",
    "map_name": "de_ancient",
    "match_id": "b3501fa08fdf8b8000737715162434eb95464320e710a381e2e2c25241705822",
    "generated_at": "2026-06-11T21:13:27.969110+00:00"
  },
  "player": {
    "steamid": 76561198254686734,
    "name": "hypex",
    "start_side": "t",
    "rounds_played": 21
  },
  "coach_summary": {
    "main_weakness": "Headshot consistency is below the benchmark pool.",
    "best_strength": "Round survival/trade value is strong compared with the benchmark pool.",
    "top_vod_focus": "Round 18: zero-damage death, -37.99 pp ML impact, high death risk",
    "decision_pattern": "You often take fights where backing off would preserve round equity.",
    "practice_focus": "Run 10 minutes of head-height pathing and first-bullet discipline."
  }
}
```

Reports are written to:

```text
data/reports/<match_id>_<steamid>_structured_report.json
```

---

## Example workflow

```text
1. Load a parsed demo from cache, or parse a new demo.
2. Select the player to analyse.
3. Build player stats and side breakdowns.
4. Compare the player against local benchmarks.
5. Generate feedback tips with supporting evidence.
6. Estimate ML impact for kills and deaths.
7. Rank the most useful VOD review moments.
8. Add 5-second death-risk context.
9. Generate risk-aware decision suggestions.
10. Produce a short coach summary.
11. Export the structured JSON report.
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/JBRKR000/cs2-demo-coach.git
cd cs2-demo-coach
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the analyser from the `src` directory:

```bash
cd src
py main.py
```

Or, depending on your Python setup:

```bash
python main.py
```

The CLI will list the available players from the loaded match:

```text
Available players for detailed analysis:
idx | steamid           | name
0   | 76561198310561479 | PR
1   | 76561198005107817 | Staehr
2   | 76561198408199043 | phzy
3   | 76561198254686734 | hypex
...
```

Select a player by index, SteamID, or name:

```text
Select player [idx/steamid/name], Enter=0: 3
```

You should then see logs similar to:

```text
Structured report sections: meta, player, overall, impact, side_breakdown, economy, clutch, benchmarks, ml_impact, tips, vod_review_priority, decision_simulation, coach_summary
Structured report valid | sections=13
Structured report exported | path=data/reports/<match_id>_<steamid>_structured_report.json
```

---

## Features

### Demo analysis

- cached demo loading
- selected-player analysis
- round count
- map detection
- player identity extraction
- CT/T side handling

### Player statistics

- kills
- deaths
- assists
- KPR
- DPR
- ADR
- KAST
- headshot percentage
- opening duel win percentage
- trade kills
- untraded deaths
- zero-damage deaths
- damage before death
- death timing

### Economy analysis

- full-buy win rate
- force-buy win rate
- eco kills
- broken economy rounds
- save rounds

### Clutch analysis

- total clutches
- clutch wins
- clutch win rate
- 1v1 / 1v2 / 1v3+ breakdown

### Benchmarks

The benchmark system compares the selected player against a local benchmark pool and assigns metric percentiles and labels.

Example:

```text
Metric     | Value     | Percentile | Rating
ADR        | 105.00    | 68.0       | good
KAST       | 85.72     | 90.0       | excellent
KPR        | 0.81      | 80.0       | excellent
HS%        | 17.65     | 4.0        | critical
OPEN%      | 50.00     | 63.2       | good
```

Where available, benchmark checks can also take map and side context into account.

### Feedback tips

Tips are generated from benchmark ratings and player-specific evidence.

Example:

```text
[CRITICAL] Too many zero-impact deaths
6 deaths came with zero damage, and 8 deaths were under 40 damage before dying.
Take first contact with utility or teammate timing so you can create impact before going down.
```

### ML impact

The project estimates event impact using an experimental round-win probability model.

Example:

```text
Best kills:
- Round 10 | T | hypex killed phzy with awp | +43.23 pp
- Round 8 | T | hypex killed ryu with awp | +42.40 pp

Worst deaths:
- Round 18 | CT | phzy killed hypex with m4a1 | -37.99 pp
- Round 5 | T | HooXi killed hypex with fiveseven | -31.25 pp
```

### VOD Review Priority

The VOD priority system ranks moments that are likely to be worth reviewing first.

It looks at signals such as:

- zero-damage deaths
- untraded deaths
- negative ML impact
- high death risk
- failed clutch context
- positive round-swinging kills
- mixed event patterns

Example:

```text
1. Round 18 | CT | high | mistake
   Reasons: zero-damage death, -37.99 pp ML impact, untraded death
   Risk before death: high, 18.6%, top_10_percent
   Summary: died to phzy with m4a1 in a high-cost ML swing.
```

### 5-second death risk

The death-risk model estimates whether a player is likely to die within the next 5 seconds. It uses round-state and spatial context, including:

- alive teammates
- alive enemies
- seconds remaining
- bomb state
- player HP
- side
- map
- weapon
- armor
- equipment value
- nearest teammate distance
- nearest enemy distance
- round phase

Risk labels:

```text
low
medium
high
critical
```

Risk buckets:

```text
normal
top_20_percent
top_10_percent
top_5_percent
top_1_percent
```

### Risk explanation

Risk predictions are summarized in plain language so the report does not only give a score.

Example:

```text
Risk explanation: low HP, enemy close, teammate nearby but duel still high-risk, model marked this as top-tier risk
```

### Risk-aware Decision Simulation

Decision Simulation is an MVP layer that suggests safer alternatives for high-priority VOD moments.

Example:

```text
Actual: died to HooXi with fiveseven in a high-cost ML swing.
Risk before death: critical, 21.1%
score -0.61

Better alternatives:
- wait_for_trade | score +0.85 | keeps trade possibility alive
- fall_back | score +0.75 | avoids repeating a high-cost ML death
- hold_angle | score +0.25 | reduces isolation risk
```

---

## Project structure

```text
.
├── src/
│   ├── main.py
│   ├── Parser.py
│   ├── Analyser.py
│   ├── report_builder.py
│   ├── benchmarks.py
│   ├── sectors/
│   │   ├── overall.py
│   │   ├── economy.py
│   │   ├── clutch.py
│   │   ├── round_timeline.py
│   │   ├── feedback.py
│   │   └── decision_simulator.py
│   └── ml/
│       ├── build_dataset.py
│       ├── train_lgbm.py
│       ├── evaluate_impact.py
│       ├── evaluate_player_impact.py
│       ├── build_situations.py
│       ├── build_all_situations.py
│       ├── build_decision_snapshots.py
│       ├── train_death_risk_lgbm.py
│       ├── train_death_risk_timeseries_lgbm.py
│       └── predict_death_risk_timeseries.py
├── data/
│   ├── reports/
│   ├── ml/
│   └── situations/
├── docs/
├── requirements.txt
└── README.md
```

---

## ML pipeline overview

The ML workflow is currently built around a pro-baseline approach:

```text
demo data
→ event snapshots
→ round win model
→ event impact scoring
→ player impact report
→ situations dataset
→ decision snapshots
→ time-sampled death-risk model
→ risk labels/buckets
→ report integration
```

Main ML components:

- round win probability model
- event impact evaluation
- player-level impact aggregation
- situation extraction
- decision snapshot generation
- 5-second death-risk model
- death-risk prediction export

---

## Development notes

Before committing changes, run:

```bash
python -m compileall -q src
```

Basic smoke test:

```bash
cd src
py main.py
```

Expected success signals:

```text
Structured report valid | sections=13
Structured report exported | path=data/reports/<match_id>_<steamid>_structured_report.json
```

---

## Release history

### v1.0.0-coach-mvp

First complete Coach MVP release.

Highlights:

- cached CS2 demo analysis
- player performance report
- local benchmark comparison
- evidence-backed feedback tips
- experimental ML impact scoring
- VOD review priority ranking
- 5-second death-risk estimation
- risk explanations
- risk-aware decision simulation
- Coach Summary v1
- structured JSON report export for API/frontend usage

---

## Recommended interpretation

This project is best understood as:

```text
a CS2 post-match coaching engine
```

rather than:

```text
a stats parser
```

The current MVP is strongest at:

- finding review-worthy rounds
- explaining costly deaths
- highlighting repeated weaknesses
- surfacing strong plays
- ranking VOD priorities
- producing structured report data for a future UI

---

## License

Add a license before publishing or sharing the repository widely.

Possible options:

- MIT License if you want a simple permissive open-source license

---

## Status

```text
Coach MVP complete.
```
