# GTA VI Twitch CCU Forecaster

[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![Rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ceaser-7/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![Reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ceaser-7/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

Entry for [Forecast GTA VI's launch-day Twitch viewers hour by hour](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) (make.fun, $250 pot).

`main.py` (Python 3.13, standard library only) reads one case and forecasts Twitch concurrent viewers in the Grand Theft Auto VI category for hours 2–24 of the release day (2026-11-19 UTC), as five quantiles (5%, 25%, 50%, 75%, 95%) per hour, from the first hour's print and everything recorded before it.

## Approach

Size-conditioned comparable-curve forecaster: a per-hour local-linear weighted-least-squares fit of the log-ratio ln(viewers_h / viewers_1) on log hour-1 size over the comparable launches in the case file, with a second kernel on pre-release hype (pre14max), unlock-wave alignment by US-midnight hour, contamination tier weights, and a spread from leave-one-out residuals. Deterministic, no network, runs in ~0.13 s per case.

## Practice score

Pinball loss on the log scale over the four practice launches: **0.142466** (house persistence baseline: 0.387097).
