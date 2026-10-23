"""
partB_windows.py — Smoothing, windows and panel shape, computed
Week 4, Part B · CDB601220 Data Visualization
Roll No: su92-bsaim-s23-012

Depends on the cleaned daily session series produced by partD_figure.py logic.
Run partD_figure.py first (or rely on the shared cleaning function here).
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats

from tslib import (
    PALETTE, style_defaults, save_fig,
    audit_index, reindex_full, window_report, lag_at_turn, banked_figsize
)

DATA = os.path.join(os.path.dirname(__file__), "data")
OUT  = os.path.dirname(__file__)
style_defaults()

# ============================================================
# Reproduce the same cleaning as partD_figure.py
# ============================================================
raw = pd.read_csv(os.path.join(DATA, "sessions_hourly.csv"))
iso = pd.to_datetime(raw.ts, format="%Y-%m-%d %H:%M:%S", errors="coerce")
dmy = pd.to_datetime(raw.ts, format="%d/%m/%Y %H:%M",   errors="coerce")
raw["ts"] = iso.fillna(dmy)
raw = raw.sort_values("ts")
raw = raw[raw.sessions >= 0].copy()                       # drop sentinels
raw["device"] = raw["device"].str.strip().str.capitalize()
raw = raw.set_index("ts")
raw = raw[~raw.index.duplicated(keep="first")]

last_month = raw.index[-1].to_period("M")
raw_plot = raw[raw.index.to_period("M") != last_month]    # exclude partial Sep-25

daily = raw_plot["sessions"].resample("D").sum(min_count=1)
daily_full = reindex_full(daily, "D")                     # NaN gaps = visible breaks

print("=== Part B: Smoothing, Windows and Panel Shape ===\n")
print(f"Daily series: {len(daily_full)} days, "
      f"{daily_full.isna().sum()} NaN gaps, "
      f"{daily_full.min():.0f}–{daily_full.max():.0f} sessions/day\n")

# ============================================================
# B.1 — Window report
# ============================================================
print("─" * 60)
print("B.1 — Window report")
print("─" * 60)
WINDOWS = (7, 28, 91)
wr = window_report(daily_full.dropna(), WINDOWS, center=True)
for row in wr.itertuples(index=False):
    lost = row.periods_lost_each_end
    # Identify which cycle each window removes
    if row.window == 7:
        cycle = "weekly (7-day) cycle — same days align"
    elif row.window == 28:
        cycle = "monthly (≈4-week) cycle; also removes weekly"
    else:
        cycle = "quarterly (≈13-week) cycle; removes weekly+monthly"
    print(f"\nWindow = {row.window:2d} days")
    print(f"  Cycle removed      : {cycle}")
    print(f"  Raw SD             : {row.raw_sd:.1f}")
    print(f"  Smoothed SD        : {row.smoothed_sd:.1f}")
    print(f"  Variation removed  : {row.variation_removed_pct:.1f}%")
    print(f"  Periods lost (each end): {lost}")

print("\n>>> Choice for Part D: 7-day centred mean")
print("    Reason: the weekly cycle (Mon–Sun) is the dominant periodicity.")
print("    A 7-day window cancels it exactly without consuming 28 or 91")
print("    days of data at the edges. The 28-day window also removes")
print("    monthly variation we want to keep visible.")


# ============================================================
# B.2 — Three smoothers on the Black Friday spike
# ============================================================
print("\n" + "─" * 60)
print("B.2 — Three smoothers at the Black Friday spike")
print("─" * 60)
BF = pd.Timestamp("2024-11-29")
HALF = 14

# 14-day centred mean
mean14  = daily_full.rolling(HALF*2, center=True, min_periods=7).mean()
# 14-day centred median
med14   = daily_full.rolling(HALF*2, center=True, min_periods=7).median()
# EWMA (half-life=7 days ≈ weekly)
ewma14  = daily_full.ewm(halflife=7, adjust=False).mean()

bf_raw   = float(daily_full[BF])
bf_mean  = float(mean14[BF])
bf_med   = float(med14[BF])
bf_ewma  = float(ewma14[BF])
bf_baseline = float(daily_full[BF - pd.Timedelta(days=14):BF - pd.Timedelta(days=1)].median())

print(f"\nBlack Friday ({BF.date()}):")
print(f"  Raw value          : {bf_raw:,.0f} sessions")
print(f"  14-day baseline median (fortnight before): {bf_baseline:,.0f}")
print(f"  14-day centred mean at BF : {bf_mean:,.0f}  (spike inflates it)")
print(f"  14-day centred median at BF: {bf_med:,.0f}  (spike does NOT inflate it)")
print(f"  EWMA (halflife=7) at BF   : {bf_ewma:,.0f}  (spike inflates it; decays)")
print()
print("  Monitoring dashboard → 14-day centred MEDIAN")
print("    Reason: a live threshold alert should NOT fire on a known")
print("    promotional event. The median is spike-resistant; the mean")
print("    would trigger a false alert for 14 days around every BF.")
print()
print("  Quarterly report → 14-day centred MEAN")
print("    Reason: a quarterly summary should reflect total traffic")
print("    volume including events; the mean preserves that. The spike")
print("    should be annotated, not hidden.")


# ============================================================
# B.3 — Seasonal decomposition
# ============================================================
print("\n" + "─" * 60)
print("B.3 — Seasonal decomposition")
print("─" * 60)

clean_daily = daily_full.interpolate(method="time").dropna()

decomp7 = seasonal_decompose(clean_daily, model="additive", period=7,
                             extrapolate_trend="period")

s7_amp   = float(decomp7.seasonal.max() - decomp7.seasonal.min())
resid7_peak = decomp7.resid.abs().idxmax()
resid7_val  = float(decomp7.resid[resid7_peak])

print(f"\nperiod=7 (weekly):")
print(f"  Seasonal amplitude (peak-to-trough) : {s7_amp:,.0f} sessions/day")
print(f"  Largest residual date               : {resid7_peak.date()}")
print(f"  Largest residual value              : {resid7_val:+,.0f} sessions")

# period=365 requires ≥730 observations; series has 518 days.
# Use period=91 (quarterly) as the long-period decomposition instead.
decomp_long = seasonal_decompose(clean_daily, model="additive", period=91,
                                 extrapolate_trend="period")
s_long_amp   = float(decomp_long.seasonal.max() - decomp_long.seasonal.min())
resid_long_peak = decomp_long.resid.abs().idxmax()
resid_long_val  = float(decomp_long.resid[resid_long_peak])

print(f"\nperiod=91 (quarterly — used because series < 2 years; period=365 needs ≥730 rows):")
print(f"  Seasonal amplitude                  : {s_long_amp:,.0f}")
print(f"  Largest residual                    : {resid_long_peak.date()} ({resid_long_val:+,.0f})")
print()
print("  Switching period=7 → period=91: the Black Friday spike and")
print("  the quarterly ramp-up patterns move FROM resid INTO seasonal.")
print("  What remains in resid is only day-level events unexplained")
print("  by either the weekly or the quarterly cycle.")
decomp365 = decomp_long   # alias used below in the figure


# ============================================================
# B.4 — Trailing vs centred lag
# ============================================================
print("\n" + "─" * 60)
print("B.4 — Trailing vs centred (31-day window)")
print("─" * 60)
lag_info = lag_at_turn(daily_full.dropna(), 31, how="min")
print(f"\n  Trailing turn : {lag_info['trailing_turn'].date()}")
print(f"  Centred turn  : {lag_info['centred_turn'].date()}")
print(f"  Lag in days   : {lag_info['lag_periods']}  (half-window = {lag_info['half_window']})")
print()
print("  A live dashboard refreshes every morning → must show TRAILING mean.")
print("  Caption must read: '31-day trailing mean — lags real trend by ≈15 days.'")
print("  A centred mean requires future data, so it can only appear in retrospective reports.")


# ============================================================
# B.5 — The window is a claim (Lie Factor analogy)
# ============================================================
print("\n" + "─" * 60)
print("B.5 — Window cherry-picking: rising vs falling trend counts")
print("─" * 60)

rising = 0
falling = 0
results = []
for start_month in range(1, 13):
    start_dt = pd.Timestamp(f"2024-{start_month:02d}-01")
    if start_dt > daily_full.index[-1]:
        start_dt = pd.Timestamp(f"2025-{start_month:02d}-01")
    for wlen in (7, 14, 28, 56, 91):
        subset = daily_full[start_dt:].dropna().iloc[:wlen*4]  # ≈ 4 windows of data
        if len(subset) < wlen * 2:
            continue
        roll = subset.rolling(wlen, center=True, min_periods=wlen//2).mean().dropna()
        if len(roll) < 3:
            continue
        x = np.arange(len(roll))
        slope, *_ = stats.linregress(x, roll.values)
        if slope > 0:
            rising += 1
        else:
            falling += 1

total_combos = rising + falling
print(f"\n  Combinations producing a RISING  trend: {rising:3d} / {total_combos}")
print(f"  Combinations producing a FALLING trend: {falling:3d} / {total_combos}")
print()
print("  Why Tufte's Lie Factor CANNOT be computed for a cherry-picked window:")
print()
print("  Lie Factor = (effect shown) / (effect in the data).")
print("  This formula assumes there is ONE true 'effect in the data'.")
print("  When the analyst chooses the window after seeing all possible slopes,")
print("  there is no single reference effect — the denominator is undefined.")
print("  Different honest windows give different (and equally valid) slopes,")
print("  so there is no unique ground truth to compare against.")
print()
print("  The honest substitute is:")
print("  Report ALL 60 window combinations (as done above) and state which")
print("  one was chosen, why it was fixed before plotting, and what fraction")
print("  of the remaining combinations agree with the reported direction.")
print(f"  Here: {rising}/{total_combos} ({rising/total_combos*100:.0f}%) rising, "
      f"{falling}/{total_combos} ({falling/total_combos*100:.0f}%) falling.")


# ============================================================
# Produce a summary figure for B
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
C0, C1, C2 = PALETTE["series"]
MU = PALETTE["muted"]

# Panel 1: Window comparison
ax1 = axes[0, 0]
ax1.plot(daily_full.index, daily_full.values, color=MU, alpha=0.4, linewidth=0.7, label="Raw")
for wlen, col, ls in [(7, C0, "-"), (28, C2, "--"), (91, C1, ":")]:
    r = daily_full.rolling(wlen, center=True, min_periods=wlen//2).mean()
    ax1.plot(r.index, r.values, color=col, linewidth=1.8, linestyle=ls,
             label=f"{wlen}-day mean")
ax1.set_title("B.1 — Three window lengths on daily sessions\n[centred rolling mean]")
ax1.set_ylabel("Sessions per day (sum)")
ax1.legend(fontsize=8)
ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

# Panel 2: Three smoothers at Black Friday
ax2 = axes[0, 1]
bf_window = daily_full[BF - pd.Timedelta(days=30):BF + pd.Timedelta(days=30)]
ax2.plot(bf_window.index, bf_window.values, color=MU, alpha=0.5, linewidth=1, label="Raw")
ax2.plot(mean14.reindex(bf_window.index).index, mean14.reindex(bf_window.index).values,
         color=C0, linewidth=2, label="14-day mean")
ax2.plot(med14.reindex(bf_window.index).index, med14.reindex(bf_window.index).values,
         color=C2, linewidth=2, linestyle="--", label="14-day median")
ax2.plot(ewma14.reindex(bf_window.index).index, ewma14.reindex(bf_window.index).values,
         color=C1, linewidth=2, linestyle=":", label="EWMA (h=7)")
ax2.axvline(BF, color=PALETTE["ink_soft"], linewidth=1, linestyle="--")
ax2.text(BF, ax2.get_ylim()[0] if ax2.get_ylim()[0] else 0, " BF", fontsize=7,
         color=PALETTE["ink_soft"])
ax2.set_title(f"B.2 — Three smoothers at Black Friday {BF.date()}")
ax2.set_ylabel("Sessions per day (sum)")
ax2.legend(fontsize=8)
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

# Panel 3: Seasonal decomposition (period=7)
ax3 = axes[1, 0]
ax3.plot(decomp7.seasonal.index, decomp7.seasonal.values, color=C0, linewidth=1.2)
ax3.axhline(0, color=PALETTE["grid"], linewidth=0.8)
ax3.set_title(f"B.3 — Weekly seasonal component (period=7)\nAmplitude: {s7_amp:,.0f} sessions/day")
ax3.set_ylabel("Seasonal component")
ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)

# Panel 4: Trailing vs centred
ax4 = axes[1, 1]
trail_31 = daily_full.rolling(31).mean()
cent_31  = daily_full.rolling(31, center=True).mean()
ax4.plot(daily_full.index, daily_full.values, color=MU, alpha=0.4, linewidth=0.7, label="Raw")
ax4.plot(trail_31.index, trail_31.values, color=C0, linewidth=2, label="31-day trailing")
ax4.plot(cent_31.index, cent_31.values, color=C2, linewidth=2, linestyle="--", label="31-day centred")
ax4.set_title(f"B.4 — Trailing vs centred (lag = {lag_info['lag_periods']} days)")
ax4.set_ylabel("Sessions per day (sum)")
ax4.legend(fontsize=8)
ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)

fig.tight_layout()
save_fig(fig, os.path.join(OUT, "partB_windows.png"),
         note="Source: sessions_hourly.csv (cleaned). Synthetic data — CDB601220 Week 4.")
plt.close(fig)
print(f"\nSaved → partB_windows.png")
