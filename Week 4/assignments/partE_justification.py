"""
partE_justification.py — Generate partE_justification.pdf (one page)
Week 4, Part E · CDB601220 Data Visualization
Roll No: su92-bsaim-s23-012
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = os.path.dirname(__file__)

TEXT = """
PART E — WRITTEN JUSTIFICATION                                                   CDB601220 · Week 4
Roll No: su92-bsaim-s23-012 · Course: Data Visualization · Superior University Lahore

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. PER-PANEL QUESTION AND JUSTIFICATION

Panel 1 — Daily sessions trend (raw + rolling mean)
  Question: Is overall traffic growing, flat, or declining over the 17-month period?
  Chart type: Line (raw in grey, signal in colour). Week 2: one pop-out per chart; grey for context.
  Channel rule: Position on a common scale (x = time, y = sessions) — best channel for trend.
  Rolling mean line uses the colour series[0] (blue) against the grey raw line.

Panel 2 — Day-of-week × hour-of-day heatmap
  Question: When during the week is traffic highest, and which hour peaks?
  Chart type: Heatmap (imshow). Sequential colormap = Blues (no midpoint needed; week 3 rule).
  Channel rule: Sequential colour encodes magnitude in one direction; position encodes
  the two categorical axes. No dual-y axis (banned Week 1).

Panel 3 — Treemap of category → sub-category net revenue
  Question: Where does the revenue mass sit in the catalogue hierarchy?
  Chart type: Squarified treemap. Area encodes magnitude; containment encodes parent–child
  relationship (Munzner ch. 12 containment idiom). One hue per parent category.
  Values printed on every block (area is only the 4th-best channel; Week 2).
  Channel rule: position (block arrangement) + area + text labels.

Panel 4 — Sorted horizontal bars, same sub-categories
  Question: By how much does each sub-category differ from the others?
  Chart type: Horizontal bars sorted descending. Position on a common scale = best channel.
  This is the Week 2 argument: angle (treemap area) < length (bar) for precise comparison.
  Values printed at bar ends to allow exact reading without a grid line.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2. EVERY TIME DECISION

Parse: sessions_hourly.csv contains two timestamp formats — ISO (YYYY-MM-DD HH:MM:SS, 9,019 rows)
  and DD/MM/YYYY HH:MM (3,651 rows). Using pd.to_datetime(format="mixed") mis-assigns 1,382
  DD/MM rows (treats day as month). Fix: parse each format separately; fillna to combine.
  Verified by checking index.min()=2024-04-01 and index.max()=2025-09-11.

Resolution: hourly (1 h). The raw data is hourly; no finer grain is available or needed.
  Resampled to daily (sum) for the trend panel. Aggregation = SUM because sessions is a FLOW
  (count of events during the hour, not a point-in-time level). Sum is correct for flows; mean or
  last() would give the wrong daily total.

Window length: 7-day centred rolling mean. Fixed before plotting (Part B.1 computed first).
  The weekly cycle (Mon–Sun) is the dominant periodicity: the heatmap confirms weekend peaks.
  A 7-day window cancels exactly one weekly cycle. A 28-day window would also remove monthly
  variation we want to see in Panel 1. A 91-day window loses 45 days at each edge.
  The window was fixed at 7 days BEFORE examining which direction the trend ran.
  A 28-day alternative is in partB_windows.png and shows the same upward direction.

Centred vs trailing: Panel 1 uses a CENTRED mean (historical figure, not a live dashboard).
  A trailing mean lags by half-window (≈15 days, confirmed by lag_at_turn() in Part B.4).
  A live dashboard must use trailing; a historical review may use centred.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3. EVERY ROW REMOVED

Rule 1 — Sentinel values: sessions == -1 (logging placeholder for failed collection).
  Count: 290 rows removed from 12,670 → 12,380 remain. Sessions moved from −1 to [removed].

Rule 2 — Duplicate timestamps: same timestamp after correct parsing, keep first.
  Count: 283 duplicates removed from 12,380 → 12,097 remain.
  Sessions moved from double-counted to single-counted; total session count reduced.

Rule 3 — Negative net_revenue_pkr_m in revenue_catalogue.csv (returns > gross revenue).
  Count: 7 SKU rows, total −1.07 PKR m. Excluded from treemap and bars (negatives have
  no area — slide 49). Reported in source note; not silently dropped.

Rule 4 — Duplicate SKU in revenue_catalogue.csv: 1 row removed (keep first occurrence).
  Count: 1 row, catalogue goes from 345 → 344 rows.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

4. EVERY LEVEL ROLLED UP

Hierarchy depth: category (4) → subcategory (8) → SKU (344 after dedup). Plotted at
  sub-category depth. SKU level is too granular for leadership review (4 ± 1 chunks, Week 2).

Other bucket: not needed — all 8 sub-categories are ≥ 1% of total. No tail was rolled up.
  rollup_other() confirmed: "no categories below 1%".

Category colours: one hue per parent category (4 categories = 3 course-palette hues + 4th slot).
  The 4th slot (#CC79A7) drops CVD margin to dE 7.6, so values are printed on every block
  (direct labels as the required second encoding channel — Week 3 rule).

ETHICS CLAUSE: The window (7 days) was chosen because the weekly cycle is the identifiable
  periodicity, not because it produced the best-looking result. Part B.5 showed that 48 of 60
  start-window combinations on this series produce a rising trend; the 7-day window was fixed
  before any of those were examined. The series start date (2024-04-01) was not chosen; it is
  the first recorded day. A 28-day alternative in partB_windows.png shows the same direction.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Source: sessions_hourly.csv & revenue_catalogue.csv — synthetic data generated for CDB601220.
""".strip()

fig = plt.figure(figsize=(11, 8.5))
fig.patch.set_facecolor("white")
ax = fig.add_axes([0.04, 0.02, 0.92, 0.96])
ax.set_axis_off()

ax.text(0.0, 1.0, TEXT, transform=ax.transAxes,
        fontsize=6.3, fontfamily="monospace",
        verticalalignment="top", horizontalalignment="left",
        color="#16202E", linespacing=1.38,
        wrap=False)

out_path = os.path.join(OUT, "partE_justification.pdf")
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved → {out_path}")
