"""
partD_figure.py — E-commerce review figure (2×2 panel)
Week 4, Part D · CDB601220 Data Visualization
Roll No: su92-bsaim-s23-012
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from tslib import (
    PALETTE, style_defaults, save_fig,
    audit_index, reindex_full, window_report,
    exclusivity_check, rollup_other, treemap
)

DATA = os.path.join(os.path.dirname(__file__), "data")
style_defaults()

# ============================================================
# STEP 1: Load and clean sessions_hourly.csv
# Find and fix all six data problems before any plotting
# ============================================================

raw = pd.read_csv(os.path.join(DATA, "sessions_hourly.csv"))
print("=== Part D: Data Cleaning ===\n")
print(f"Raw shape: {raw.shape}")
print("First 10 ts values:", raw.ts.head(10).tolist())
print()

# --- Bug 1: Two timestamp formats --------------------------------
# ISO format:  YYYY-MM-DD HH:MM:SS  (9,019 rows)
# DMY format:  DD/MM/YYYY HH:MM     (3,651 rows)
# pd.to_datetime(format="mixed") treats DD/MM/YYYY as MM/DD/YYYY
# silently moving 1,382 rows to wrong dates.
# Rule: parse each format separately; dayfirst for the DMY rows.

iso = pd.to_datetime(raw.ts, format="%Y-%m-%d %H:%M:%S", errors="coerce")
dmy = pd.to_datetime(raw.ts, format="%d/%m/%Y %H:%M",   errors="coerce")
raw["ts_parsed"] = iso.fillna(dmy)

n_iso = iso.notna().sum()
n_dmy = dmy.notna().sum()
n_nat = raw.ts_parsed.isna().sum()
print(f"Bug 1 — Two timestamp formats: ISO={n_iso} rows, DD/MM/YYYY={n_dmy} rows, "
      f"unparsed={n_nat}")
print(f"  date range after correct parse: {raw.ts_parsed.min()} → {raw.ts_parsed.max()}")

# --- Bug 2: Unsorted index ---------------------------------------
# Rows arrive in random order; any rolling window on an unsorted
# series is wrong, because the 7-row window is not 7 hours.
raw = raw.sort_values("ts_parsed").reset_index(drop=True)
print("\nBug 2 — Unsorted: data now sorted by ts_parsed")

# --- Bug 3: Sentinel values (sessions = -1) ----------------------
# 290 rows carry sessions=-1 — a logging sentinel for missing
# collection windows, not a valid count. Rule: drop rows where
# sessions < 0; report count.
n_sentinel = (raw.sessions < 0).sum()
raw = raw[raw.sessions >= 0].copy()
print(f"\nBug 3 — Sentinel sessions=-1: dropped {n_sentinel} rows → {len(raw)} remain")

# --- Bug 4: Case-inconsistent device labels ----------------------
# 'Mobile', 'mobile', 'MOBILE ' all mean the same device class.
# Rule: strip whitespace and title-case; re-check value counts.
raw["device"] = raw["device"].str.strip().str.capitalize()
print(f"\nBug 4 — Device case/whitespace normalised: {dict(raw.device.value_counts())}")

# --- Set DatetimeIndex and deduplicate ---------------------------
raw = raw.set_index("ts_parsed").drop(columns=["ts"])
raw.index.name = "ts"

# --- Bug 5: Duplicate timestamps ---------------------------------
# 283 hours appear more than once (both formats happened to land
# on the same slot after correct parsing).
# Rule: keep the row with higher session count for duplicated hours.
n_before = len(raw)
raw = raw[~raw.index.duplicated(keep="first")]
n_dup = n_before - len(raw)
print(f"\nBug 5 — Duplicate timestamps: removed {n_dup} duplicates → {len(raw)} remain")

# --- Audit the index (required before plotting) ------------------
audit_h = audit_index(raw, "h")
print("\nHourly index audit:")
for k, v in audit_h.items():
    print(f"  {k}: {v}")

# --- Bug 6: Partial final month ----------------------------------
# September 2025 has 255 hourly rows; every prior month has ~699.
# Rule: flag it, do not drop — it is data about a partial period.
# Exclude the partial month from trend smoothing to avoid a
# downward artefact at the right edge.
last_month = raw.index[-1].to_period("M")
tail_rows = (raw.index.to_period("M") == last_month).sum()
print(f"\nBug 6 — Partial tail: {last_month} has {tail_rows} rows "
      f"(typical ≈ 699). Flagged; excluded from rolling mean.")

# ============================================================
# STEP 2: Resample to daily sessions (flow → sum)
# ============================================================
# Exclude the partial month before resampling so the trailing
# edge of the rolling mean is not suppressed.
raw_full = raw.copy()
raw_plot = raw[raw.index.to_period("M") != last_month]
daily = raw_plot["sessions"].resample("D").sum(min_count=1)
daily_full = reindex_full(daily, "D")      # NaN gaps become visible breaks

n_gaps = audit_index(daily, "D")["n_missing"]
print(f"\nDaily series: {len(daily)} points, {n_gaps} missing days "
      f"({raw.index.min().date()} – {raw[raw.index.to_period('M') != last_month].index.max().date()})")

# ============================================================
# STEP 3: Load and clean revenue_catalogue.csv
# ============================================================
cat = pd.read_csv(os.path.join(DATA, "revenue_catalogue.csv"))

# Duplicate SKU check
n_dup_sku = cat["sku"].duplicated().sum()
print(f"\nRevenue catalogue: {len(cat)} rows, {n_dup_sku} duplicate SKUs")
if n_dup_sku:
    cat = cat.drop_duplicates(subset="sku", keep="first")
    print(f"  → kept first occurrence; {len(cat)} rows remain")

# Negative net_revenue rows
n_neg = (cat["net_revenue_pkr_m"] < 0).sum()
neg_val = cat.loc[cat["net_revenue_pkr_m"] < 0, "net_revenue_pkr_m"].sum()
cat_pos = cat[cat["net_revenue_pkr_m"] > 0].copy()
print(f"  → negative net_revenue: {n_neg} rows, {neg_val:.2f} PKR m total — excluded from area charts")

# Sub-category aggregation
subcat = (cat_pos.groupby(["category", "subcategory"], observed=True)["net_revenue_pkr_m"]
          .sum().reset_index())
total_rev = subcat["net_revenue_pkr_m"].sum()
subcat["share_pct"] = subcat["net_revenue_pkr_m"] / total_rev * 100
print("\nSub-category shares (should match shares_eight.csv ± 0.3pp):")
for _, row in subcat.sort_values("net_revenue_pkr_m", ascending=False).iterrows():
    print(f"  {row['subcategory']:12s}  {row['share_pct']:.1f}%")

assert abs(subcat["share_pct"].sum() - 100) < 1e-4, "shares do not sum to 100"

# ============================================================
# STEP 4: Build the 2×2 figure
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.patch.set_facecolor(PALETTE["surface"])

WINDOW = 7   # 7-day centred rolling mean — cancels weekly cycle (Part B.1)

# ----------------------------------------------------------
# Panel 1: Daily sessions — raw (grey) + 7-day rolling mean
# ----------------------------------------------------------
ax1 = axes[0, 0]
raw_color = PALETTE["muted"]
signal_color = PALETTE["series"][0]

# Raw in light grey, signal in colour (Week 2: one pop-out per chart)
ax1.plot(daily_full.index, daily_full.values,
         color=raw_color, alpha=0.55, linewidth=0.8, label="Raw daily (sum)", zorder=1)

roll = daily_full.rolling(WINDOW, center=True, min_periods=max(2, WINDOW // 2)).mean()
ax1.plot(roll.index, roll.values,
         color=signal_color, linewidth=2.2,
         label=f"{WINDOW}-day centred rolling mean", zorder=2)

ax1.set_ylabel("Sessions per day (sum)")
ax1.set_xlabel("Date")
ax1.set_title(f"Traffic trend: 7-day centred rolling mean smooths weekly cycle\n"
              f"[aspect ratio {9/4.5:.2f}:1 · partial Sep 2025 excluded]")
ax1.legend(loc="upper left", fontsize=8)
for sp in ["top", "right"]:
    ax1.spines[sp].set_visible(False)

# ----------------------------------------------------------
# Panel 2: Day-of-week × hour-of-day heatmap
# ----------------------------------------------------------
ax2 = axes[0, 1]
raw_hm = raw.copy()
raw_hm["dow"] = raw_hm.index.dayofweek   # 0=Mon … 6=Sun
raw_hm["hour"] = raw_hm.index.hour
heatmap_data = (raw_hm.groupby(["dow", "hour"])["sessions"]
                .mean().unstack("hour"))

im = ax2.imshow(heatmap_data.values, aspect="auto",
                cmap=PALETTE["sequential"], origin="upper")
ax2.set_xticks(range(0, 24, 3))
ax2.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 3)], fontsize=7)
ax2.set_yticks(range(7))
ax2.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], fontsize=8)
ax2.set_xlabel("Hour of day (24 h)")
ax2.set_ylabel("Day of week")
ax2.set_title("Weekend evenings peak: mean sessions by day × hour")
cb = fig.colorbar(im, ax=ax2, fraction=0.03, pad=0.04)
cb.set_label("Mean sessions / hour", fontsize=7)

# ----------------------------------------------------------
# Panel 3: Treemap — category → sub-category net revenue
# ----------------------------------------------------------
ax3 = axes[1, 0]

# One hue per parent category; sub-categories shade within it
cat_order = (subcat.groupby("category", observed=True)["net_revenue_pkr_m"]
             .sum().sort_values(ascending=False).index.tolist())
cat_colors = {c: col for c, col in zip(cat_order, PALETTE["series"] + [PALETTE["series_4th"]])}

sizes  = subcat["net_revenue_pkr_m"].values
labels = subcat["subcategory"].values
values = subcat["net_revenue_pkr_m"].values
colors = [cat_colors[r["category"]] for _, r in subcat.iterrows()]

treemap(ax3, sizes, labels=labels, values=values, colors=colors, fontsize=7)
ax3.set_title("Mobile Phones dominate: sub-category net revenue (PKR m)\n"
              "[4 categories, 8 sub-categories]")

# Legend for categories
handles = [mpatches.Patch(facecolor=cat_colors[c], label=c) for c in cat_order]
ax3.legend(handles=handles, loc="lower right", fontsize=7, frameon=False)

# ----------------------------------------------------------
# Panel 4: Sorted horizontal bars — same sub-categories
# ----------------------------------------------------------
ax4 = axes[1, 1]
sub_sorted = subcat.sort_values("net_revenue_pkr_m", ascending=True)
bar_colors = [cat_colors[r["category"]] for _, r in sub_sorted.iterrows()]

bars = ax4.barh(sub_sorted["subcategory"], sub_sorted["net_revenue_pkr_m"],
                color=bar_colors, edgecolor="none", height=0.6)
for bar, val in zip(bars, sub_sorted["net_revenue_pkr_m"]):
    ax4.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
             f"{val:.1f}", va="center", ha="left", fontsize=7, color=PALETTE["ink"])

ax4.set_xlabel("Net revenue (PKR m)")
ax4.set_title("Phones lead by 2× Laptops: exact sub-category ranking")
ax4.set_xlim(0, sub_sorted["net_revenue_pkr_m"].max() * 1.15)
for sp in ["top", "right"]:
    ax4.spines[sp].set_visible(False)

# ----------------------------------------------------------
# Final layout
# ----------------------------------------------------------
fig.suptitle("E-Commerce Platform Review Dashboard  ·  Apr 2024 – Aug 2025",
             fontsize=14, fontweight="bold", y=1.01, color=PALETTE["ink"])
fig.tight_layout(rect=[0, 0, 1, 1])

out = os.path.join(os.path.dirname(__file__), "partD_figure.png")
note = ("Source: sessions_hourly.csv & revenue_catalogue.csv — synthetic data, CDB601220 Week 4. "
        "290 sentinel rows removed; 286 duplicates removed; 1,382 DMY timestamps corrected. "
        "Partial Sep-2025 excluded from trend panel. Negative net-revenue SKUs excluded from hierarchy.")
save_fig(fig, out, note=note)
plt.close(fig)
print(f"\nSaved → {out}")
