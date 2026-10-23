"""
partA_charts.py — Rebuild nine broken charts, then redesign them
Week 4, Part A · CDB601220 Data Visualization
Roll No: su92-bsaim-s23-012

Layout: 9 rows × 2 columns
  Left column  = broken chart (exact recipe)
  Right column = honest redesign
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

from tslib import PALETTE, style_defaults, save_fig, audit_index, reindex_full

DATA = os.path.join(os.path.dirname(__file__), "data")
OUT  = os.path.dirname(__file__)
style_defaults()

C0 = PALETTE["series"][0]   # blue
C1 = PALETTE["series"][1]   # orange
C2 = PALETTE["series"][2]   # green
MU = PALETTE["muted"]


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ================================================================
#  CASE 01 · "Orders peak in Q1 and collapse every year"
# ================================================================
df1 = pd.read_csv(os.path.join(DATA, "case01_orders_stringdate.csv"))

fig, (ax_b, ax_r) = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle("Case 01 · String-sorted date column", fontweight="bold", fontsize=11)

# --- BROKEN ---
df1_broken = df1.sort_values("month")   # lexicographic text sort
ax_b.plot(range(len(df1_broken)), df1_broken["orders"],
          color=C0, marker="o", markersize=4)
ax_b.set_xticks(range(len(df1_broken)))
ax_b.set_xticklabels(df1_broken["month"], rotation=90, fontsize=6)
ax_b.set_ylim(0, None)
ax_b.set_title("BROKEN · text-sorted 'YYYY-M' column\n"
               "Self-check: 24 pts · drops of 958 and 978", fontsize=9)
ax_b.set_ylabel("Orders")
_clean(ax_b)

# verify self-check
drops_b = df1_broken["orders"].diff().dropna()
big_drops = drops_b.nsmallest(2).values
print(f"Case 01 self-check drops: {big_drops}")   # should be ≈ -958, -978

# --- REDESIGN ---
df1_r = df1.copy()
df1_r["dt"] = pd.to_datetime(
    df1_r["month"].str.replace(r"-(\d)$", r"-0\1", regex=True),
    format="%Y-%m")
df1_r = df1_r.sort_values("dt")
ax_r.plot(df1_r["dt"], df1_r["orders"], color=C0, marker="o", markersize=4)
ax_r.set_ylim(0, None)
ax_r.xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: pd.Timestamp(x, unit="D").strftime("%b %Y")))
plt.setp(ax_r.get_xticklabels(), rotation=45, ha="right", fontsize=7)
ax_r.set_title("REDESIGN · Parsed to DatetimeIndex — real seasonal cycle\n"
               "No Q1 collapse: the dip in Dec–Jan is the same every year", fontsize=9)
ax_r.set_ylabel("Orders")
_clean(ax_r)

fig.tight_layout()
save_fig(fig, os.path.join(OUT, "case01.png"),
         note="Mechanism: 'YYYY-M' sorted as text places Jan-1 at row 0 and Oct-1 at row 1 "
              "(lexicographic: '2024-1' < '2024-10' < '2024-2'). "
              "Affected: all 24 rows. Two 'drops' are sorting artefacts, not real falls.")
plt.close(fig)
print("Saved case01.png")


# ================================================================
#  CASE 02 · "Sales grew steadily, 22%, across the quarter"
# ================================================================
df2 = pd.read_csv(os.path.join(DATA, "case02_shop_closed.csv"))
df2["dt"] = pd.to_datetime(df2["date"])

fig, (ax_b, ax_r) = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle("Case 02 · Shop-closed days invisible on row-number axis", fontweight="bold", fontsize=11)

# --- BROKEN ---
x_b = range(len(df2))
slope_b, intercept_b, *_ = stats.linregress(list(x_b), df2["sales_pkr_k"])
ax_b.plot(x_b, df2["sales_pkr_k"], color=C0, linewidth=1.2)
trend_x = np.array([0, len(df2)-1])
ax_b.plot(trend_x, intercept_b + slope_b * trend_x, color=C1, linewidth=2, label="Trend")
ax_b.set_ylim(0, None)
ax_b.set_title("BROKEN · Row number as x-axis\n"
               f"Self-check: 69 pts · first {df2.sales_pkr_k.iloc[0]} · last {df2.sales_pkr_k.iloc[-1]}", fontsize=9)
ax_b.set_ylabel("Sales (PKR k)")
ax_b.set_xlabel("Row number")
ax_b.legend(fontsize=8)
_clean(ax_b)

# --- REDESIGN ---
full_idx = pd.date_range(df2["dt"].min(), df2["dt"].max(), freq="D")
df2_r = df2.set_index("dt")["sales_pkr_k"].reindex(full_idx)
missing_days = df2_r.isna().sum()

ax_r.plot(df2_r.index, df2_r.values, color=C0, linewidth=1.2, label="Trading days")
ax_r.fill_between(df2_r.index, 0,
                  [df2_r.max() * 0.15 if np.isnan(v) else 0 for v in df2_r.values],
                  color=C1, alpha=0.35, label=f"Closed days (n={missing_days})")
days_since = (df2["dt"] - df2["dt"].min()).dt.days
slope_r, intercept_r, *_ = stats.linregress(days_since, df2["sales_pkr_k"])
trend_dates = pd.date_range(df2["dt"].min(), df2["dt"].max(), freq="D")
trend_days  = (trend_dates - df2["dt"].min()).days
ax_r.plot(trend_dates, intercept_r + slope_r * trend_days,
          color=C2, linewidth=2, label="Trend (calendar)")
ax_r.set_ylim(0, None)
ax_r.set_title(f"REDESIGN · Real dates — {missing_days} closed days shown\n"
               "Trend slope halves when calendar spacing is respected", fontsize=9)
ax_r.set_ylabel("Sales (PKR k)")
ax_r.set_xlabel("Date")
ax_r.legend(fontsize=8)
plt.setp(ax_r.get_xticklabels(), rotation=30, ha="right")
_clean(ax_r)

fig.tight_layout()
save_fig(fig, os.path.join(OUT, "case02.png"),
         note="Mechanism: row-number axis compresses 90 calendar days into 69 equal slots, "
              "erasing 21 closed days. Trend slope on row numbers is steeper than on real time. "
              f"Affected: all {missing_days} missing days (weekends + holidays).")
plt.close(fig)
print("Saved case02.png")


# ================================================================
#  CASE 03 · "Peak load jumped 43% overnight on 1 July"
# ================================================================
df3 = pd.read_csv(os.path.join(DATA, "case03_bandwidth_levels.csv"))
df3["dt"] = pd.to_datetime(df3["ts"])
df3 = df3.set_index("dt")

fig, (ax_b, ax_r) = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle("Case 03 · Level resampled with sum()", fontweight="bold", fontsize=11)

# --- BROKEN ---
broken_daily = df3["concurrent_users"].resample("D").sum()
jun_mean = broken_daily["2025-06"].mean()
jul_mean = broken_daily["2025-07"].mean()
print(f"Case 03 self-check: Jun daily mean={jun_mean:.0f}  Jul daily mean={jul_mean:.0f}  "
      f"rise={(jul_mean/jun_mean-1)*100:.0f}%")

ax_b.plot(broken_daily.index, broken_daily.values, color=C0, linewidth=1.5)
ax_b.axvline(pd.Timestamp("2025-07-01"), color=C1, linewidth=1.5, linestyle="--", label="1 Jul")
ax_b.set_ylim(0, None)
ax_b.set_title("BROKEN · concurrent_users resampled with .sum()\n"
               f"Jun mean≈{jun_mean:,.0f}  Jul mean≈{jul_mean:,.0f}  (+43%)", fontsize=9)
ax_b.set_ylabel("Daily sum of concurrent_users (wrong)")
ax_b.legend(fontsize=8)
plt.setp(ax_b.get_xticklabels(), rotation=30, ha="right")
_clean(ax_b)

# --- REDESIGN ---
correct_daily = df3["concurrent_users"].resample("D").mean()
flow_daily    = df3["bytes_served"].resample("D").sum()
jun_m2 = correct_daily["2025-06"].mean()
jul_m2 = correct_daily["2025-07"].mean()

ax_r.plot(correct_daily.index, correct_daily.values, color=C0, linewidth=1.5,
          label="concurrent_users (daily mean — correct)")
ax_r.axvline(pd.Timestamp("2025-07-01"), color=C1, linewidth=1.5, linestyle="--", label="1 Jul")
ax_r.set_ylim(0, None)
ax_r.set_title(f"REDESIGN · .mean() for levels (June≈{jun_m2:.0f}, July≈{jul_m2:.0f})\n"
               "No overnight jump: the +43% was 24× hour-count, not usage", fontsize=9)
ax_r.set_ylabel("Daily mean concurrent_users")
ax_r.set_xlabel("Date")
ax_r.legend(fontsize=8)
plt.setp(ax_r.get_xticklabels(), rotation=30, ha="right")
_clean(ax_r)

fig.tight_layout()
save_fig(fig, os.path.join(OUT, "case03.png"),
         note="Mechanism: concurrent_users is a level (instantaneous count). "
              "Summing 24 hourly levels gives a number 24× the mean—not a meaningful daily total. "
              "July has 31 days vs June's 30, adding another ×(31/30) multiplier. "
              "Use .mean() for levels; .sum() only for flows (bytes_served).")
plt.close(fig)
print("Saved case03.png")


# ================================================================
#  CASE 04 · "The new checkout flow started working on 7 September"
# ================================================================
df4 = pd.read_csv(os.path.join(DATA, "case04_intervention_lag.csv"))
df4["dt"] = pd.to_datetime(df4["date"])
df4 = df4.set_index("dt")["conversion_pct"]
WINDOW4 = 31

trailing = df4.rolling(WINDOW4).mean()
centred  = df4.rolling(WINDOW4, center=True).mean()
trail_bottom = trailing.idxmin()
cent_bottom  = centred.idxmin()
lag = int((trail_bottom - cent_bottom).days)
print(f"Case 04 self-check: trailing bottom {trail_bottom.date()}  "
      f"centred bottom {cent_bottom.date()}  lag={lag} days (half-window={WINDOW4//2})")

fig, (ax_b, ax_r) = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle("Case 04 · Trailing mean's half-window lag", fontweight="bold", fontsize=11)

# --- BROKEN ---
ax_b.plot(df4.index, df4.values, color=MU, alpha=0.5, linewidth=0.8, label="Raw")
ax_b.plot(trailing.index, trailing.values, color=C0, linewidth=2,
          label=f"31-day trailing mean")
ax_b.axvline(trail_bottom, color=C1, linewidth=1.5, linestyle="--",
             label=f"Min: {trail_bottom.date()}")
ax_b.set_title(f"BROKEN · 31-day trailing mean\n"
               f"Self-check: bottom on {trail_bottom.date()}", fontsize=9)
ax_b.set_ylabel("Conversion rate (%)")
ax_b.legend(fontsize=7)
plt.setp(ax_b.get_xticklabels(), rotation=30, ha="right")
_clean(ax_b)

# --- REDESIGN ---
ax_r.plot(df4.index, df4.values, color=MU, alpha=0.5, linewidth=0.8, label="Raw")
ax_r.plot(centred.index, centred.values, color=C2, linewidth=2,
          label="31-day centred mean")
ax_r.plot(trailing.index, trailing.values, color=C0, linewidth=1.2, linestyle=":",
          alpha=0.6, label="31-day trailing (shown for comparison)")
ax_r.axvline(cent_bottom, color=C2, linewidth=2, linestyle="--",
             label=f"Real bottom: {cent_bottom.date()}")
ax_r.axvline(trail_bottom, color=C0, linewidth=1.2, linestyle=":",
             alpha=0.6, label=f"Trailing bottom: {trail_bottom.date()}")
ax_r.set_title(f"REDESIGN · Centred mean — real turning point {cent_bottom.date()}\n"
               f"Trailing mean was {lag} days late (= half-window = {WINDOW4//2})", fontsize=9)
ax_r.set_ylabel("Conversion rate (%)")
ax_r.legend(fontsize=7)
plt.setp(ax_r.get_xticklabels(), rotation=30, ha="right")
_clean(ax_r)

fig.tight_layout()
save_fig(fig, os.path.join(OUT, "case04.png"),
         note=f"Mechanism: a {WINDOW4}-day trailing mean can only detect a turn once the full "
              f"window has accumulated past it — producing a {WINDOW4//2}-day lag. "
              f"Centred mean places the turning point at {cent_bottom.date()}, "
              f"not {trail_bottom.date()}.")
plt.close(fig)
print("Saved case04.png")


# ================================================================
#  CASE 05 · "Revenue has fallen for six straight months"
# ================================================================
df5 = pd.read_csv(os.path.join(DATA, "case05_revenue_window.csv"))
df5["dt"] = pd.to_datetime(df5["month"])
df5 = df5.set_index("dt")["revenue_pkr_m"]

window5 = df5["2025-05":"2025-11"]
pct_fall = (window5.iloc[-1] / window5.iloc[0] - 1) * 100
print(f"Case 05 self-check: 7 pts, {window5.iloc[0]} → {window5.iloc[-1]}, {pct_fall:.1f}%")

fig, (ax_b, ax_r) = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle("Case 05 · Cherry-picked window starts at seasonal peak", fontweight="bold", fontsize=11)

# --- BROKEN ---
ax_b.plot(window5.index, window5.values, color=C0, marker="o", markersize=5)
ax_b.set_ylim(0, None)
ax_b.set_title(f"BROKEN · May–Nov 2025 only, from zero\n"
               f"Self-check: {window5.iloc[0]} → {window5.iloc[-1]} ({pct_fall:.1f}%)", fontsize=9)
ax_b.set_ylabel("Revenue (PKR m)")
plt.setp(ax_b.get_xticklabels(), rotation=30, ha="right")
_clean(ax_b)

# --- REDESIGN ---
ax_r.plot(df5.index, df5.values, color=MU, alpha=0.5, linewidth=0.8, label="All 36 months")
ax_r.plot(window5.index, window5.values, color=C1, linewidth=2, marker="o", markersize=5,
          label="Reported window (May–Nov 2025)")
may_pk = float(df5["2024-05-01"])
ax_r.axhline(may_pk, color=C2, linewidth=1, linestyle="--", alpha=0.6,
             label=f"May 2024 level ({may_pk:.1f}) — same seasonal peak")
ax_r.set_ylim(0, None)
ax_r.set_title("REDESIGN · Full 36 months — same drop repeats every year\n"
               "The window starts at the seasonal peak; the fall is normal seasonality", fontsize=9)
ax_r.set_ylabel("Revenue (PKR m)")
ax_r.legend(fontsize=8)
plt.setp(ax_r.get_xticklabels(), rotation=30, ha="right")
_clean(ax_r)

fig.tight_layout()
save_fig(fig, os.path.join(OUT, "case05.png"),
         note="Mechanism: the 7-month window starts at May, the seasonal peak (158.8 PKR m — "
              "the maximum of all 36 months). Every year shows a May-to-Nov decline of similar "
              "magnitude. The window was chosen after the data was seen; it is not a method.")
plt.close(fig)
print("Saved case05.png")


# ================================================================
#  CASE 06 · "Orders collapsed in October"
# ================================================================
df6 = pd.read_csv(os.path.join(DATA, "case06_monthly_partial.csv"))
df6["dt"] = pd.to_datetime(df6["date"])
monthly6 = df6.resample("MS", on="dt")["orders"].sum()

last_day = df6["dt"].max()
days_in_oct = pd.Timestamp("2025-10-31")
days_recorded = last_day.day
pct_oct = days_recorded / 31 * 100

print(f"Case 06 self-check: Sep={monthly6['2025-09-01']}, Oct={monthly6['2025-10-01']}, "
      f"Oct/Sep ratio={monthly6['2025-10-01']/monthly6['2025-09-01']*100:.0f}%")

fig, (ax_b, ax_r) = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle("Case 06 · Incomplete final period shown unmarked", fontweight="bold", fontsize=11)

# --- BROKEN ---
ax_b.bar(monthly6.index, monthly6.values, width=25, color=C0, align="center")
ax_b.set_ylim(0, None)
ax_b.set_title(f"BROKEN · Monthly sum with partial October\n"
               f"Self-check: Sep={monthly6['2025-09-01']:,}  Oct={monthly6['2025-10-01']:,} (37%)", fontsize=9)
ax_b.set_ylabel("Orders (monthly sum)")
plt.setp(ax_b.get_xticklabels(), rotation=30, ha="right")
_clean(ax_b)

# --- REDESIGN ---
monthly_complete = monthly6[monthly6.index < pd.Timestamp("2025-10-01")]
ax_r.bar(monthly_complete.index, monthly_complete.values,
         width=25, color=C0, align="center", label="Complete months")
ax_r.bar([pd.Timestamp("2025-10-01")], [monthly6["2025-10-01"]],
         width=25, color=C1, align="center", alpha=0.7,
         label=f"Partial Oct ({days_recorded}/31 days, {pct_oct:.0f}%)")
oct_projected = monthly6["2025-10-01"] / days_recorded * 31
ax_r.bar([pd.Timestamp("2025-10-01")], [oct_projected - monthly6["2025-10-01"]],
         bottom=monthly6["2025-10-01"],
         width=25, color=C1, align="center", alpha=0.25,
         label=f"Projected full month ({oct_projected:,.0f})")
ax_r.set_ylim(0, None)
ax_r.set_title(f"REDESIGN · October flagged as partial ({days_recorded}/31 days)\n"
               f"Projected full month ≈ {oct_projected:,.0f} — no collapse", fontsize=9)
ax_r.set_ylabel("Orders (monthly sum)")
ax_r.legend(fontsize=8)
plt.setp(ax_r.get_xticklabels(), rotation=30, ha="right")
_clean(ax_r)

fig.tight_layout()
save_fig(fig, os.path.join(OUT, "case06.png"),
         note=f"Mechanism: the extract ends on {last_day.date()} — day {days_recorded} of October. "
              f"Monthly .sum() gives {monthly6['2025-10-01']:,} orders for {days_recorded} days "
              f"vs {monthly6['2025-09-01']:,} for 30 days in September. "
              f"Annualised rate is similar. Partial final periods must be flagged.")
plt.close(fig)
print("Saved case06.png")


# ================================================================
#  CASE 07 · "Product D has been erratic since mid-2024"
# ================================================================
df7 = pd.read_csv(os.path.join(DATA, "case07_stacked_products.csv"))
df7["dt"] = pd.to_datetime(df7["month"])

total_env = df7[["product_a","product_b","product_c","product_d"]].sum(axis=1)
print(f"Case 07 self-check: total envelope {total_env.min():.0f}–{total_env.max():.0f}")

fig, axes7 = plt.subplots(1, 3, figsize=(16, 4))
ax_b, ax_r1, ax_r2 = axes7
fig.suptitle("Case 07 · Stacked-area moving baseline inflates Product D volatility",
             fontweight="bold", fontsize=11)

prods = ["product_a","product_b","product_c","product_d"]
cols  = [C0, C2, PALETTE["series_4th"], C1]

# --- BROKEN ---
ax_b.stackplot(df7["dt"],
               [df7[p] for p in prods],
               labels=["A","B","C","D"], colors=cols, alpha=0.85)
ax_b.set_ylim(0, None)
ax_b.set_title("BROKEN · Stacked area — D rides on A+B+C\n"
               "D's band bounces with B's high variance (std=35.6)", fontsize=9)
ax_b.set_ylabel("Revenue (PKR m)")
ax_b.legend(loc="upper left", fontsize=8)
plt.setp(ax_b.get_xticklabels(), rotation=30, ha="right")
_clean(ax_b)

# --- REDESIGN 1: faceted ---
for prod, col in zip(prods, cols):
    ax_r1.plot(df7["dt"], df7[prod], color=col, linewidth=1.5, label=prod.replace("product_","Prod ").upper())
ax_r1.set_ylim(0, None)
ax_r1.set_title("REDESIGN 1 · Overlapping lines — D is the stable one\n"
                "D std=3.6; B std=35.6 drives the stacked-band illusion", fontsize=9)
ax_r1.set_ylabel("Revenue (PKR m)")
ax_r1.legend(fontsize=8)
plt.setp(ax_r1.get_xticklabels(), rotation=30, ha="right")
_clean(ax_r1)

# --- REDESIGN 2: faceted small multiples ---
for i, (prod, col) in enumerate(zip(prods, cols)):
    ax_r2.plot(df7["dt"], df7[prod] - df7[prod].mean(),
               color=col, linewidth=1.5, alpha=0.8,
               label=f"{prod[-1].upper()} (σ={df7[prod].std():.1f})")
ax_r2.axhline(0, color=PALETTE["ink_soft"], linewidth=0.8, linestyle=":")
ax_r2.set_title("REDESIGN 2 · Mean-centred deviations — volatility visible directly\n"
                "B and C volatile; D is nearly flat (σ=3.6)", fontsize=9)
ax_r2.set_ylabel("Deviation from product mean (PKR m)")
ax_r2.legend(fontsize=8)
plt.setp(ax_r2.get_xticklabels(), rotation=30, ha="right")
_clean(ax_r2)

fig.tight_layout()
save_fig(fig, os.path.join(OUT, "case07.png"),
         note="Mechanism: in a stacked area chart, Product D sits on top of A+B+C. "
              "Its band height is correct (std≈3.6 PKR m) but its POSITION oscillates with "
              "B's variance (std=35.6). The eye reads position as value, so D looks erratic. "
              "Fix: overlapping lines or mean-centred deviations isolate each product.")
plt.close(fig)
print("Saved case07.png")


# ================================================================
#  CASE 08 · "Paid search drives 43% of our customers"
# ================================================================
df8 = pd.read_csv(os.path.join(DATA, "case08_channel_attrib.csv"))
touch_cols = ["touch_search","touch_social","touch_email","touch_direct"]
shares8 = (df8[touch_cols].sum() / len(df8) * 100).round(1)
total_share = shares8.sum()
per_unit = df8[touch_cols].sum(axis=1)
n_multi = (per_unit >= 2).sum()
print(f"Case 08 self-check: {shares8.to_dict()}  sum={total_share:.1f}%  multi-touch={n_multi}")

fig, (ax_b, ax_r) = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle("Case 08 · Non-exclusive parts sum to 131%", fontweight="bold", fontsize=11)

# --- BROKEN ---
labels8 = ["Search","Social","Email","Direct"]
ax_b.pie(shares8.values, labels=[f"{l}\n{v:.1f}%" for l, v in zip(labels8, shares8.values)],
         colors=[C0, C2, PALETTE["series_4th"], C1],
         startangle=90, counterclock=False)
ax_b.set_title(f"BROKEN · Pie of non-exclusive channels\n"
               f"Self-check: Search {shares8['touch_search']}% · Social {shares8['touch_social']}% · "
               f"Email {shares8['touch_email']}% · Direct {shares8['touch_direct']}% "
               f"(sum={total_share:.1f}%)", fontsize=8)

# --- REDESIGN ---
ax_r.barh(labels8[::-1], shares8.values[::-1],
          color=[C0, C2, PALETTE["series_4th"], C1][::-1], height=0.55)
for i, v in enumerate(shares8.values[::-1]):
    ax_r.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
ax_r.axvline(100, color=C1, linewidth=1.2, linestyle="--", label="100% mark")
ax_r.set_xlabel("% of customers touched")
ax_r.set_xlim(0, 55)
ax_r.set_title(f"REDESIGN · Horizontal bars (parts are NOT exclusive)\n"
               f"Sum={total_share:.1f}%: {n_multi:,} customers touched by ≥2 channels", fontsize=9)
ax_r.legend(fontsize=8)
_clean(ax_r)

fig.tight_layout()
save_fig(fig, os.path.join(OUT, "case08.png"),
         note=f"Mechanism: the four touch_* columns are not mutually exclusive. "
              f"{n_multi:,} customers ({n_multi/len(df8)*100:.1f}%) appear in ≥2 channels. "
              f"Shares sum to {total_share:.1f}%, violating the part-to-whole condition. "
              f"A pie's slices must sum to 100%; these are set-membership counts, not shares.")
plt.close(fig)
print("Saved case08.png")


# ================================================================
#  CASE 09 · "Premium is now 38% of revenue — a record"
# ================================================================
df9 = pd.read_csv(os.path.join(DATA, "case09_premium_share.csv"))
df9["share"] = (df9["premium_revenue_pkr_m"] / df9["total_revenue_pkr_m"] * 100).round(1)
print(f"Case 09 self-check: share {df9['share'].iloc[0]}% → {df9['share'].iloc[-1]}%")

fig, (ax_b, ax_r) = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle("Case 09 · Rising share of a falling total", fontweight="bold", fontsize=11)

quarters = range(len(df9))
qlabels  = df9["quarter"].tolist()

# --- BROKEN ---
ax_b.plot(quarters, df9["share"], color=C0, marker="o", markersize=5)
ax_b.annotate(f"{df9['share'].iloc[-1]:.1f}%",
              xy=(len(df9)-1, df9['share'].iloc[-1]),
              xytext=(len(df9)-1-0.5, df9['share'].iloc[-1]+1.5),
              fontsize=9, color=C1)
ax_b.set_ylim(0, None)
ax_b.set_xticks(quarters)
ax_b.set_xticklabels(qlabels, rotation=45, ha="right", fontsize=7)
ax_b.set_title(f"BROKEN · Rising share ({df9['share'].iloc[0]}% → {df9['share'].iloc[-1]}%)\n"
               "Implies premium line is growing", fontsize=9)
ax_b.set_ylabel("Premium share of revenue (%)")
_clean(ax_b)

# --- REDESIGN ---
ax_r.fill_between(quarters, df9["total_revenue_pkr_m"],
                  color=MU, alpha=0.35, label="Other revenue (PKR m)")
ax_r.fill_between(quarters, df9["premium_revenue_pkr_m"],
                  color=C0, alpha=0.75, label="Premium revenue (PKR m)")
ax_r.set_ylim(0, None)
ax_r2b = ax_r.twinx()
ax_r2b.plot(quarters, df9["share"], color=C1, linewidth=2, marker="o", markersize=4,
            linestyle="--", label="Premium share % (right)")
ax_r2b.set_ylabel("Premium share (%)", color=C1)
ax_r2b.tick_params(axis="y", labelcolor=C1)
ax_r2b.set_ylim(0, 60)
ax_r.set_xticks(quarters)
ax_r.set_xticklabels(qlabels, rotation=45, ha="right", fontsize=7)
ax_r.set_ylabel("Revenue (PKR m)")
ax_r.set_title("REDESIGN · Share rises because total shrinks 40%\n"
               "Premium revenue nearly flat (240→228 PKR m)", fontsize=9)
lines1, labels1 = ax_r.get_legend_handles_labels()
lines2, labels2 = ax_r2b.get_legend_handles_labels()
ax_r.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

fig.tight_layout()
save_fig(fig, os.path.join(OUT, "case09.png"),
         note="Mechanism: total_revenue_pkr_m fell 40% (1000→600 PKR m) over 8 quarters. "
              "Premium revenue is nearly flat (240→228 PKR m, −5%). "
              "The rising share is a mathematical consequence of the shrinking denominator, "
              "not growth in the premium line. A rising share of a falling whole is not a record.")
plt.close(fig)
print("Saved case09.png")

print("\nAll 9 cases complete.")
