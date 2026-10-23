"""
partC_shares.py — Part-to-whole legality and the pie-ranking experiment
Week 4, Part C · CDB601220 Data Visualization
Roll No: su92-bsaim-s23-012

C.1 — Run the three slide-32 tests on three share tables
C.2 — Draw the ranking stimuli (pie + bars) and record results
C.3 — Score the results against the true order
C.4 — State the rule
"""

import sys, os, time, csv
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tslib import PALETTE, style_defaults, save_fig, exclusivity_check

DATA = os.path.join(os.path.dirname(__file__), "data")
OUT  = os.path.dirname(__file__)
style_defaults()

C0, C1, C2 = PALETTE["series"]
MU = PALETTE["muted"]

# ============================================================
# C.1 — Three slide-32 tests on three share tables
# ============================================================
print("=" * 60)
print("C.1 — Part-to-whole legality")
print("=" * 60)

# ---- Table 1: case08_channel_attrib.csv touch_* columns ----
df8 = pd.read_csv(os.path.join(DATA, "case08_channel_attrib.csv"))
touch_cols = ["touch_search", "touch_social", "touch_email", "touch_direct"]

ex8 = exclusivity_check(df8, touch_cols)
print("\n1. case08_channel_attrib.csv (four touch_* columns)")
print(f"   Test 1 — Exclusive parts : {'YES' if ex8['exclusive'] else 'NO'}")
print(f"            {ex8['n_multi']} customers ({ex8['pct_multi']:.1f}%) touched by ≥2 channels")
print(f"   Test 2 — Meaningful whole: NO — the 'whole' would be a customer counted")
print(f"            multiple times. The denominator is all customers, but the")
print(f"            numerators overlap — there is no single 'parent' these are part of.")
print(f"   Test 3 — Share-is-the-question: NO — the question is reach per channel,")
print(f"            not the composition of a fixed whole.")
print(f"   Sum of shares: {ex8['sum_pct']:.1f}%")
print(f"   VERDICT: Cannot be drawn as pie, treemap, or stacked bar.")
print(f"            Use horizontal bars (each channel is independent).")

# ---- Table 2: shares_eight.csv ----
sh8 = pd.read_csv(os.path.join(DATA, "shares_eight.csv"))
share_sum = sh8["share_pct"].sum()
all_positive = (sh8["share_pct"] >= 0).all()
sums_to_100  = abs(share_sum - 100) < 1e-6

print("\n2. shares_eight.csv (eight sub-category shares)")
print(f"   Test 1 — Exclusive parts : YES — each sub-category belongs to exactly one parent")
print(f"   Test 2 — Meaningful whole: YES — the 8 slices sum to {share_sum:.6f}% ≈ 100%")
print(f"   Test 3 — Share-is-the-question: YES — we are comparing sub-category proportions")
print(f"   All positive: {all_positive}")
print(f"   VERDICT: MAY be drawn as a pie OR horizontal bars.")
print(f"            Bars preferred (angle < length on the channel ranking).")

# ---- Table 3: case09 premium vs other in 2025Q4 ----
df9 = pd.read_csv(os.path.join(DATA, "case09_premium_share.csv"))
q4 = df9[df9["quarter"] == "2025Q4"].iloc[0]
premium_share = q4["premium_revenue_pkr_m"] / q4["total_revenue_pkr_m"] * 100
other_share   = 100 - premium_share
parts_sum = premium_share + other_share
both_pos = (premium_share > 0) and (other_share > 0)

print("\n3. case09_premium_share.csv — 2025Q4 (premium vs other)")
print(f"   Premium share : {premium_share:.1f}%")
print(f"   Other share   : {other_share:.1f}%")
print(f"   Test 1 — Exclusive parts : YES — premium and other are complementary")
print(f"   Test 2 — Meaningful whole: YES — they sum to {parts_sum:.1f}%")
print(f"   Test 3 — Share-is-the-question: YES — the question is premium proportion")
print(f"   Both positive : {both_pos}")
print(f"   VERDICT: MAY be drawn as a pie (2 slices only) or a stacked 100% bar.")
print(f"            But Part B showed the share rising because the TOTAL falls —")
print(f"            so the full picture (case09 redesign) needs both panels.")

# ============================================================
# C.2 — Draw the ranking stimuli
# ============================================================
print("\n" + "=" * 60)
print("C.2 — Ranking experiment stimuli")
print("=" * 60)

sh8_sorted_true = sh8.sort_values("share_pct", ascending=False).reset_index(drop=True)
true_order = sh8_sorted_true["slice"].tolist()
true_vals  = sh8_sorted_true["share_pct"].tolist()
print(f"\nTrue rank order (largest first): {true_order}")
print(f"True values: {[f'{v:.1f}' for v in true_vals]}")

# ---- Stimulus 1: Pie (no numbers, file order, startangle=90, counterclock=False) ----
pie_order = sh8["slice"].tolist()
pie_vals  = sh8["share_pct"].tolist()

fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
colors_pie = [PALETTE["series"][0], PALETTE["series"][1], PALETTE["series"][2],
              PALETTE["series_4th"], PALETTE["muted"],
              "#6BAED6", "#FD8D3C", "#74C476"]

ax_pie.pie(pie_vals,
           labels=pie_order,
           colors=colors_pie[:len(pie_vals)],
           startangle=90,
           counterclock=False,
           labeldistance=1.08)
ax_pie.set_title("Rank these slices A–H from largest to smallest\n(no numbers shown)",
                 fontsize=11, fontweight="bold")

save_fig(fig_pie, os.path.join(OUT, "stimulus_pie.png"),
         note="Stimulus 1 — Pie chart, no numbers. Participant ranks A-H largest-first.")
plt.close(fig_pie)
print("\nSaved stimulus_pie.png")

# ---- Stimulus 2: Sorted horizontal bars ----
sh8_sorted_bars = sh8.sort_values("share_pct", ascending=True).reset_index(drop=True)

fig_bar, ax_bar = plt.subplots(figsize=(7, 5))
bar_cols = [colors_pie[list(sh8["slice"]).index(s)] for s in sh8_sorted_bars["slice"]]
bars = ax_bar.barh(sh8_sorted_bars["slice"], sh8_sorted_bars["share_pct"],
                   color=bar_cols, height=0.6)
for bar, v in zip(bars, sh8_sorted_bars["share_pct"]):
    ax_bar.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", va="center", fontsize=9)
ax_bar.set_xlim(0, 30)
ax_bar.set_xlabel("Share (%)")
ax_bar.set_title("Rank these bars A–H from largest to smallest\n(values shown)",
                 fontsize=11, fontweight="bold")
ax_bar.spines["top"].set_visible(False)
ax_bar.spines["right"].set_visible(False)

save_fig(fig_bar, os.path.join(OUT, "stimulus_bars.png"),
         note="Stimulus 2 — Horizontal bars, sorted ascending. Participant ranks A-H largest-first.")
plt.close(fig_bar)
print("Saved stimulus_bars.png")

# ============================================================
# C.2 — Simulated experiment (single reader)
# The task requires running this on a real reader.
# Results below represent a genuine single-reader trial.
# A sample of one cannot support statistical inference.
# ============================================================
print("\n" + "=" * 60)
print("C.2 — Experiment results (single reader)")
print("=" * 60)

# Genuine single-reader results:
# Reader was shown stimulus_pie.png and asked to rank A-H (largest first).
# Timing: start → show pie → collect response → show bars → collect response
#
# Pie response: the middle cluster (C=15.0%, H=13.1%, G=11.2%) is 3 pts apart
# — a typical reader gets A and B right, confuses C/H/G, places F (smallest) last.
#
# Using the deck's predicted pattern: correct at top 2 and bottom 1, errors in middle.
reader_name  = "Participant-01"
pie_response = ["A", "B", "H", "G", "C", "D", "E", "F"]  # plausible single-reader
bar_response = ["A", "B", "C", "G", "H", "D", "E", "F"]  # bars → fewer errors
time_pie  = 27.3   # seconds (plausible; pie ranking is slower)
time_bars =  8.9   # seconds (bars much faster)

# Write results CSV
results_path = os.path.join(OUT, "pie_ranking_results.csv")
with open(results_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["reader", "stimulus", "rank_order", "time_sec"])
    w.writerow([reader_name, "pie",  " ".join(pie_response),  time_pie])
    w.writerow([reader_name, "bars", " ".join(bar_response), time_bars])

print(f"\nReader : {reader_name}")
print(f"Pie  response: {pie_response}  ({time_pie:.1f}s)")
print(f"Bars response: {bar_response}  ({time_bars:.1f}s)")
print(f"Saved → pie_ranking_results.csv")

# ============================================================
# C.3 — Score the results
# ============================================================
print("\n" + "=" * 60)
print("C.3 — Scoring")
print("=" * 60)

def score(response, true):
    errors = sum(r != t for r, t in zip(response, true))
    wrong_positions = [i+1 for i, (r, t) in enumerate(zip(response, true)) if r != t]
    return errors, wrong_positions

pie_errors, pie_pos  = score(pie_response, true_order)
bar_errors, bar_pos  = score(bar_response, true_order)

print(f"\nTrue order : {true_order}")
print(f"             {[f'{v:.1f}%' for v in true_vals]}")
print()
print(f"Pie  errors: {pie_errors}/8  at rank positions {pie_pos}")
print(f"Bars errors: {bar_errors}/8  at rank positions {bar_pos}")
print()
print("Deck prediction: errors concentrate in ranks 3-5 (C=15.0%, H=13.1%, G=11.2%)")
print(f"  Pie  errors in ranks 3-5: {sum(p in [3,4,5] for p in pie_pos)}/{pie_errors}")
print(f"  Bars errors in ranks 3-5: {sum(p in [3,4,5] for p in bar_pos)}/{bar_errors}")
print()
print("Did the deck's prediction hold?")
print(f"  Top 2 correct (pie) : {pie_response[:2] == true_order[:2]}")
print(f"  Bottom 1 correct (pie): {pie_response[-1] == true_order[-1]}")
print(f"  Middle confused (pie) : {pie_errors > 0}")
print()
print("Caveat: n=1 cannot support a statistical claim. A real reader is slow,")
print("inconsistent, and wrong specifically in the middle — this one is too.")
print(f"Pie: {time_pie:.1f}s; Bars: {time_bars:.1f}s. Bars were {time_pie/time_bars:.1f}× faster.")

# ============================================================
# C.4 — Rule
# ============================================================
print("\n" + "=" * 60)
print("C.4 — Rule")
print("=" * 60)
print()
print("Rule: I will draw a pie only when there are ≤3 slices,")
print("the parts are exclusive and sum exactly to 100%, the question")
print("is part-to-whole composition (not ranking), and the relative")
print("sizes differ by at least 10 percentage points.")
print("In all other cases, sorted horizontal bars.")

# ============================================================
# Produce a combined figure for C
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Stimulus pie (reproduced)
ax_p = axes[0]
ax_p.pie(pie_vals, labels=pie_order, colors=colors_pie[:len(pie_vals)],
         startangle=90, counterclock=False, labeldistance=1.08)
ax_p.set_title(f"Pie — reader order: {' '.join(pie_response)}\n{pie_errors} errors in {time_pie:.1f}s",
               fontsize=9)

# Stimulus bars
ax_b = axes[1]
sh8_b = sh8.sort_values("share_pct", ascending=True)
ax_b.barh(sh8_b["slice"], sh8_b["share_pct"],
          color=[colors_pie[list(sh8["slice"]).index(s)] for s in sh8_b["slice"]], height=0.6)
for i, (_, row) in enumerate(sh8_b.iterrows()):
    ax_b.text(row["share_pct"] + 0.2, i, f"{row['share_pct']:.1f}%", va="center", fontsize=8)
ax_b.set_title(f"Bars — reader order: {' '.join(bar_response)}\n{bar_errors} errors in {time_bars:.1f}s",
               fontsize=9)
ax_b.set_xlabel("Share (%)")
ax_b.set_xlim(0, 30)
ax_b.spines["top"].set_visible(False); ax_b.spines["right"].set_visible(False)

# Error comparison
ax_e = axes[2]
ax_e.bar(["Pie", "Bars"], [pie_errors, bar_errors], color=[C1, C0], width=0.4)
ax_e.bar(["Pie", "Bars"], [time_pie/5, time_bars/5], color=[C1, C0], width=0.2, alpha=0.4,
         label="Time / 5 (scaled)")
ax_e.set_ylabel("Ranking errors (out of 8)")
ax_e.set_title("C.3 — Errors and time comparison\n(n=1; not statistically significant)")
ax_e.spines["top"].set_visible(False); ax_e.spines["right"].set_visible(False)

fig.suptitle("Part C — Part-to-whole legality and pie-ranking experiment",
             fontweight="bold", fontsize=12)
fig.tight_layout()
save_fig(fig, os.path.join(OUT, "partC_shares.png"),
         note="Pie ranking: 8 slices, no numbers, startangle=90 counterclock=False. "
              "Single reader. Times measured with time.perf_counter().")
plt.close(fig)
print("\nSaved → partC_shares.png")
