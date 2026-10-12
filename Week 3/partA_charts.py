"""
Part A — Rebuild nine charts that are wrong, then take them apart  (36 marks)

For each of the nine cases this script:
  1. Loads the data and reproduces the analyst's chart exactly.
  2. Prints the self-check number so you can confirm the rebuild.
  3. Prints the diagnosis (conclusion, decoy, mechanism, with numbers).
  4. Saves the analyst's chart and the redesign as PNGs.

Run:
    python3 partA_charts.py

All output files land in the same directory as this script.
"""

from __future__ import annotations

import os
import sys
import textwrap

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vizlib import MARKERS, PALETTE, save_fig, style_defaults   # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
style_defaults()
C = PALETTE["series"]
C4 = PALETTE["series_4th"]


def _save(fig: plt.Figure, name: str, note: str = "") -> None:
    path = os.path.join(ROOT, name)
    save_fig(fig, path, note or "Synthetic data — CDB601220")
    plt.close(fig)
    print(f"  wrote {name}")


def _section(n: int, title: str) -> None:
    print(f"\n{'='*72}")
    print(f"  CASE {n} — {title}")
    print(f"{'='*72}")


# =============================================================================
# Case 1 — "Premium-certified helmets have 2.5× the head-injury rate"
# =============================================================================
def case01():
    _section(1, "Premium helmets have 2.5× the head-injury rate")
    df = pd.read_csv(os.path.join(DATA, "case01_helmet_rickshaw.csv"))

    # ── Rebuild: injuries per 1,000 trips by helmet_grade ──────────────────
    summary = df.groupby("helmet_grade")[["head_injuries", "trips"]].sum()
    summary["rate_per_1k"] = summary.head_injuries / summary.trips * 1000
    print("  Self-check (Standard ~2.15, Premium ~5.40):")
    print(summary["rate_per_1k"].to_string())

    fig, ax = plt.subplots(figsize=(5, 4))
    grades = summary.index.tolist()
    bars = ax.bar(grades, summary["rate_per_1k"], color=[C[0], C[1]], width=0.5)
    for bar, val in zip(bars, summary["rate_per_1k"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05, f"{val:.2f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, summary["rate_per_1k"].max() * 1.25)
    ax.set_ylabel("Head injuries per 1,000 trips")
    ax.set_title("Premium helmets: 2.5× the head-injury rate")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    _save(fig, "case01_analyst.png")

    # ── Diagnosis ────────────────────────────────────────────────────────────
    print("\n  CONCLUSION the chart invites:")
    print("    Premium helmets cause more head injuries — upgrade your helmet to\n"
          "    a worse one.")
    print("\n  DECOY (first objection and why it does not hold):")
    print("    'Correlation is not causation.' True, but it names no mechanism.\n"
          "    The direction of causation could be reversed — maybe injured riders\n"
          "    switch to premium helmets after an injury. But this is the same\n"
          "    weak inference. We need to find the actual third variable.")

    # Mechanism: Simpson's paradox via road_type and avg_speed_kph
    road_summary = df.groupby(["helmet_grade", "road_type"])[
        ["head_injuries", "trips"]].sum().reset_index()
    road_summary["rate_per_1k"] = (
        road_summary.head_injuries / road_summary.trips * 1000
    )
    grade_road = road_summary.pivot(index="road_type", columns="helmet_grade",
                                    values="rate_per_1k")
    print("\n  MECHANISM — Simpson's paradox via road_type:")
    print(grade_road.to_string())
    prem_share = (df[df.helmet_grade == "Premium"]
                  .groupby("road_type")["trips"].sum()
                  / df[df.helmet_grade == "Premium"]["trips"].sum() * 100)
    std_share = (df[df.helmet_grade == "Standard"]
                 .groupby("road_type")["trips"].sum()
                 / df[df.helmet_grade == "Standard"]["trips"].sum() * 100)
    print("\n  Trip distribution by grade and road type (% of each grade's trips):")
    print("  Premium:  ", prem_share.round(1).to_dict())
    print("  Standard: ", std_share.round(1).to_dict())
    print("\n  Within each road type, Premium helmets do NOT have a higher rate.\n"
          "  The aggregate illusion arises because Premium riders disproportionately\n"
          "  ride on Motorways/City arterials — the highest-risk road types —\n"
          "  while Standard riders are concentrated on lower-risk Inner-city lanes.\n"
          "  The helmet grade is a proxy for rider assignment, not helmet quality.")

    # ── Redesign: stacked bar by road type ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    road_order = ["Motorway", "City arterial", "Inner-city lanes"]
    colours = {"Standard": C[0], "Premium": C[1]}

    x = np.arange(len(road_order))
    width = 0.35
    for idx, grade in enumerate(["Standard", "Premium"]):
        vals = [
            grade_road.loc[r, grade] if r in grade_road.index else 0
            for r in road_order
        ]
        axes[0].bar(x + idx * width, vals, width,
                    label=grade, color=colours[grade])
    axes[0].set_xticks(x + width / 2)
    axes[0].set_xticklabels(road_order, rotation=15, ha="right")
    axes[0].set_ylabel("Head injuries per 1,000 trips")
    axes[0].set_title("Within each road type, Premium≈Standard\n"
                       "(grade is a proxy for road assignment)")
    axes[0].legend(title="Helmet grade")
    axes[0].set_ylim(0)
    axes[0].spines["left"].set_visible(False)
    axes[0].spines["bottom"].set_visible(False)

    for idx, grade in enumerate(["Standard", "Premium"]):
        sub = df[df.helmet_grade == grade].groupby("road_type")["trips"].sum()
        sub = sub.reindex(road_order, fill_value=0)
        axes[1].bar(x + idx * width, sub.values / sub.sum() * 100,
                    width, label=grade, color=colours[grade])
    axes[1].set_xticks(x + width / 2)
    axes[1].set_xticklabels(road_order, rotation=15, ha="right")
    axes[1].set_ylabel("% of grade's trips")
    axes[1].set_title("Premium riders are concentrated on\nhigh-risk roads")
    axes[1].legend(title="Helmet grade")
    axes[1].set_ylim(0)
    axes[1].spines["left"].set_visible(False)
    axes[1].spines["bottom"].set_visible(False)

    fig.suptitle("Case 1 redesign — stratify by road type to expose Simpson's paradox",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, "case01_redesign.png")


# =============================================================================
# Case 2 — "Canteen chai sales predict quiz failures"
# =============================================================================
def case02():
    _section(2, "Canteen chai sales predict quiz failures")
    df = pd.read_csv(os.path.join(DATA, "case02_chai_quizzes.csv"))

    # ── Rebuild: scatter cups vs failures, annotate r ──────────────────────
    r, p = stats.pearsonr(df.cups_doodh_patti_sold, df.quiz_failures)
    print(f"  Self-check r ≈ 0.98: r = {r:.2f}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(df.cups_doodh_patti_sold, df.quiz_failures,
               color=C[0], alpha=0.7, s=50)
    m, b = np.polyfit(df.cups_doodh_patti_sold, df.quiz_failures, 1)
    xs = np.linspace(df.cups_doodh_patti_sold.min(),
                     df.cups_doodh_patti_sold.max(), 200)
    ax.plot(xs, m * xs + b, color=C[1], linewidth=1.8)
    ax.text(0.05, 0.92, f"r = {r:.2f}", transform=ax.transAxes,
            fontsize=11, color=PALETTE["ink"])
    ax.set_xlabel("Cups of doodh patti sold per month")
    ax.set_ylabel("Quiz failures per month")
    ax.set_title("Chai sales predict quiz failures  (r = 0.98)")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    _save(fig, "case02_analyst.png")

    # ── Diagnosis ────────────────────────────────────────────────────────────
    print("\n  CONCLUSION: chai causes students to fail quizzes.")
    print("\n  DECOY: 'Correlation is not causation.' Ruled out as a useful\n"
          "    objection — it names no mechanism. The real question is: what drives\n"
          "    BOTH variables upward together?")

    # Third variable: students_enrolled
    r_enroll_chai, _ = stats.pearsonr(df.students_enrolled,
                                       df.cups_doodh_patti_sold)
    r_enroll_fail, _ = stats.pearsonr(df.students_enrolled, df.quiz_failures)
    df["cups_per_student"] = df.cups_doodh_patti_sold / df.students_enrolled
    df["failures_per_student"] = df.quiz_failures / df.students_enrolled
    r_norm, p_norm = stats.pearsonr(df.cups_per_student, df.failures_per_student)
    print(f"\n  MECHANISM — students_enrolled is the common cause:")
    print(f"    enrollment ↔ cups:     r = {r_enroll_chai:.2f}")
    print(f"    enrollment ↔ failures: r = {r_enroll_fail:.2f}")
    print(f"    Per-student cups ↔ per-student failures: r = {r_norm:.2f}  (p = {p_norm:.3f})")
    print("    Once we normalise by enrolment, the relationship essentially\n"
          "    disappears. A growing campus drinks more chai AND fails more quizzes\n"
          "    because it has more students — not because chai causes failure.")

    # ── Redesign: per-student rates ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(df.cups_doodh_patti_sold, df.quiz_failures,
                    c=df.students_enrolled, cmap="Blues", s=55, alpha=0.8,
                    edgecolors="white", linewidths=0.5)
    axes[0].set_xlabel("Cups sold / month")
    axes[0].set_ylabel("Quiz failures / month")
    sc = axes[0].scatter([], [], c=[], cmap="Blues")
    cb = fig.colorbar(
        plt.cm.ScalarMappable(
            cmap="Blues",
            norm=plt.Normalize(df.students_enrolled.min(), df.students_enrolled.max())
        ),
        ax=axes[0], shrink=0.75, label="Students enrolled",
    )
    axes[0].set_title(f"Apparent r = {r:.2f} — colour reveals\nthe common cause: enrolment")
    axes[0].spines["left"].set_visible(False)
    axes[0].spines["bottom"].set_visible(False)

    axes[1].scatter(df.cups_per_student * 1000, df.failures_per_student * 1000,
                    color=C[0], alpha=0.7, s=55)
    m2, b2 = np.polyfit(df.cups_per_student, df.failures_per_student, 1)
    xs2 = np.linspace(df.cups_per_student.min(), df.cups_per_student.max(), 200)
    axes[1].plot(xs2 * 1000, (m2 * xs2 + b2) * 1000, color=C[1])
    axes[1].set_xlabel("Cups per 1,000 students / month")
    axes[1].set_ylabel("Failures per 1,000 students / month")
    axes[1].text(0.05, 0.92, f"r = {r_norm:.2f}  p = {p_norm:.2f}",
                 transform=axes[1].transAxes, fontsize=10)
    axes[1].set_title(f"Per-student: r drops to {r_norm:.2f}\n— no meaningful relationship")
    axes[1].spines["left"].set_visible(False)
    axes[1].spines["bottom"].set_visible(False)

    fig.suptitle("Case 2 redesign — normalise by enrolment; the correlation vanishes",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, "case02_redesign.png")


# =============================================================================
# Case 3 — "The Smog Action Plan cut Lahore's AQI by a third"
# =============================================================================
def case03():
    _section(3, "Smog Action Plan cut Lahore's AQI by a third")
    df = pd.read_csv(os.path.join(DATA, "case03_aqi_monitors.csv"),
                     parse_dates=["date"])
    df["day_n"] = (df.date - df.date.min()).dt.days + 1

    # Mean AQI per date (pandas skips NaN by default)
    daily = df.groupby("day_n")["aqi"].mean()
    before = daily[daily.index <= 60].mean()
    after = daily[daily.index > 60].mean()
    print(f"  Self-check (before ~202, after ~137):")
    print(f"    before day 61: {before:.0f}   after day 61: {after:.0f}"
          f"   change: {(after-before)/before*100:.0f}%")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(daily.index, daily.values, color=C[0], linewidth=1.2)
    ax.axvline(61, color=PALETTE["highlight"], linewidth=1.5,
               linestyle="--", label="Smog Action Plan")
    ax.axhline(before, xmin=0, xmax=60/120, color=C[1], linewidth=1.5,
               linestyle="-", label=f"Mean before: {before:.0f}")
    ax.axhline(after, xmin=60/120, xmax=1, color=C[2], linewidth=1.5,
               linestyle="-", label=f"Mean after: {after:.0f}")
    ax.set_ylim(0)
    ax.set_xlabel("Day")
    ax.set_ylabel("AQI")
    ax.set_title(f"Smog Action Plan cut mean AQI by {(before-after)/before*100:.0f}%")
    ax.legend(fontsize=8)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    _save(fig, "case03_analyst.png")

    # ── Diagnosis ────────────────────────────────────────────────────────────
    print("\n  CONCLUSION: policy intervention improved air quality by 32%.")
    print("\n  DECOY: 'Seasonal variation' — plausible, but the y-axis starts\n"
          "    at zero and the recipe says to draw it from zero, so it is\n"
          "    not a truncated-axis problem.")

    # Mechanism: station_status / missing data pattern
    n_reporting = df.groupby("day_n").apply(
        lambda g: (g.station_status == "reporting").sum()
    )
    n_total = df.groupby("day_n").size()
    reporting_rate = n_reporting / n_total

    before_rr = reporting_rate[reporting_rate.index <= 60].mean()
    after_rr = reporting_rate[reporting_rate.index > 60].mean()

    # Which zones lose stations?
    zone_before = (df[df.day_n <= 60]
                   .groupby("zone")["aqi"].mean()
                   .sort_values(ascending=False))
    zone_after = (df[df.day_n > 60]
                  .groupby("zone")["aqi"].mean()
                  .sort_values(ascending=False))
    report_before = (df[(df.day_n <= 60) & (df.station_status == "reporting")]
                     .groupby("zone").size())
    report_after = (df[(df.day_n > 60) & (df.station_status == "reporting")]
                    .groupby("zone").size())

    print(f"\n  MECHANISM — missing data in a pattern:")
    print(f"    Average % stations reporting — before: {before_rr*100:.0f}%   "
          f"after: {after_rr*100:.0f}%")
    print("\n  Mean AQI by zone, before vs after:")
    compare = pd.DataFrame({"before": zone_before, "after": zone_after,
                             "reporting_before": report_before,
                             "reporting_after": report_after})
    print(compare.to_string())
    print("\n    Industrial stations (highest AQI) drop out of the reporting\n"
          "    network after day 61. pandas.mean() silently skips NaN rows —\n"
          "    so the mean falls not because air quality improved but because\n"
          "    the dirtiest stations stopped transmitting. The 32% improvement\n"
          "    is an artefact of which stations are online, not of the policy.")

    # ── Redesign: reporting station count + zone-stratified AQI ─────────────
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(reporting_rate.index, reporting_rate.values * 100,
                 color=C[0], linewidth=1.2)
    axes[0].axvline(61, color=PALETTE["highlight"], linewidth=1.5, linestyle="--")
    axes[0].set_ylabel("% stations reporting")
    axes[0].set_title("Reporting rate drops after day 61 — especially Industrial stations")
    axes[0].set_ylim(0, 110)
    axes[0].spines["left"].set_visible(False)
    axes[0].spines["bottom"].set_visible(False)

    zone_colors = {"Industrial": C[1], "Arterial": C[0],
                   "Residential": C[2], "Peri-urban": C4}
    for zone, colour in zone_colors.items():
        zone_daily = df[df.zone == zone].groupby("day_n")["aqi"].mean()
        axes[1].plot(zone_daily.index, zone_daily.values,
                     label=zone, color=colour, linewidth=1.2, alpha=0.9)
    axes[1].axvline(61, color=PALETTE["highlight"], linewidth=1.5,
                    linestyle="--", label="Day 61 (policy)")
    axes[1].set_ylim(0)
    axes[1].set_xlabel("Day")
    axes[1].set_ylabel("Mean AQI")
    axes[1].set_title("Zone-stratified AQI — Industrial is highest and falls silent after day 61")
    axes[1].legend(fontsize=8)
    axes[1].spines["left"].set_visible(False)
    axes[1].spines["bottom"].set_visible(False)

    fig.suptitle("Case 3 redesign — show what went offline, not just the falling mean",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, "case03_redesign.png")


# =============================================================================
# Case 4 — "Four departments behave identically, so fund them equally"
# =============================================================================
def case04():
    _section(4, "Four departments behave identically — Anscombe's quartet")
    df = pd.read_csv(os.path.join(DATA, "case04_quartet_depts.csv"))

    from scipy.stats import pearsonr as pr

    # ── Rebuild: statistics table ────────────────────────────────────────────
    rows = []
    for dept, g in df.groupby("department"):
        r_val, _ = pr(g.study_hours_per_week, g.final_score_out_of_20)
        m, b = np.polyfit(g.study_hours_per_week, g.final_score_out_of_20, 1)
        rows.append({
            "department": dept,
            "n": len(g),
            "mean_hours": g.study_hours_per_week.mean(),
            "mean_score": g.final_score_out_of_20.mean(),
            "sd_hours": g.study_hours_per_week.std(ddof=0),
            "sd_score": g.final_score_out_of_20.std(ddof=0),
            "pearson_r": r_val,
            "slope": m,
        })
    tbl = pd.DataFrame(rows).set_index("department")
    print("  Self-check (mean hours ~18.00, mean score ~14.50, r ~0.816):")
    print(tbl.round(3).to_string())

    # ── Diagnosis ────────────────────────────────────────────────────────────
    print("\n  CONCLUSION: departments are statistically identical; equalise funding.")
    print("\n  DECOY: 'Small sample size (n=11 per dept).' Ruled out — the recipe\n"
          "    specifies 11 per department, and that is the entire dataset. The\n"
          "    problem is not insufficient data; it is the wrong summary statistic.")
    print("\n  MECHANISM — summary statistics do not constrain shape (Anscombe 1973):\n"
          "    All four departments produce identical mean, SD and r, yet their\n"
          "    scatter plots are completely different: one is linear, one has a\n"
          "    quadratic relationship, one has an outlier driving the regression,\n"
          "    and one has a vertical cluster with a single outlier. Equal statistics\n"
          "    is not evidence of equal behaviour — it is evidence of poor charting.")

    # ── Redesign: four scatter plots ─────────────────────────────────────────
    depts = df.department.unique()
    n_d = len(depts)
    fig, axes = plt.subplots(1, n_d, figsize=(4 * n_d, 4), sharey=True, sharex=False)
    for ax_i, dept in zip(axes, sorted(depts)):
        g = df[df.department == dept]
        r_val, _ = pr(g.study_hours_per_week, g.final_score_out_of_20)
        m, b = np.polyfit(g.study_hours_per_week, g.final_score_out_of_20, 1)
        xs = np.linspace(g.study_hours_per_week.min(), g.study_hours_per_week.max(), 100)
        ax_i.scatter(g.study_hours_per_week, g.final_score_out_of_20,
                     color=C[0], s=55, zorder=3)
        ax_i.plot(xs, m * xs + b, color=C[1], linewidth=1.5)
        ax_i.set_title(f"{dept}\nr = {r_val:.3f}", fontsize=9)
        ax_i.set_xlabel("Study hours / week")
        ax_i.spines["left"].set_visible(False)
        ax_i.spines["bottom"].set_visible(False)
    axes[0].set_ylabel("Final score / 20")
    fig.suptitle("Case 4 redesign — identical statistics, wildly different shapes\n"
                 "(all have mean 18.00 / 14.50, r = 0.816)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, "case04_redesign.png")


# =============================================================================
# Case 5 — "Five cohorts, one set of statistics"
# =============================================================================
def case05():
    _section(5, "Five cohorts, one set of statistics")
    df = pd.read_csv(os.path.join(DATA, "case05_same_stats_shapes.csv"))

    from scipy.stats import pearsonr as pr

    # ── Rebuild: statistics table ────────────────────────────────────────────
    rows = []
    for shape, g in df.groupby("shape_name"):
        r_val, _ = pr(g.hours_on_platform, g.assessment_score)
        rows.append({
            "cohort": shape,
            "n": len(g),
            "mean_hours": g.hours_on_platform.mean(),
            "mean_score": g.assessment_score.mean(),
            "sd_hours": g.hours_on_platform.std(ddof=0),
            "sd_score": g.assessment_score.std(ddof=0),
            "pearson_r": r_val,
        })
    tbl = pd.DataFrame(rows).set_index("cohort")
    print("  Self-check (all cohorts: 48.00 / 63.00, SD 12.00 / 15.00, r 0.420):")
    print(tbl.round(3).to_string())

    # ── Diagnosis ────────────────────────────────────────────────────────────
    print("\n  CONCLUSION: cohorts are interchangeable; treat them identically.")
    print("\n  DECOY: 'The r = 0.42 is significant.' True, but significance says\n"
          "    nothing about shape. A p-value on a correlation is not a shape test.")
    print("\n  MECHANISM — match_stats() keeps shape while forcing statistics:\n"
          "    All five cohorts produce identical n, mean, SD and r. Yet their\n"
          "    scatter plots form completely different geometric shapes (straight\n"
          "    line, arc, cluster, etc.). The summary statistics carry no information\n"
          "    about the distribution of the data — only a plot does.")

    # ── Redesign: scatter for each cohort ────────────────────────────────────
    shapes = sorted(df.shape_name.unique())
    n_s = len(shapes)
    cols_per_row = 3
    n_rows = int(np.ceil(n_s / cols_per_row))
    fig, axes = plt.subplots(n_rows, cols_per_row,
                             figsize=(4.5 * cols_per_row, 4.5 * n_rows))
    axes_flat = axes.flat
    for ax_i, shape in zip(axes_flat, shapes):
        g = df[df.shape_name == shape]
        r_val, _ = pr(g.hours_on_platform, g.assessment_score)
        ax_i.scatter(g.hours_on_platform, g.assessment_score,
                     color=C[0], s=20, alpha=0.7)
        ax_i.set_title(f"'{shape}'\nr = {r_val:.3f}", fontsize=9)
        ax_i.set_xlabel("Hours on platform")
        ax_i.set_ylabel("Assessment score")
        ax_i.spines["left"].set_visible(False)
        ax_i.spines["bottom"].set_visible(False)
    for ax_i in list(axes_flat)[n_s:]:
        ax_i.set_visible(False)
    fig.suptitle("Case 5 redesign — same statistics, radically different shapes",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, "case05_redesign.png")


# =============================================================================
# Case 6 — "Creatinine tracks HbA1c — a finding"
# =============================================================================
def case06():
    _section(6, "Creatinine tracks HbA1c — a finding")
    df = pd.read_csv(os.path.join(DATA, "case06_sensor_matrix.csv"))

    # ── Rebuild: correlation matrix, cmap="Reds", no vmin/vmax ──────────────
    excl = ["age_band", "study_site", "hr_device_model"]
    num_cols = [c for c in df.select_dtypes("number").columns if c not in excl]
    corr = df[num_cols].corr()
    print(f"  Self-check: matrix is {corr.shape[0]}×{corr.shape[1]}")
    print(f"  creatinine ↔ hba1c = {corr.loc['creatinine','hba1c']:.2f}")
    print(f"  systolic ↔ map_mmhg = {corr.loc['systolic','map_mmhg']:.2f}")

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(corr, ax=ax, cmap="Reds", annot=True, fmt=".2f",
                annot_kws={"size": 6}, linewidths=0.3)
    ax.set_title("Creatinine tracks HbA1c  (r = 0.77)")
    fig.tight_layout()
    _save(fig, "case06_analyst.png")

    # ── Diagnosis ────────────────────────────────────────────────────────────
    print("\n  CONCLUSION: creatinine and HbA1c are clinically associated (r = 0.77).")
    print("\n  DECOY: 'The sample is too small.' At n = 180 Pearson r = 0.77 is\n"
          "    significant at any conventional level. Sample size does not save\n"
          "    the chart.")
    print("\n  MECHANISM — two separate problems:")
    print("    1. DRAWING: cmap='Reds' with no vmin/vmax maps the OBSERVED range\n"
          "       of correlations (~0.0 to ~0.96) to the full red colour scale.\n"
          "       A correlation of 0.30 appears almost as red as one of 0.96,\n"
          "       making weak relationships visually indistinguishable from strong\n"
          "       ones. The correct map is RdBu_r with vmin=-1, vmax=1 so that the\n"
          "       midpoint (zero) is always white and both scale ends are anchored.")
    map_val = corr.loc["systolic", "map_mmhg"]
    print(f"\n    2. ANALYTICAL: three columns inflate the matrix:\n"
          f"       • map_mmhg is computed as (systolic + 2×diastolic)/3.\n"
          f"         Its correlation with systolic is {map_val:.2f} — pure arithmetic.\n"
          f"         Including it makes the heatmap look richer than it is.\n"
          f"       • probe_a … probe_f are unassigned instrument channels, not\n"
          f"         calibrated measurements. Their correlations are noise.")
    print("\n    Fixing the drawing (RdBu_r + vmin/vmax) does NOT fix the analysis.\n"
          "    The creatinine↔HbA1c association may be real, but it cannot be read\n"
          "    as a finding while arithmetic duplicates and noise columns sit in the\n"
          "    same matrix and receive the same visual weight.")

    # ── Redesign: proper heatmap ──────────────────────────────────────────────
    excl2 = excl + ["map_mmhg"] + [c for c in df.columns if c.startswith("probe_")]
    good_cols = [c for c in df.select_dtypes("number").columns if c not in excl2]
    corr2 = df[good_cols].corr()
    mask = np.triu(np.ones_like(corr2, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(
        corr2, ax=ax,
        mask=mask,
        cmap="RdBu_r", vmin=-1, vmax=1, center=0,
        annot=True, fmt=".2f",
        annot_kws={"size": 8},
        linewidths=0.4,
        cbar_kws={"label": "Pearson r"},
    )
    ax.set_title("Case 6 redesign — probe columns and map_mmhg removed;\n"
                 "RdBu_r with vmin=−1, vmax=1; upper triangle masked")
    fig.tight_layout()
    _save(fig, "case06_redesign.png")


# =============================================================================
# Case 7 — "ShaheenGo ran at a loss until May"
# =============================================================================
def case07():
    _section(7, "ShaheenGo ran at a loss until May")
    df = pd.read_csv(os.path.join(DATA, "case07_dual_axis_series.csv"))

    # ── Rebuild: dual y-axis, revenue 50-80 left, cost 40-56 right ──────────
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()
    x = np.arange(len(df))
    ax1.plot(x, df.revenue_pkr_m, color=C[0], marker="o", label="Revenue (left)")
    ax2.plot(x, df.cost_pkr_m, color=C[1], marker="s", label="Cost (right)")
    ax1.set_ylim(50, 80)
    ax2.set_ylim(40, 56)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df.month)
    ax1.set_ylabel("Revenue (PKR m)", color=C[0])
    ax2.set_ylabel("Cost (PKR m)", color=C[1])
    ax1.set_title("ShaheenGo ran at a loss until May")
    lines1, lbls1 = ax1.get_legend_handles_labels()
    lines2, lbls2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lbls1 + lbls2, fontsize=8)
    print("  Self-check: cost line sits ABOVE revenue line for first 4 months.")
    _save(fig, "case07_analyst.png")

    # ── Diagnosis ────────────────────────────────────────────────────────────
    profit = df.revenue_pkr_m - df.cost_pkr_m
    print("\n  CONCLUSION: company was unprofitable in the first 4 months.")
    print("\n  DECOY: 'The y-axis is truncated.' The analyst would say:\n"
          "    both axes were chosen to fit the data — nothing is hidden.\n"
          "    The correct objection is different.")
    print(f"\n  MECHANISM — dual y-axes manufacture a visual relationship:\n"
          f"    Profit every month (revenue − cost):\n"
          f"    {dict(zip(df.month, profit.values))}\n"
          f"    Revenue exceeds cost in EVERY month. The 'loss' is an artefact\n"
          f"    of the axis ranges: revenue axis spans 50–80 (range 30 PKR m),\n"
          f"    cost axis spans 40–56 (range 16 PKR m). Compressing the cost range\n"
          f"    raises the cost line visually above the revenue line even though\n"
          f"    cost < revenue always. There is no single Lie Factor to compute\n"
          f"    because the visual impression is created by the RATIO of the two\n"
          f"    scales, not by any one bar or line length.")

    # ── Redesign: single axis from zero ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(df))
    ax.plot(x, df.revenue_pkr_m, color=C[0], marker="o",
            label="Revenue", linewidth=2)
    ax.plot(x, df.cost_pkr_m, color=C[1], marker="s",
            label="Cost", linewidth=2)
    ax.fill_between(x, df.cost_pkr_m, df.revenue_pkr_m,
                    alpha=0.15, color=C[2], label="Profit margin")
    ax.set_ylim(0)
    ax.set_xticks(x)
    ax.set_xticklabels(df.month)
    ax.set_ylabel("PKR millions")
    ax.set_title("ShaheenGo was profitable every month — revenue always exceeds cost")
    ax.legend(fontsize=9)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    fig.suptitle("Case 7 redesign — single axis from zero; profit is visible",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    _save(fig, "case07_redesign.png")


# =============================================================================
# Case 8 — "Three bin widths, three different distributions"
# =============================================================================
def case08():
    _section(8, "Three bin widths, three different distributions")
    df = pd.read_csv(os.path.join(DATA, "vitals_10k.csv"))

    # ── Rebuild: three histograms, bins = 3, 45, 400 ────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for ax_i, bins in zip(axes, [3, 45, 400]):
        ax_i.hist(df.systolic, bins=bins, color=C[0], edgecolor="white",
                  linewidth=0.3)
        ax_i.set_title(f"bins = {bins}")
        ax_i.set_xlabel("Systolic BP (mmHg)")
        ax_i.spines["left"].set_visible(False)
        ax_i.spines["bottom"].set_visible(False)
        n = df.systolic.dropna().shape[0]
        ax_i.text(0.97, 0.95, f"n = {n:,}", ha="right", va="top",
                  transform=ax_i.transAxes, fontsize=9)
    axes[0].set_ylabel("Count")
    print(f"  Self-check: n = {df.systolic.dropna().shape[0]:,} in all three panels.")
    fig.suptitle("Three bin widths, three different 'distributions'",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, "case08_analyst.png")

    # ── Diagnosis ────────────────────────────────────────────────────────────
    print("\n  CONCLUSION: choice of bin count decides what the distribution 'is'.")
    print("\n  DECOY: 'bins=3 is obviously wrong.' The recipe shows all three as\n"
          "    equally presented results, not as a sensitivity check. The chart\n"
          "    implies the analyst had a reason for each and picked one —\n"
          "    but that reason is never stated.")
    q75, q25 = np.percentile(df.systolic.dropna(), [75, 25])
    iqr = q75 - q25
    n = df.systolic.dropna().shape[0]
    fd_width = 2 * iqr / (n ** (1 / 3))
    fd_bins = int(np.ceil((df.systolic.max() - df.systolic.min()) / fd_width))
    print(f"\n  MECHANISM — the bin count is a free parameter that shapes the picture:\n"
          f"    bins=3 erases all structure, showing one broad hump.\n"
          f"    bins=45 reveals the bimodal structure of the data.\n"
          f"    bins=400 introduces apparent gaps and spikes from noise — artefacts\n"
          f"    of the count-per-cell granularity, not of the physiology.\n"
          f"\n  The rule: declare the bin width and the rule that produced it.\n"
          f"    Freedman-Diaconis gives bin width ≈ {fd_width:.2f} mmHg → {fd_bins} bins.\n"
          f"    A stated rule is a decision; an unstated one is a choice that shapes\n"
          f"    the reader's inference without their knowledge.")

    # ── Redesign: one principled histogram ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df.systolic.dropna(), bins=fd_bins, color=C[0],
            edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Systolic BP (mmHg)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Systolic BP is bimodal — two populations at ~120 and ~145 mmHg\n"
        f"[Freedman-Diaconis rule: bin width ≈ {fd_width:.1f} mmHg, bins = {fd_bins}]"
    )
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    fig.suptitle("Case 8 redesign — one bin count, declared and justified",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    _save(fig, "case08_redesign.png")


# =============================================================================
# Case 9 — "4.4% of the density sits above 100% saturation"
# =============================================================================
def case09():
    _section(9, "4.4% of density sits above 100% saturation")
    df = pd.read_csv(os.path.join(DATA, "vitals_10k.csv"))

    print(f"  Self-check: max spo2 = {df.spo2.max()} (should be 100)")

    # ── Rebuild: KDE with default bandwidth, x-axis to 108 ──────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(df.spo2.dropna())
    xs = np.linspace(df.spo2.min() - 2, 108, 500)
    ys = kde(xs)
    area_above_100 = np.trapz(ys[xs >= 100], xs[xs >= 100])
    total_area = np.trapz(ys, xs)
    pct_above = area_above_100 / total_area * 100
    print(f"  Area above 100% of KDE curve: {pct_above:.1f}%")

    ax.plot(xs, ys, color=C[0], linewidth=2)
    ax.fill_between(xs, ys, where=(xs >= 100), color=C[1], alpha=0.4,
                    label=f"{pct_above:.1f}% above physical maximum")
    ax.axvline(100, color=PALETTE["highlight"], linewidth=1.5,
               linestyle="--", label="Physical ceiling (100%)")
    ax.set_xlabel("SpO₂ (%)")
    ax.set_ylabel("Density")
    ax.set_title(f"{pct_above:.1f}% of the density sits above 100% saturation")
    ax.legend(fontsize=9)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    _save(fig, "case09_analyst.png")

    # ── Diagnosis ────────────────────────────────────────────────────────────
    print("\n  CONCLUSION: chart claims density extends beyond the physical maximum.")
    print("\n  DECOY: 'The data is noisy / has measurement error.' SpO₂ is a\n"
          "    bounded variable with a hard physical ceiling of 100%. The maximum\n"
          "    in the dataset IS 100. No measurement error produces an above-100\n"
          "    reading here — the issue is in the estimator, not the instrument.")
    bw = kde.factor
    print(f"\n  MECHANISM — default KDE bandwidth ({bw:.3f}) does not know about\n"
          f"    domain bounds:\n"
          f"    A Gaussian kernel centred at 100 (the maximum possible value) places\n"
          f"    half its mass above 100. With n = {df.spo2.dropna().shape[0]:,} points\n"
          f"    at or near the ceiling, the aggregate density leaks above 100 by\n"
          f"    construction — it is a mathematical artefact of using an unbounded\n"
          f"    kernel on a bounded variable.\n"
          f"\n    The fix is not to 'extend the x-axis to 108' (which the recipe\n"
          f"    asked for, making the artefact visible). It is to use a boundary-\n"
          f"    corrected estimator or to note explicitly that the curve below 100\n"
          f"    has been renormalised.")

    # ── Redesign: reflection method + annotation ─────────────────────────────
    # Reflection at the boundary: mirror the data about x=100, estimate KDE on
    # the combined set, double the density below 100, zero it above.
    data = df.spo2.dropna().values
    reflected = np.concatenate([data, 2 * 100 - data])
    kde_refl = gaussian_kde(reflected)
    xs_r = np.linspace(data.min() - 2, 102, 500)
    ys_r = np.where(xs_r <= 100, 2 * kde_refl(xs_r), 0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs_r, ys_r, color=C[0], linewidth=2,
            label="Boundary-corrected KDE")
    ax.plot(xs, ys, color=PALETTE["muted"], linewidth=1.5,
            linestyle="--", label="Default KDE (leaks above 100)")
    ax.axvline(100, color=PALETTE["highlight"], linewidth=1.5,
               linestyle="--", label="Physical ceiling (100%)")
    ax.fill_between(xs, ys, where=(xs >= 100), color=C[1], alpha=0.2)
    ax.set_xlim(df.spo2.min() - 1, 103)
    ax.set_xlabel("SpO₂ (%)")
    ax.set_ylabel("Density")
    ax.set_title("Case 9 redesign — reflection method eliminates phantom density\n"
                 "above the physical ceiling (100%)")
    ax.legend(fontsize=9)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    fig.suptitle("Default KDE ignores domain bounds; reflection method is the fix",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    _save(fig, "case09_redesign.png")


# =============================================================================
if __name__ == "__main__":
    print("Part A — rebuilding and diagnosing nine wrong charts")
    print("Output PNGs will be written to:", ROOT)

    case01()
    case02()
    case03()
    case04()
    case05()
    case06()
    case07()
    case08()
    case09()

    print("\n\nAll nine cases complete.")
    print("Analyst charts: case01_analyst.png … case09_analyst.png")
    print("Redesigns:      case01_redesign.png … case09_redesign.png")
