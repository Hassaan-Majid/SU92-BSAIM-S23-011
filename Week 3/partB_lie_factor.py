"""
Part B — Lie Factor, computed in Python  (15 marks)

Reads case07_lie_factor.csv and audits five published graphics with
vizlib.lie_factor(), then answers the four questions in the brief.

Run:
    python3 partB_lie_factor.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vizlib import lie_factor, lie_verdict   # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "case07_lie_factor.csv")
DUAL = os.path.join(ROOT, "data", "case07_dual_axis_series.csv")


def main():
    df = pd.read_csv(DATA)

    # ── Question 1: table of lie factors ────────────────────────────────────
    print("=" * 76)
    print("QUESTION 1 — Lie Factor table")
    print("=" * 76)
    fmt = "{:3}  {:<32}  {:>9}  {:>9}  {:>7}  {}"
    print(fmt.format("ID", "Subject", "Data Δ", "Ink Δ", "LF", "Verdict"))
    print("-" * 76)
    results = {}
    for _, row in df.iterrows():
        lf, de, ge = lie_factor(
            row.value_start, row.value_end,
            row.ink_start, row.ink_end,
            ink_dimension=int(row.ink_dimension),
        )
        v = lie_verdict(lf)
        results[row.chart_id] = (lf, de, ge)
        print(fmt.format(
            row.chart_id, row.subject[:32],
            f"{de:+.4f}", f"{ge:+.4f}", f"{lf:.3f}", v,
        ))
    print()

    # ── Question 2: which one is honest? ────────────────────────────────────
    print("=" * 76)
    print("QUESTION 2 — Which graphic is honest?")
    print("=" * 76)
    print(
        "Graphic C (Export earnings) is honest — LF ≈ 1.00.\n"
        "Bars start at zero and the ink grows proportionally to the data:\n"
        "  25 → 30 is a +20.0 % data rise; 5.0 cm → 6.0 cm is a +20.0 % ink rise.\n"
        "\n"
        "The other four encode the same data dishonestly:\n"
        "\n"
        "  A (Corporate tax rate)\n"
        "    Bars drawn from a 29.6 % baseline, not zero. The eye sees only the\n"
        "    small tip above the cut, so a 6.7 % data rise reads as a ~500 %\n"
        "    visual jump. The ink encodes distance above the cut axis, not the\n"
        "    absolute value — which is the definition of a truncated-axis lie.\n"
        "\n"
        "  B (Federal health budget)\n"
        "    A 2-D coin icon scaled 1.5× in BOTH dimensions encodes a 50 % rise\n"
        "    with 125 % more area (1.5² = 2.25×). Area drives perception, so the\n"
        "    chart makes a 50 % rise look like 125 %.\n"
        "\n"
        "  D (Air-pollution index)\n"
        "    Bars start at zero and nothing is obviously wrong, yet the ink\n"
        "    barely moves (8 → 9 cm, +12.5 %) while the data rose 50 %.\n"
        "    The chart understates a worsening pollution episode — a subtle\n"
        "    distortion that benefits whoever wants the trend to look flat.\n"
        "\n"
        "  E (City population)\n"
        "    A bubble sized by radius: value 2 → 8 (×4) gives radius ×4, but\n"
        "    the perceived quantity is AREA, which grows as the square: ×16.\n"
        "    A quadrupling of population looks like a 16× explosion.\n"
    )

    # ── Question 3: ink_dimension=2 ─────────────────────────────────────────
    print("=" * 76)
    print("QUESTION 3 — Which graphics need ink_dimension=2?")
    print("=" * 76)
    lf_b_wrong, de_b, ge_b_wrong = lie_factor(
        df.loc[df.chart_id == "B", "value_start"].iloc[0],
        df.loc[df.chart_id == "B", "value_end"].iloc[0],
        df.loc[df.chart_id == "B", "ink_start"].iloc[0],
        df.loc[df.chart_id == "B", "ink_end"].iloc[0],
        ink_dimension=1,
    )
    lf_e_wrong, de_e, ge_e_wrong = lie_factor(
        df.loc[df.chart_id == "E", "value_start"].iloc[0],
        df.loc[df.chart_id == "E", "value_end"].iloc[0],
        df.loc[df.chart_id == "E", "ink_start"].iloc[0],
        df.loc[df.chart_id == "E", "ink_end"].iloc[0],
        ink_dimension=1,
    )
    print(
        "Graphics B and E encode their quantity as an AREA, not a length.\n"
        "\n"
        f"  B (coin icon): with ink_dimension=1 → LF = {lf_b_wrong:.2f} (appears honest).\n"
        f"                 with ink_dimension=2 → LF = {results['B'][0]:.2f} (exposes the distortion).\n"
        "    The icon is scaled 1.5× in both height and width; its area grows as\n"
        "    1.5² = 2.25×. Treating ink as a length misses the doubling of perceived\n"
        "    quantity and would report the chart as honest when it is not.\n"
        "\n"
        f"  E (bubble):    with ink_dimension=1 → LF = {lf_e_wrong:.2f} (misreads the encoding).\n"
        f"                 with ink_dimension=2 → LF = {results['E'][0]:.2f} (captures the explosion).\n"
        "    The bubble's RADIUS was set proportional to the value, so area grows as\n"
        "    the square. The eye reads area — leaving ink_dimension=1 makes a 16×\n"
        "    perceived increase look like a 4× one in the audit.\n"
    )
    print()

    # ── Question 4: LF below 1 ──────────────────────────────────────────────
    print("=" * 76)
    print("QUESTION 4 — The understated chart (LF < 1)")
    print("=" * 76)
    lf_d, _, _ = results["D"]
    print(
        f"Graphic D (Air-pollution index) has LF = {lf_d:.3f} < 1.\n"
        "\n"
        "The data rose 50 % (100 → 150) but the ink rose only ~12.5 %\n"
        "(8 cm → 9 cm), so the chart shows less change than actually occurred.\n"
        "\n"
        "An understated chart is still a distortion: the reader forms an impression\n"
        "of stasis or slow change when the reality was acute. The direction of the\n"
        "distortion is strategic — a regulator, an emitter, or a government agency\n"
        "facing scrutiny over worsening air quality benefits from a chart that makes\n"
        "the spike look modest. The distortion is harder to challenge than a\n"
        "truncated axis because 'bars start at zero' disarms the most common\n"
        "criticism. Tufte's 0.95–1.05 acceptance band is symmetric precisely\n"
        "because both directions mislead.\n"
    )

    # ── Question 5: dual-axis Lie Factor ────────────────────────────────────
    print("=" * 76)
    print("QUESTION 5 — Dual-axis Lie Factor, and what the data actually shows")
    print("=" * 76)
    dual = pd.read_csv(DUAL)
    print(dual.to_string(index=False))
    print()
    profit = dual.revenue_pkr_m - dual.cost_pkr_m
    print(f"Profit every month (revenue − cost):\n{profit.values}\n")
    print(
        "The Lie Factor formula requires a single quantitative encoding: you measure\n"
        "two values on ONE scale and compare the ink ratio to the data ratio.\n"
        "A dual-axis chart has TWO independent scales, so there is no single ink\n"
        "measurement and no single data effect. The Lie Factor CANNOT be computed\n"
        "for a dual-axis chart — the visual impression is created by the RATIO of\n"
        "the two scale ranges, not by any one bar or line length.\n"
        "\n"
        "What the data actually shows:\n"
        "Revenue runs 52–78 PKR m; costs run 44–53 PKR m. Revenue exceeds cost in\n"
        "every single month — the company was profitable throughout. The chart\n"
        "manufactured a 'loss' by compressing the cost axis (40–56, range 16) relative\n"
        "to the revenue axis (50–80, range 30): starting the cost axis 10 below the\n"
        "revenue baseline lifted the cost line visually above the revenue line for the\n"
        "first four months, even though cost < revenue in every month. This is the\n"
        "mechanism dual-axis charts exist to exploit — no single number can be wrong\n"
        "because there is no shared reference.\n"
    )


if __name__ == "__main__":
    main()
