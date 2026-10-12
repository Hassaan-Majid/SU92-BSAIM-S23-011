"""
Part C — Prove a palette is colour-blind safe  (9 marks)

Run:
    python3 partC_palette.py

Produces partC_palette.png — the before/after swatch sheet for both palettes —
and prints the quantitative audit with delta-E distances.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vizlib import (PALETTE, delta_e, save_fig, simulate_cvd,   # noqa: E402
                    style_defaults, verdict, worst_pair)

ROOT = os.path.dirname(os.path.abspath(__file__))
style_defaults()

# ── Question 1: a palette that looks nice but is NOT known to be safe ───────
# Excel's default "Office" theme colours — bright, common in dashboards, and
# frequently misread by people with colour-vision deficiency.
MY_PALETTE = ["#4472C4", "#ED7D31", "#A9D18E", "#FF0000"]
MY_LABEL = "Excel Office theme (blue, orange, green, red)"

# ── Question 2: course palette — three series slots plus the fourth ──────────
COURSE_PALETTE = PALETTE["series"] + [PALETTE["series_4th"]]
KINDS = ["normal", "deutan", "protan", "tritan"]


def swatch_sheet(palettes, path):
    """Render before/after swatches for every palette × CVD type."""
    n_rows = len(palettes)
    n_cols = len(KINDS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.2 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for row, (pal, label) in enumerate(palettes):
        for col, kind in enumerate(KINDS):
            ax = axes[row, col]
            shown = pal if kind == "normal" else simulate_cvd(pal, kind)
            for i, hexc in enumerate(shown):
                ax.add_patch(plt.Rectangle((i, 0), 0.92, 1, color=hexc))
                ax.text(i + 0.46, -0.25, f"S{i+1}", ha="center", fontsize=8,
                        color=PALETTE["ink_soft"])
            ax.set_xlim(-0.1, len(pal))
            ax.set_ylim(-0.5, 1.15)
            ax.axis("off")
            title_col = PALETTE["ink"] if kind == "normal" else PALETTE["highlight"]
            ax.set_title(kind, fontsize=10, color=title_col, loc="center")
        axes[row, 0].text(
            -0.05, 1.55, label,
            transform=axes[row, 0].transAxes,
            fontsize=11, fontweight="bold", color=PALETTE["ink"],
        )

    fig.suptitle("Part C — Prove the palette, do not claim it",
                 x=0.01, ha="left", fontsize=13, fontweight="bold",
                 color=PALETTE["ink"], y=1.02)
    fig.tight_layout()
    out = save_fig(
        fig, path,
        "Machado, Oliveira & Fernandes (2009) severity-1.0 matrices, "
        "applied in linear RGB · synthetic course data CDB601220",
    )
    plt.close(fig)
    return out


def audit(pal, label):
    """Print every CVD's worst pair with its delta-E, then return rows."""
    print(f"\n{'─'*60}")
    print(label)
    print(f"{'─'*60}")
    rows = []
    for kind in ["normal", "deutan", "protan", "tritan"]:
        i, j, a, b, d = worst_pair(pal, kind)
        rows.append((kind, i, j, a, b, d))
        print(f"  {kind:8s}  worst pair  S{i} {a} vs S{j} {b}"
              f"   ΔE {d:5.1f}   {verdict(d)}")
    return rows


def main():
    # ── Save the swatch sheet ────────────────────────────────────────────────
    out = swatch_sheet(
        [(MY_PALETTE, MY_LABEL),
         (COURSE_PALETTE, "Course palette — 3 slots + 4th")],
        os.path.join(ROOT, "partC_palette.png"),
    )
    print("wrote", out)

    # ── Quantitative audit ───────────────────────────────────────────────────
    my_rows = audit(MY_PALETTE, MY_LABEL)
    course_rows = audit(COURSE_PALETTE, "Course palette (3 slots + 4th)")

    # ── Question 3: name the worst pair in each palette ─────────────────────
    print("\n" + "=" * 60)
    print("QUESTION 3 — Worst pair per palette")
    print("=" * 60)

    # Find the worst pair across all CVD types for MY_PALETTE
    my_worst = min(my_rows, key=lambda r: r[5])
    print(
        f"\nMy palette (Excel Office):\n"
        f"  Worst pair under '{my_worst[0]}' simulation:\n"
        f"  S{my_worst[1]} {my_worst[2]}  vs  S{my_worst[3]} {my_worst[4]}\n"
        f"  ΔE = {my_worst[5]:.1f}  →  {verdict(my_worst[5])}\n"
        "\n"
        "  The red (#FF0000) and the green (#A9D18E) collapse under both\n"
        "  deuteranopia and protanopia: the two hues shift toward the same\n"
        "  yellow-brown, which is exactly why red/green charts fail for\n"
        "  roughly 1 in 12 men.\n"
    )

    course_worst = min(course_rows, key=lambda r: r[5])
    print(
        f"Course palette (3 + 4th):\n"
        f"  Worst pair under '{course_worst[0]}' simulation:\n"
        f"  S{course_worst[1]} {course_worst[2]}  vs  S{course_worst[3]} {course_worst[4]}\n"
        f"  ΔE = {course_worst[5]:.1f}  →  {verdict(course_worst[5])}\n"
        "\n"
        "  The 4th slot #CC79A7 (pink/mauve) collapses toward one of the blue\n"
        "  or orange slots under certain CVD simulations. ΔE drops to ~7–8,\n"
        "  which is in the 'needs a second channel' band — legal, but only if\n"
        "  the chart also uses a marker shape or direct label for every series\n"
        "  that uses the 4th colour. The three-slot palette without #CC79A7\n"
        "  is fully safe on its own.\n"
    )

    # ── Question 4: the rule for more than three categories ─────────────────
    print("=" * 60)
    print("QUESTION 4 — Rule for charts with more than three categories")
    print("=" * 60)
    print(
        "\nRule: when a chart needs a fourth or further category, do not add\n"
        "another colour. Instead, add a second encoding channel:\n"
        "\n"
        "  • Use a different marker shape (circle, square, triangle, diamond)\n"
        "    so identity is readable without colour.\n"
        "  • Use direct labels on each series rather than a legend, which removes\n"
        "    the need to distinguish colour at a glance.\n"
        "\n"
        "If colour is still needed beyond three slots, the 4th slot (#CC79A7) is\n"
        "allowed ONLY when every series that uses it carries a distinct marker shape\n"
        "AND a direct label — and that usage must be declared in the write-up.\n"
        "\n"
        "The stronger answer: if you genuinely need four categories, ask whether\n"
        "the chart is answering one question or two. A chart that needs four\n"
        "colours often needs to be split into two panels instead.\n"
    )


if __name__ == "__main__":
    main()
