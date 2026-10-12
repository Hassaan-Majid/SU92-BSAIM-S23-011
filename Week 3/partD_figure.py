"""
Part D — Clinical review figure  (25 marks)

Builds a 2×2 figure from vitals_10k.csv:
  Panel 1 (0,0): histogram of systolic BP — bin width declared on figure
  Panel 2 (0,1): box plot of length-of-stay by ward (log scale)
  Panel 3 (1,0): scatter of the pair the heatmap identifies as worth plotting
  Panel 4 (1,1): annotated correlation heatmap (lower triangle only, RdBu_r)

Run:
    python3 partD_figure.py

Writes partD_figure.png at 300 dpi.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vizlib import MARKERS, PALETTE, save_fig, style_defaults   # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
style_defaults()
C = PALETTE["series"]

# Audit trail: every exclusion with rule and count — printed at the end for Part E
LOG: dict[str, str] = {}


# =============================================================================
#  STEP 0 — inspect before plotting
# =============================================================================
def inspect(df: pd.DataFrame) -> None:
    print("── df.info() ──────────────────────────────────────────────────────")
    df.info()
    print("\n── df.describe().T ────────────────────────────────────────────────")
    print(df.describe().T.to_string())
    print("\n── ward value counts ──────────────────────────────────────────────")
    print(df.ward.value_counts())
    print("\n── admit_date sample (first 15) ───────────────────────────────────")
    print(df.admit_date.head(15).tolist())
    print("\n── extreme values ─────────────────────────────────────────────────")
    for col in ["heart_rate", "bmi", "spo2", "los_days", "systolic", "diastolic"]:
        print(f"  {col:18s}  min {df[col].min():>8.1f}   max {df[col].max():>8.1f}")


# =============================================================================
#  STEP 1 — clean
# =============================================================================
def clean(df: pd.DataFrame) -> pd.DataFrame:
    LOG["rows loaded"] = f"{len(df):,}"

    # ── Ward spellings ───────────────────────────────────────────────────────
    # Four real wards recorded with inconsistent capitalisation and spacing.
    ward_map = {}
    for raw in df.ward.unique():
        key = raw.strip().lower().replace(".", "").replace("-", " ").replace("  ", " ")
        ward_map[raw] = key
    df["ward_clean"] = df.ward.map(ward_map).str.strip()
    # Consolidate to canonical names
    canonical = {
        "icu": "ICU",
        "i c u": "ICU",
        "intensive care unit": "ICU",
        "general": "General",
        "general ward": "General",
        "gen ward": "General",
        "gen": "General",
        "cardiology": "Cardiology",
        "cardio": "Cardiology",
        "cardiac": "Cardiology",
        "orthopaedics": "Orthopaedics",
        "orthopaedic": "Orthopaedics",
        "ortho": "Orthopaedics",
        "orthopedics": "Orthopaedics",
        "orthopedic": "Orthopaedics",
    }
    df["ward_clean"] = df["ward_clean"].map(
        lambda s: canonical.get(s, s.title())
    )
    n_wards = df.ward_clean.nunique()
    missing_wards = df.ward_clean.isna().sum()
    LOG["ward spellings conformed"] = (
        f"{n_wards} canonical wards, {missing_wards} unmapped rows"
    )
    assert missing_wards == 0, f"Unmapped ward values: {df.loc[df.ward_clean.isna(), 'ward'].unique()}"

    # ── admit_date: three source formats ────────────────────────────────────
    # Systems export in: YYYY-MM-DD, DD/MM/YYYY, DD-Mon-YYYY (e.g. 01-Jan-2026)
    def parse_dates(series: pd.Series) -> pd.Series:
        formats = ["%Y-%m-%d", "%d/%m/%Y", "%d-%b-%Y", "%m/%d/%Y"]
        result = pd.Series(pd.NaT, index=series.index)
        remaining = series.copy()
        for fmt in formats:
            parsed = pd.to_datetime(remaining, format=fmt, errors="coerce")
            filled = parsed.notna()
            result[filled] = parsed[filled]
            remaining = remaining[~filled]
        return result

    df["admit_dt"] = parse_dates(df["admit_date"])
    unparsed = df["admit_dt"].isna().sum()
    LOG["dates parsed"] = (
        f"{df['admit_dt'].notna().sum():,} parsed; {unparsed} unparsed"
    )

    # ── BMI sentinels ────────────────────────────────────────────────────────
    mean_before = df.bmi.mean()
    sentinel_mask = df.bmi > 100          # no human BMI exceeds ~100 legitimately
    n_sentinels = sentinel_mask.sum()
    df.loc[sentinel_mask, "bmi"] = np.nan
    mean_after = df.bmi.mean()
    LOG["bmi sentinels removed"] = (
        f"n={n_sentinels} rows with bmi > 100 set to NaN; "
        f"mean bmi {mean_before:.2f} → {mean_after:.2f}"
    )

    # ── heart_rate device faults ─────────────────────────────────────────────
    # A human heart cannot sustain > 250 bpm. Readings above this threshold are
    # bedside-monitor faults. Removing them is physiologically correct. The mean
    # is reported before and after so the reader can judge the effect.
    mean_hr_before = df.heart_rate.mean()
    fault_mask = df.heart_rate > 250
    n_faults = fault_mask.sum()
    df = df[~fault_mask].copy()
    mean_hr_after = df.heart_rate.mean()
    LOG["heart_rate device faults removed"] = (
        f"n={n_faults} rows with heart_rate > 250 bpm removed; "
        f"mean hr {mean_hr_before:.1f} → {mean_hr_after:.1f} bpm"
    )

    # ── derived column ───────────────────────────────────────────────────────
    # map_mmhg = (systolic + 2 × diastolic) / 3  (standard clinical formula).
    # Its correlation with systolic and diastolic is arithmetic, not a discovery.
    # Excluded from the heatmap in panel 4.
    LOG["derived column excluded"] = (
        "map_mmhg excluded from heatmap — arithmetically derived from "
        "systolic and diastolic (MAP = (SBP + 2*DBP) / 3); correlation is "
        "not physiological."
    )

    LOG["rows after cleaning"] = f"{len(df):,}"
    return df


# =============================================================================
#  STEP 2 — build figure
# =============================================================================
def build_figure(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(2, 2, figsize=(13.0, 8.8))

    # ── Panel 1 (0,0): histogram of systolic BP ──────────────────────────────
    a = ax[0, 0]
    # Freedman-Diaconis rule for bin width
    q75, q25 = np.percentile(df.systolic.dropna(), [75, 25])
    iqr = q75 - q25
    n = df.systolic.dropna().shape[0]
    fd_width = 2 * iqr / (n ** (1 / 3))
    fd_width = max(fd_width, 1)          # floor at 1 mmHg
    bins = int(np.ceil((df.systolic.max() - df.systolic.min()) / fd_width))
    a.hist(df.systolic.dropna(), bins=bins, color=C[0], edgecolor="white",
           linewidth=0.4)
    a.set_xlabel("Systolic BP (mmHg)")
    a.set_ylabel("Count")
    a.set_title(
        f"Systolic BP peaks in two modes (~120 and ~145 mmHg)\n"
        f"[bins = {bins}, bin width ≈ {fd_width:.1f} mmHg — Freedman-Diaconis]",
        fontsize=10,
    )
    a.spines["left"].set_visible(False)
    a.spines["bottom"].set_visible(False)

    # ── Panel 2 (0,1): LOS by ward, log scale ───────────────────────────────
    b = ax[0, 1]
    # Sort wards by median LOS descending
    los_medians = df.groupby("ward_clean")["los_days"].median().sort_values(ascending=False)
    ward_order = los_medians.index.tolist()
    los_long = df[["ward_clean", "los_days"]].dropna()
    sns.boxplot(
        data=los_long,
        x="ward_clean", y="los_days",
        order=ward_order,
        palette={w: c for w, c in zip(ward_order, C + [PALETTE["series_4th"]])},
        ax=b,
        showfliers=True,          # outliers are retained and visible
        linewidth=0.8,
    )
    b.set_yscale("log")
    b.set_xlabel("Ward")
    b.set_ylabel("Length of stay (days, log scale)")
    b.set_title("ICU stays are the longest; LOS is right-skewed across all wards")
    b.spines["left"].set_visible(False)
    b.spines["bottom"].set_visible(False)

    # ── Panel 4 (1,1): heatmap first — tells us which pair is worth plotting ─
    d = ax[1, 1]
    # Exclude map_mmhg (arithmetic), spo2 (bounded), risk_score (ratio),
    # readmitted_30d (binary), patient identifiers.
    heat_cols = [
        "age", "systolic", "diastolic", "heart_rate",
        "bmi", "chol_mmol_l", "glucose_mmol_l", "los_days",
    ]
    corr = df[heat_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)   # hide upper triangle
    sns.heatmap(
        corr, ax=d,
        mask=mask,
        cmap=PALETTE["diverging"],   # RdBu_r — correlation has a sign
        vmin=-1, vmax=1, center=0,
        annot=True, fmt=".2f",
        annot_kws={"size": 7},
        linewidths=0.4,
        cbar_kws={"shrink": 0.75, "label": "Pearson r"},
    )
    d.set_title("Glucose and BMI show the strongest positive association")
    d.tick_params(axis="x", rotation=30, labelsize=8)
    d.tick_params(axis="y", rotation=0, labelsize=8)

    # ── Panel 3 (1,0): scatter of the pair the heatmap identifies ───────────
    # Heatmap shows glucose_mmol_l vs bmi as the highest biologically meaningful
    # correlation that is not an arithmetic artefact.
    c = ax[1, 0]
    scatter_df = df[["bmi", "glucose_mmol_l", "ward_clean"]].dropna()
    sample_df = scatter_df.sample(min(2000, len(scatter_df)), random_state=42)
    wards = sorted(sample_df.ward_clean.unique())
    for idx, ward in enumerate(wards):
        sub = sample_df[sample_df.ward_clean == ward]
        colour = (C + [PALETTE["series_4th"]])[idx % 4]
        c.scatter(
            sub.bmi, sub.glucose_mmol_l,
            label=ward, alpha=0.45, s=18,
            color=colour, marker=MARKERS[idx % len(MARKERS)],
        )
    c.set_xlabel("BMI (kg/m²)")
    c.set_ylabel("Fasting glucose (mmol/L)")
    c.set_title("Higher BMI associates with elevated fasting glucose\n"
                "(n=2 000 sampled; colour + shape = ward)")
    c.legend(title="Ward", fontsize=8, title_fontsize=8)
    c.spines["left"].set_visible(False)
    c.spines["bottom"].set_visible(False)

    # ── Finish ───────────────────────────────────────────────────────────────
    fig.suptitle(
        "Clinical safety review — vital-sign anomalies and risk indicators",
        fontsize=14, fontweight="bold", x=0.01, ha="left", y=1.02,
        color=PALETTE["ink"],
    )
    for p in ax.flat:
        p.spines["top"].set_visible(False)
        p.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


# =============================================================================
def main():
    df = pd.read_csv(os.path.join(ROOT, "data", "vitals_10k.csv"))

    # Comment out inspect() once you have read its output.
    # inspect(df)

    df = clean(df)
    fig = build_figure(df)

    n_final = len(df)
    n_hr = int(LOG["heart_rate device faults removed"].split("n=")[1].split(" ")[0])
    n_bmi = int(LOG["bmi sentinels removed"].split("n=")[1].split(" ")[0])

    note = (
        f"n = {n_final:,} · "
        f"{n_hr} heart_rate > 250 bpm removed as device faults · "
        f"{n_bmi} bmi > 100 removed as sentinel values · "
        f"map_mmhg excluded (arithmetic) · "
        f"bins = Freedman-Diaconis · "
        f"2 000 points sampled for scatter · "
        f"Synthetic data — CDB601220"
    )

    out = save_fig(fig, os.path.join(ROOT, "partD_figure.png"), note)
    plt.close(fig)
    print("wrote", out)

    print("\nCleaning log (Part E item 3):\n")
    for k, v in LOG.items():
        print(f"  {k:<42} {v}")


if __name__ == "__main__":
    main()
