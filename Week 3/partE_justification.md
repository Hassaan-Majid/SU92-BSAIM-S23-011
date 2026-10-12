# Part E — Written Justification  (15 marks)
**CDB601220 · Week 3 · Data Visualization**
**Roll No:** su92-bsaim-s23-012

---

## 1. Per-panel justification  (6 marks)

All data: `data/vitals_10k.csv` — 10,000 synthetic patient records, CDB601220.

### Panel 1 — Histogram of systolic BP

**Question answered:** How is systolic blood pressure distributed across
admitted patients?

**Chart type:** Histogram. A distribution is best shown as a histogram
when the variable is continuous and unbounded; a KDE would add smoothing
artefacts at the data's natural range boundaries (see Case 9, Part A).

**Channel rule (Week 2):** Position along a common scale (x-axis) is the
highest-accuracy channel for reading a distribution. Colour is used for
fill only — it carries no second variable and does not need to be
colour-blind safe here.

### Panel 2 — Box plot of length-of-stay by ward

**Question answered:** Does length of stay differ across wards, and if so,
how?

**Chart type:** Box plot. LOS is right-skewed (a small number of very long
stays inflate the mean); the box plot on a log scale shows the median and
spread without the extreme values collapsing the boxes. A bar chart of
mean LOS would be misleading because the mean is not representative of a
skewed distribution.

**Channel rule:** Position along a common scale (y-axis, log) is still
the strongest channel. Colour encodes ward identity redundantly with the
x-axis label — it adds a second encoding channel but is used within the
course palette (3 + 4th slot). The 4th slot (`#CC79A7`) requires direct
axis labels as a backup, which the x-axis provides.

**Outliers:** Retained and visible. `showfliers=True` (the default). Any
record that appears as an outlier is a legitimate data point; hiding it
with `showfliers=False` would obscure a potentially clinically significant
long stay without disclosure.

### Panel 3 — Scatter of BMI vs fasting glucose

**Question answered:** Is there an association between BMI and fasting
glucose across admitted patients?

**Chart type:** Scatter plot. Two continuous numeric variables; the
question is about their relationship. The heatmap in Panel 4 was built
first and identified BMI ↔ glucose as the highest biologically meaningful
correlation that is not an arithmetic artefact (`map_mmhg` was excluded).

**Parameter declared:** 2,000 rows sampled with `random_state=42` from
the cleaned dataset. 10,000 overplotted points would form a solid blob
at `alpha=1`; `alpha=0.45` and a 2,000-point sample make individual points
legible while preserving the trend.

**Channel rules:** Position encodes both variables. Colour and marker shape
encode ward (redundant coding — a second channel for categorical identity,
following Week 2's guidance on encoding beyond the first three categories).

### Panel 4 — Annotated correlation heatmap

**Question answered:** Which pairs of clinical variables are most strongly
associated, and are any relationships strong enough to warrant a closer
look?

**Chart type:** Heatmap. A correlation matrix has two categorical axes
(variable pairs) and one quantitative channel (Pearson r). A heatmap
encodes r with colour, which is lower accuracy than position but the only
practical encoding for an N×N matrix.

**Rules applied:**
- `cmap="RdBu_r"`, `vmin=-1`, `vmax=1`, `center=0`: a correlation has a
  sign. Using `"Reds"` with no vmin/vmax (Case 6, Part A) maps the observed
  range to the full colour scale, making r=0.30 look as dark as r=0.96.
- Upper triangle masked (`np.triu(..., k=1)`): a symmetric matrix says
  everything twice; showing both triangles halves the information density.
- `annot=True, fmt=".2f"`: direct annotation avoids the colour-legend
  lookup for any cell the reader wants to read precisely.
- `map_mmhg` excluded: arithmetically derived from `systolic` and
  `diastolic` (MAP = (SBP + 2·DBP) / 3); its correlations are structural,
  not physiological, and would mislead a clinical reader.

---

## 2. Every parameter declared and defended  (5 marks)

| Parameter | Value | Reason |
|---|---|---|
| Histogram bins | Freedman-Diaconis rule: `2·IQR·n^(−1/3)` | Adaptive to the data's spread and sample size; not the default (which is not a decision) |
| Scatter sample size | 2,000 rows, `random_state=42` | 10,000 overplotted points are unreadable; sampling is reproducible |
| Scatter `alpha` | 0.45 | Enough transparency to show density without hiding individual points |
| Box plot y-scale | `log` | LOS is right-skewed; linear scale collapses all boxes to a line |
| Box plot `showfliers` | `True` (default, not suppressed) | Ethics — see Section 3 |
| Heatmap `vmin`/`vmax` | −1 / 1 | A correlation is bounded; mapping the observed range would distort perceived magnitude |
| Heatmap mask | Upper triangle | Symmetric matrix; showing both halves is redundant |
| Colour palette | `PALETTE["series"]` + 4th slot | Okabe-Ito subset, verified colour-blind safe (Part C); 4th slot used only with axis labels as backup channel |
| Figure DPI | 300 | Clinical-quality export; required by the brief |
| `bbox_inches` | `"tight"` | Prevents source note and suptitle from being clipped |

---

## 3. Every row removed, with rule, count, and effect  (4 marks)

Three exclusion rules were applied, in this order:

### Rule 1 — BMI sentinel values

> **Rule:** `bmi > 100` treated as a sentinel for missing data.
> **Count:** *n* rows removed (printed by the script on each run).
> **Effect:** Mean BMI moves from the script-printed value to a lower, plausible value.

No human BMI legitimately exceeds 100. Values above this threshold are
database sentinels (commonly 999 or 9999) used by some source systems to
represent "not recorded". Keeping them would inflate the mean by orders of
magnitude and corrupt the scatter in Panel 3.

### Rule 2 — Heart rate device faults

> **Rule:** `heart_rate > 250 bpm` treated as a bedside-monitor fault.
> **Count:** *n* rows removed (printed by the script on each run).
> **Effect:** Mean heart rate moves from the script-printed value to a lower, physiologically plausible value.

The maximum sustained human heart rate is approximately 200–220 bpm in
elite athletes. Readings above 250 bpm are impossible in a resting or
ambulatory setting and indicate a sensor fault. Removing them is
physiologically correct and required.

**Ethics clause:** `showfliers=False` was NOT used in Panel 2. Hiding
outliers with a Seaborn parameter without disclosure would be a silent
edit. The exclusion rule above is stated explicitly: rows were removed
from the dataframe before plotting, with the count and mean-shift logged.
The figure was also run without the exclusion to confirm that retaining
the device-fault rows collapses the box plot to a near-zero box with an
extreme whisker — the exclusion does not weaken a conclusion, it makes
one possible.

**Symmetry test (Part E ethics clause):** Would the same removal have
been made if heart_rate > 250 had been associated with a shorter LOS
instead of a longer one? Yes — the rule is physiological (> 250 bpm is
impossible), not directional. It holds regardless of which conclusion it
supports.

### Rule 3 — `map_mmhg` excluded from heatmap

> **Rule:** `map_mmhg` excluded from the correlation matrix (Panel 4).
> **Count:** 0 rows removed — this is a column exclusion, not a row exclusion.
> **Effect:** Removes three cells from the heatmap (map_mmhg ↔ systolic,
> map_mmhg ↔ diastolic, map_mmhg ↔ map_mmhg) that would show correlations
> of ~0.96 and ~0.99 — both arithmetic, not physiological.

MAP = (SBP + 2·DBP) / 3 is a formula, not a measurement. The correlation
of map_mmhg with systolic is structural: it must be high by definition.
Displaying it as a finding would mislead a clinical reader into thinking
a novel association had been discovered.

---

*All data in this submission is synthetic and generated for CDB601220.
No patient-identifiable information is present.*
