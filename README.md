# DiabetesInterveneBD

**Visual analytics for geographic targeting of diabetes interventions in Bangladesh.**

District diabetes prevalence in Bangladesh spans a sevenfold range — 4.8% in
Natore, 33.6% in Narayanganj. A conventional choropleth of those numbers is
misleading, because the survey behind them supports wildly different levels of
precision district to district: 692 respondents in Dhaka, 40 in the thinnest
district, and design effects up to 3.4. This tool encodes prevalence and
precision in the same visual channel, finds each district's structural peers,
decomposes its excess risk exactly, and simulates what a funded programme would
be worth there — with an interval that widens where the evidence thins.

Built on BDHS 2022 (14,167 adults, 674 sampling clusters, 64 districts).

---

## Screens

<!-- Replace these with real captures after your first deploy.
     Suggested set: (1) the VSUP map with the wedge legend visible,
     (2) the parallel-coordinates peer view, (3) the Shapley waterfall,
     (4) the tornado chart with the wrong-signed lever flagged. -->

| | |
|---|---|
| `[docs/img/01-map.png](https://drive.google.com/file/d/1i_TrtCZawgb7YcwWmUc9ekoI5cQFuBvq/view?usp=sharing)` | `https://drive.google.com/file/d/1RRh51erjyfuPLfj22rxYE5Afw41apc4v/view?usp=sharing` |
| `[docs/img/03-drivers.png](https://drive.google.com/file/d/12cRy85bW4qDeFrYtoJCBk3xEz5PbhRWB/view?usp=sharing)` | `[docs/img/04-intervention.png](https://drive.google.com/file/d/1f2vl25dbVhjYOb8s1fKW90I7uPJPH2Oo/view?usp=sharing)` |

**Live demo:** [_add your Streamlit Community Cloud URL here_](https://diabetesintervenebd-gmk7pukpmt2mgaaym3o4nl.streamlit.app/)

---

## Quickstart

```bash
git clone https://github.com/<you>/interveneBD.git
cd interveneBD
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# One-off: fetch the district boundaries (the simplified version is committed,
# so you only need this if you want to rebuild the geometry yourself).
curl -L -o data/raw/bd_districts_raw.geojson \
  https://raw.githubusercontent.com/nuhil/bangladesh-geocode/master/geojson/districts.geojson
python scripts/01_prepare_geo.py

# Build estimates, profiles, and models (~1s)
python scripts/02_build_artifacts.py

streamlit run app.py
```

The app also builds its own artifacts lazily on first load, so a fresh clone
deployed straight to Streamlit Community Cloud works without running anything.

**Verify the build:**

```bash
python -m pytest tests/ -q   # 22 tests: estimator, similarity, Shapley axioms
python -m src.theme          # palette luminance + colour-vision-deficiency check
```

---

## What each page does

| Page | Question it answers | Core technique |
|---|---|---|
| **Risk map** | Where is the burden, and how much of what I see is real? | Value-suppressing uncertainty palette over cluster-bootstrapped estimates |
| **Peer districts** | Is this district unusual, or just unusually urban? | Gower similarity over 10 profile axes; residual against structural twins |
| **Risk drivers** | What accounts for the gap to the national average? | Exact interventional Shapley, all 2¹⁰ coalitions enumerated |
| **Intervention lab** | What would a programme at *this* coverage actually buy? | Counterfactual re-scoring with cluster-bootstrap intervals |
| **Priority shortlist** | Which ten districts, and can I defend each one? | Three-criteria weighting with cross-scenario stability check |
| **Model and methods** | Should anyone believe the other five pages? | Calibration curves, odds ratios, explicit limitations |

---

## Methods

**Point estimates.** Hájek ratio estimator `Σwy / Σw` using the BDHS sampling
weight. The unweighted mean is retained alongside for comparison: weighting
moves a district estimate by a median of 0.49 pp and by 7.01 pp in Rajshahi.

**Intervals.** Nonparametric bootstrap resampling the 674 primary sampling units
with replacement, 400 replicates, percentile method. Resampling clusters rather
than individuals is what preserves intra-cluster correlation. Median design
effect 1.15, maximum 3.42 — at that maximum, a simple-random-sample standard
error understates the truth by a factor of 1.85.

**Uncertainty encoding.** Value-suppressing uncertainty palette
(Correll, Moritz & Heer, CHI 2018). Four precision tiers carrying 8, 4, 2 and 1
distinguishable value bins. A district the survey cannot pin down loses colour
resolution rather than being rendered with false authority. The conventional
palette is available as a toggle — the A/B is the usability experiment, not a
convenience.

**Similarity.** Gower's coefficient over ten survey-weighted composition axes,
range-normalised per axis, with user-adjustable per-axis weights. Every axis is
a proportion but their spreads differ by a factor of three, so raw Euclidean
distance would let urban share dominate purely by having more room to vary.

**Attribution.** Exact Shapley values over the full 1,024-subset coalition
lattice, interventional (marginal) formulation following Janzing, Minorics &
Blöbaum (AISTATS 2020). The value function compares the district's joint
distribution against the national one, so the decomposition answers "why is this
district different" rather than "why is this person at risk". The efficiency
identity `Σφ = v(N) − v(∅)` is asserted at runtime and shown in the interface.

**Risk models.** Survey-weighted logistic regression and histogram gradient
boosting, both fitted on the same split, both exposed in the UI. Test AUC 0.688
and 0.681 respectively. The simpler model winning is reported rather than
quietly discarded: with ten coarse categorical predictors and a largely additive
risk structure there is little interaction for a tree to find.

---

## Architecture

```
app.py                        Risk map — entry point
pages/
  1_Peer_districts.py         Gower similarity, parallel coordinates, residuals
  2_Risk_drivers.py           Exact Shapley waterfall
  3_Intervention_lab.py       Counterfactual simulator, leverage ranking
  4_Priority_shortlist.py     Multi-criteria ranking, stability check
  5_Model_and_methods.py      Performance, calibration, limitations
src/
  config.py                   Paths, survey design, intervention definitions
  theme.py                    Design tokens, VSUP colormap, CVD verification
  survey.py                   Hájek estimator, stratified cluster bootstrap
  profiles.py                 Profile matrix, Gower distance, residuals
  model.py                    Both risk models, calibration, odds ratios
  shapley.py                  Exact grouped interventional Shapley
  counterfactual.py           Coverage-scaled intervention simulator
  charts.py                   All Plotly figures
  loaders.py                  Cached loading, lazy artifact build
  ui.py                       Shared components, cross-page selection state
scripts/
  01_prepare_geo.py           Alias-map names, simplify polygons, centroids
  02_build_artifacts.py       Estimates, profiles, models
tests/                        22 tests
docs/
  design_decisions.md         Choices made, alternatives rejected, one mistake
  district_name_map.md        The seven-name join problem
  usability_study.md          Two-phase evaluation protocol
```

---

## Design notes

The palette is built from delta materials rather than a stock colormap:
standing water → deep channel → river current → reed bank → wet silt → exposed
sandbar. Interactive elements use river teal `#35C4B5`; alerts use laterite
`#DC5B3E`, the brick every rural clinic in the country is built from.

Luminance rises monotonically across the ramp, and the minimum separation
between adjacent stops under simulated deuteranopia is 53/255. Both are verified
in code, not asserted — `python -m src.theme` re-checks after any edit. The
first draft failed that check and the correction is recorded in
`docs/design_decisions.md`.

---

## Findings

<!-- Fill this in after running the study in docs/usability_study.md.
     Report the effect size and its interval. Do not report p-values at n<20. -->

_Usability study pending. Protocol pre-registered in `docs/usability_study.md`;
primary measure is the count of selected districts whose 95% interval overlaps
the national mean._

---

## Limitations

**No time dimension.** BDHS 2022 is a single cross-section. Nothing here can say
whether a district is improving or worsening.

**No causal identification.** The simulator moves people between observed
categories and re-scores them under a model fitted on associations. Reading its
output as a programme effect requires no unmeasured confounding, positivity and
consistency; this dataset can test none of them. The physical-activity lever
returns a *negative* reduction in most districts — almost certainly reverse
causation, since adults already diagnosed are the adults told to exercise. It is
displayed and flagged rather than suppressed, because it is the clearest
demonstration on the dashboard that these are associations.

**Self-reported diagnosed diabetes.** The outcome captures people who know they
have diabetes. Better health-service coverage produces more recorded diabetes,
so part of the urban-rural and wealth gradient is diagnostic access rather than
disease.

**Thin districts.** Thirteen districts have fewer than 100 respondents; the
smallest has 40. The palette prevents over-reading them. No palette recovers
information that was never sampled.

**Ten coarse predictors.** No fasting glucose, HbA1c, family history, diet or
anthropometry. This is a composition summary, not a clinical risk score, and
must never be used for individual screening.

---

## Data

- **Survey:** Bangladesh Demographic and Health Survey 2022 extract, 14,167
  adults with sampling weight and cluster identifiers. Not redistributed here
  beyond the analysis extract; obtain the full dataset from
  [dhsprogram.com](https://dhsprogram.com).
- **Boundaries:** district polygons via
  [nuhil/bangladesh-geocode](https://github.com/nuhil/bangladesh-geocode),
  simplified to 0.004°. Seven names remapped — see `docs/district_name_map.md`.

---

## References

Correll, M., Moritz, D., & Heer, J. (2018). Value-suppressing uncertainty
palettes. *CHI 2018*.

Gower, J. C. (1971). A general coefficient of similarity and some of its
properties. *Biometrics*, 27(4).

Janzing, D., Minorics, L., & Blöbaum, P. (2020). Feature relevance
quantification in explainable AI: a causal problem. *AISTATS 2020*.

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model
predictions. *NeurIPS 2017*.

Munzner, T. (2014). *Visualization Analysis and Design.* CRC Press.

Rust, K. F., & Rao, J. N. K. (1996). Variance estimation for complex surveys
using replication techniques. *Statistical Methods in Medical Research*, 5(3).

---

## Author

Amartay Kumar Dhar — Jahangirnagar University

## License

MIT for the code. The survey data is governed by the DHS Program's terms of use.
