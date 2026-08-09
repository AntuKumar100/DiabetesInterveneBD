# Design decisions

A running log of the choices that were not obvious, including the ones that were
wrong the first time. Kept because "why is it like this" is the question that
costs the most time six months later, and because a decision with no recorded
alternative is indistinguishable from a default.

---

## D1 — Uncertainty gets a channel, not a footnote

**Decision.** The primary map uses a value-suppressing uncertainty palette
(Correll, Moritz & Heer, CHI 2018). Colour resolution collapses as the standard
error rises: 8 distinguishable value bins at the top tier, then 4, then 2, then
1.

**Alternatives considered.**

| Option | Why rejected |
|---|---|
| Hatching or texture overlay on unreliable districts | Reads as a second, separable layer. Users learn to ignore it in about ten seconds. |
| Opacity encoding | Confounds with the basemap; a faded district over a dark background just looks like a different value. |
| Footnote with a table of standard errors | Nobody reads it. The map still lies. |
| Drop unreliable districts entirely | Biases the shortlist toward large districts, which is the opposite of the tool's purpose. |

**The argument.** Every rejected option lets the reader extract a precise value
and then optionally discount it. Value suppression removes the precise value
from the encoding, which means the reader *cannot* over-read the map even if
they want to. That asymmetry is the entire point.

**Cost.** It is unfamiliar, and users need the legend. That is exactly what the
usability study measures, and why the conventional palette is kept as a toggle
rather than deleted — the A/B is the experiment.

---

## D2 — Palette derived from the subject, then verified

**Decision.** Six-stop ramp: standing water `#0E3038` → deep channel `#15637A`
→ current `#22A08F` → reed bank `#79C07C` → wet silt `#E6B24A` → exposed sandbar
`#FFCE9A`. Interactive accent is river teal `#35C4B5`; alerts use laterite
`#DC5B3E`, the colour of the brick every rural clinic in Bangladesh is built
from.

**The mistake, kept on the record.** The first draft ended the ramp on saturated
laterite because it looked better in isolation. Verification caught it:
relative luminance *dropped* from the gold stop to the red one, so the two
highest-risk bands were non-monotonic in lightness, and under simulated
deuteranopia adjacent stops separated by only 26/255 in RGB. Ending on exposed
sand instead makes luminance rise at every step and raises the minimum adjacent
gap to 53/255.

**How to re-check after any palette edit.**

```bash
python -m src.theme
```

Prints luminance per stop, asserts monotonicity, and reports the minimum
adjacent separation under deuteranopia and protanopia.

---

## D3 — Exact Shapley by enumeration, no explainer library

**Decision.** `src/shapley.py` enumerates all 2^10 = 1,024 coalitions and
computes Shapley values by definition.

**Why not `shap`.** Two reasons. The library's per-row attributions answer "why
this person", and the question here is "why this district versus the nation" —
a comparison of distributions. And with ten features the full lattice is small
enough that approximation buys nothing: KernelSHAP would introduce sampling
error to save a computation that takes three seconds.

**Interventional, not conditional.** Janzing, Minorics & Blöbaum (AISTATS 2020).
Under the conditional formulation a feature that was never modified still
collects credit whenever it correlates with one that was. Since this
decomposition feeds an intervention decision, credit has to follow what actually
changed.

**Verification.** The efficiency identity `Σφ = v(N) − v(∅)` is asserted at
runtime and displayed on the page as `efficiency_error`. On the toy additive
model in `tests/test_attribution.py` it holds to 1e-9, and the null-player and
additivity axioms are tested directly.

---

## D4 — Cluster bootstrap, not `groupby().mean()`

**Decision.** Point estimates use the Hájek ratio estimator with the BDHS
weight. Intervals come from 400 bootstrap replicates that resample the 674
enumeration areas with replacement.

**What it costs to skip.** Median design effect across districts is 1.15, and it
reaches 3.42. At a design effect of 3.4 a naive standard error is understated by
a factor of 1.85 — the difference between "these two districts clearly differ"
and "we cannot tell". Weighting alone shifts the district estimate by a median
of 0.49 pp and by 7.01 pp in Rajshahi.

**Why percentile intervals.** District proportions sit near a boundary and the
bootstrap distributions are visibly skewed for the sparse districts; a normal
interval would push the lower bound below zero for several of them.

**Single-cluster districts.** Return `nan` for the interval rather than a
fabricated one. A single cluster carries no between-cluster information.

---

## D5 — Both models shipped, switchable in the UI

**Decision.** A survey-weighted logistic regression and a histogram gradient
booster are both fitted and both exposed. The sidebar switches between them, and
the drivers page can run both and flag any feature whose sign flips.

**The result is the argument.** AUC 0.688 for the logistic model and 0.681 for
the booster. With ten categorical predictors and a largely additive risk
structure, there are few interactions for a tree to find. Shipping only the
booster would have implied a sophistication the data does not support; shipping
only the logistic model would have left the question untested. Reporting that
the fancy model does not win is more informative than either.

---

## D6 — Wrong-signed levers are displayed, not suppressed

**Decision.** The physical-activity intervention produces a *negative*
reduction in most districts — the model says increasing activity raises
predicted risk. This is shown, flagged in an amber callout, and explained.

**Why not hide it.** It is the single clearest available demonstration that
these simulations are associational. In a cross-section, adults already
diagnosed with diabetes are precisely the adults who have been told to exercise
and to lose weight, so the exposure carries the diagnosis rather than preventing
it. A dashboard that silently clipped negative effects at zero would look more
professional and be less honest.

---

## D7 — Selection lives in session state

**Decision.** The focused district is stored in `st.session_state` and every
page reads it. Streamlit's default gives each page an independent widget.

**Why it matters.** The tool is a sequence — locate, compare, explain, simulate,
decide. Losing the selection between steps forces the user to re-answer a
question they already answered, and in pilot walkthroughs that is exactly where
attention breaks.

---

## D8 — Boundaries simplified and committed

**Decision.** The 4.6 MB source GeoJSON is simplified to 0.004° (~400 m) and the
resulting 484 KB file is committed rather than gitignored.

**Trade-off.** Committing generated artifacts is normally poor practice. Here
the upstream source is a third-party repository that could move or change
schema, and a deployed app with a broken map is a far worse outcome than 484 KB
in version control. The download command lives in the README and in
`scripts/01_prepare_geo.py` so the artifact can always be rebuilt.

---

## Open questions

- The reliability tier cut-points (2.5 / 4.5 / 7.0 pp) are judgement, not
  derived. A principled alternative would set them from the distribution of
  standard errors so the tiers hold roughly equal numbers of districts. Worth
  testing whether users read the current tiers as absolute or relative.
- Peer-similarity weights currently default to uniform. A learned weighting —
  axes weighted by how much they predict the outcome — would find more relevant
  peers but would make the residual partly circular. Not resolved.
- The intervention simulator applies levers independently. Real programmes
  interact, and there is no data here to estimate that interaction.
