# Project guide — for Amartay

This is the private companion to the README. The README sells the project; this
explains it, so that you can answer any question about any line of it without
me in the room.

Read it once end to end, then keep it open while you click through the app.

---

## 0. The one-sentence version

> Bangladesh's district diabetes prevalence varies sevenfold, but the survey
> measures some districts far better than others, so I built a tool that shows
> burden and precision in the same visual channel, finds each district's
> structural peers, decomposes its excess risk exactly, and simulates what a
> funded programme would buy — with intervals that widen where the evidence is
> thin.

If you can say that from memory, you can open any conversation about this
project. Everything below is elaboration.

---

## 1. Why this project and not the original Project 3

Your first Project 3 was a scroll-based story about **trends**. Your dataset has
no time variable — no year, no survey round, one cross-section. Every temporal
claim in that plan was unbuildable. Rather than fake it with two rounds you
don't have, this replaces "change over time" with "variation across space and
structure," which the data genuinely supports.

It also moves the project into Klaus Mueller's actual research lane. His VAI Lab
works on visual analytics, explainable AI, algorithmic fairness, and — in
XplainAct (IEEE VIS 2025) — *personalised causal intervention analysis on
epidemiological data, funded by the CDC*. The interface pattern there is a
choropleth of a health outcome, a peer-comparison panel, a feature-attribution
panel, and similarity controls. That is the same skeleton as this project, in a
country nobody in his lab has data for.

**Say that explicitly in your email.** Not "I built a dashboard" — "I applied
the XplainAct interaction pattern to a complex-survey setting, which forced me
to solve the uncertainty problem their county-level data doesn't have."

---

## 2. Run it, in order

```bash
cd interveneBD
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/02_build_artifacts.py     # ~1 second
streamlit run app.py
```

If the map is blank, the geometry is missing:

```bash
curl -L -o data/raw/bd_districts_raw.geojson \
  https://raw.githubusercontent.com/nuhil/bangladesh-geocode/master/geojson/districts.geojson
python scripts/01_prepare_geo.py
```

Check everything still works after any edit:

```bash
python -m pytest tests/ -q   # should say 22 passed
python -m src.theme          # palette check
```

---

## 3. The 14 planned steps, and where each one lives

| Step from the plan | Where it is | What to say about it |
|---|---|---|
| 1. Frame the stakeholder task | `pages/4_Priority_shortlist.py` | The decision task is "fund eight districts and justify each." Everything upstream serves that. |
| 2. Weighted estimation layer | `src/survey.py` | Hájek estimator + stratified cluster bootstrap. The heart of the project's credibility. |
| 3. District profile matrix | `src/profiles.py::build_profiles` | 64 × 10, survey-weighted composition shares. |
| 4. Peer similarity | `src/profiles.py::gower_distance` | Gower, with user-controlled axis weights exposed as UI sliders. |
| 5. Risk models | `src/model.py` | Logistic + gradient boosting, both shipped. |
| 6. Counterfactual engine | `src/counterfactual.py` | Coverage-scaled, cluster-bootstrapped. |
| 7. Causal honesty | `pages/5_Model_and_methods.py`, plus a caveat on every page | Written first, not last. |
| 8. Visual encodings | `src/charts.py` | One ramp, one grammar, shared axis ranges. |
| 9. Uncertainty encoding | `src/theme.py::VSUP` | The signature contribution. |
| 10. Wireframe before coding | You still owe this — see §8 | |
| 11. Linked coordinated views | `src/ui.py` session state + `on_select` map clicks | |
| 12. Two-phase user study | `docs/usability_study.md` | You still owe this — see §8. |
| 13. Iterate and document | before/after screenshots into `docs/img/` | You still owe this. |
| 14. Deploy and write up | `README.md` | Deploy, then paste the URL in. |

---

## 4. Module by module, what it does and why it matters

### `src/config.py`
Every path, column name, and domain constant in one place. The important part is
`MODIFIABLE` vs `STRUCTURAL`: you cannot intervene on someone's age, so only four
features are ever exposed as levers. If someone asks "how do you stop the tool
from proposing that we make people poorer," this is the answer.

### `src/survey.py` — read this one twice
This is the module that separates your project from a class assignment.

- `hajek()` — weighted proportion, `Σwy / Σw`.
- `_bootstrap_group()` — resamples **clusters**, not people. Vectorised so 64
  districts × 400 replicates runs in about a second instead of minutes.
- `district_estimates()` — returns point estimate, percentile CI, SE, and the
  design effect.

**The question you will be asked:** *why bootstrap clusters rather than
individuals?* Because people in the same enumeration area resemble each other,
so each extra respondent adds less information than an independent draw. Resample
individuals and you destroy that correlation and understate the variance. Your
median design effect is 1.15 and the maximum is 3.42 — at 3.42, a naive standard
error is 1.85× too small.

**Second question:** *why percentile intervals rather than normal?* District
proportions sit near zero and the bootstrap distributions are skewed for sparse
districts; a normal interval would put the lower bound below zero for several.

### `src/profiles.py`
Ten profile axes, Gower distance, and `structural_residual()` — each district's
prevalence minus its peers' mean.

**The question:** *why Gower and not Euclidean?* Every axis is a proportion but
their spreads differ by a factor of three. Raw Euclidean lets urban share
dominate just because it has more room to vary. Gower range-normalises each axis
first, and it lets you attach a per-axis weight the user controls — which is why
the "socioeconomic only" and "behavioural only" presets are real filters rather
than cosmetic.

### `src/model.py`
Both models, sample weights passed to `fit`, calibration curve computed.

**The question you should welcome:** *your AUC is only 0.68.* Correct, and it
should be. Ten coarse categorical predictors, no fasting glucose, no HbA1c, no
family history, three age bands. An AUC of 0.90 on these columns would be
evidence of leakage. And for this application calibration matters more than
discrimination, because the simulator averages predicted probabilities and reads
the average as a prevalence — that step is valid if the model is calibrated and
invalid otherwise, at any AUC.

Note also: the logistic model *beats* the booster, 0.688 to 0.681. Do not hide
that. Saying "the sophisticated model did not win, and here is why" reads as
maturity. Hiding it reads as a student who only knows one answer.

### `src/shapley.py` — your strongest single piece of code
Exact Shapley over all 1,024 coalitions, no `shap` dependency.

**Why it is not a per-row explainer.** The question is "why is this district
different from the nation," which compares two *distributions*. So the value
function `v(S)` is the mean prediction when features in S come from the
district's joint distribution and the rest from the national one.

**Interventional, not conditional** (Janzing et al., AISTATS 2020). Under the
conditional formulation, a feature that was never touched still collects credit
when it correlates with one that was. Since this feeds an intervention decision,
credit has to follow what actually changed.

**The killer detail:** the app displays `efficiency_error` — `Σφ` minus the gap.
It comes back at `0.00000`. That is not a claim that your decomposition is
correct; it is a proof, printed in the interface, that a reader can check. Point
at it.

### `src/counterfactual.py`
Coverage is explicit and partial. Each replicate resamples the district's
clusters *and* re-draws who the programme reaches, so the interval carries both
survey and rollout variability.

**The finding to lead with, not hide:** the physical-activity lever returns a
negative reduction. The model says more exercise means more predicted diabetes.
That is reverse causation — in a cross-section, the adults told to exercise are
the adults already diagnosed. The app flags it in amber and explains it.

When you present this, say: *"the most important output of my simulator is the
one that's wrong, because it's the one that proves the tool is honest about
being associational."* That single sentence will do more for you than any chart.

### `src/theme.py`
Design tokens and the VSUP. Also a dichromat simulation used at build time.

**The palette story, which you should tell:** the ramp walks a river delta —
standing water, deep channel, current, reed bank, wet silt, exposed sandbar. My
first version ended on saturated laterite red because it looked better. The
verification caught that luminance *dropped* at the last step, and adjacent
stops separated by only 26/255 under simulated deuteranopia. Ending on exposed
sand fixed both, raising the gap to 53/255.

Telling that story — including the mistake — is worth more than a perfect
palette with no explanation.

### `src/ui.py`
The selected district lives in `st.session_state`, so it survives page
navigation. Streamlit's default is a fresh widget per page, which breaks the
analytic thread. `selected_from_plotly()` handles map clicks defensively and
always falls through to the selectbox, because a click accelerator that becomes
the only way to do something is an accessibility failure.

---

## 5. Reading each page out loud

Practise this. Two minutes per page.

**Risk map.** "Colour is prevalence, and colour *resolution* is precision.
Across the top of the legend the survey supports eight distinguishable levels;
by the bottom row it supports one. A grey district isn't low risk — it's a
district we haven't measured well enough to place. Toggle the palette off and
watch Munshiganj, 72 respondents, acquire exactly the same visual authority as
Dhaka with 692. That's the failure mode this exists to prevent."

**Peer districts.** "Ranking by prevalence finds the biggest burden. It doesn't
find the most surprising one. Narayanganj at 33.6% would be unremarkable if
every wealthy urban district sat there — but its structural twins average 22.4%,
so there's an 11-point gap its composition doesn't explain. That gap is where
you look."

**Risk drivers.** "This is a Shapley decomposition of a difference between two
populations. Ten features, 1,024 coalitions, enumerated exactly. The efficiency
check is printed at the top — the bars sum to the gap to five decimal places.
For Narayanganj, wealth composition carries almost the entire gap, and the four
modifiable factors carry very little, which is a finding: this is a structural
problem more than a behavioural one."

**Intervention lab.** "Coverage is explicit because real programmes never reach
everyone. The interval resamples clusters and re-draws who gets reached. And
look at physical activity — negative. That's reverse causation, and I show it
rather than clipping it at zero."

**Priority shortlist.** "Three criteria that genuinely conflict. Watch the
shortlist move as I change the weights, and watch which districts survive all
four scenarios. That intersection is the part of the answer that isn't my
opinion."

**Methods.** "This is the page that undermines the other five, and it's the one
I'd want a reviewer to read first."

---

## 6. Hard questions, and honest answers

**"Isn't this just a dashboard?"**
The dashboard is the apparatus. The contribution is the uncertainty encoding and
the evaluation of whether it changes targeting decisions. That's the study in
`docs/usability_study.md`.

**"Why should I trust a model with AUC 0.68?"**
You shouldn't trust it for individual screening, and the Methods page says so.
For population averages what matters is calibration, which is reported.

**"Your interventions aren't causal."**
Correct, stated on every page, and one lever visibly points the wrong way
because of it. I'd need panel data or an instrument to make causal claims, and
BDHS 2022 gives me neither.

**"Why not just use `shap`?"**
Because the question compares distributions, not rows, and with ten features the
full lattice is small enough to enumerate exactly. Approximating would add error
to save three seconds.

**"Did you build this yourself?"**
Answer honestly. You designed the analysis, you own the data, you know why every
statistical choice was made, and you'll have run the user study. Say that you
used AI assistance for implementation the same way you'd say you used
scikit-learn — and then demonstrate command by answering the questions above
without notes. **The command is the credential.** Nobody can fake §4 of this
document in a live conversation.

---

## 7. Deploying

1. Push to GitHub. `data/processed/risk_model.joblib` is gitignored — the app
   rebuilds it on first load in about a second.
2. share.streamlit.io → New app → point at `app.py`.
3. Python 3.11. `requirements.txt` is already pinned.
4. First load takes ~30 seconds while it installs and builds. After that it's
   fast.
5. Paste the URL into the README and into your email.

If the map is blank on the deployed version, `data/processed/bd_districts.geojson`
didn't get committed. It's deliberately **not** gitignored — check it's there.

---

## 8. What you still owe — and this is the part that matters

Everything above is built. These three are not, and they are what convert a
competent build into a portfolio piece:

**Wireframe (Step 10).** Sketch the four-panel layout on paper. Photograph it.
Two minutes of work, and it's evidence you designed before you coded. Put it in
`docs/img/`.

**Usability study (Step 12).** Protocol is written and pre-registered in
`docs/usability_study.md`. Phase 1 is five people and one afternoon. Phase 2 is
twelve to sixteen people across a week. Primary measure: how many of the eight
districts a participant picks have intervals overlapping the national mean.

**Before/after evidence (Step 13).** Screenshot everything you change after
Phase 1. Two images side by side, with one sentence on what the confusion was.

Do the study. Without it you have a good dashboard. With it you have a small
piece of HCI research, and that is what the person you're writing to actually
does for a living.

---

## 9. The email

Short. Link first, one specific technical hook, one honest limitation, one ask.

> Dear Professor Mueller,
>
> Thank you for your reply. I've built a working prototype that I hope shows the
> CS, HCI and visualization side of my work concretely: [URL]
>
> It's a visual analytics tool for targeting diabetes interventions across
> Bangladesh's 64 districts, using BDHS 2022 (14,167 adults, 674 sampling
> clusters). The interaction pattern is deliberately close to XplainAct —
> choropleth, peer comparison, attribution panel, similarity controls — applied
> to a complex survey, which introduced a problem county-level data doesn't
> have: precision varies enormously across districts, from 692 respondents down
> to 40. I encode that with a value-suppressing uncertainty palette over
> cluster-bootstrapped estimates, so districts the survey can't pin down lose
> colour resolution instead of being rendered with false authority.
>
> The attribution page computes exact Shapley values over the full coalition
> lattice rather than approximating, and displays the efficiency residual so the
> decomposition is checkable. The intervention simulator is explicitly
> associational — one lever returns a negative effect, which I show and explain
> as reverse causation rather than suppressing.
>
> I'm currently running a two-condition study testing whether the uncertainty
> palette changes targeting decisions, not just comprehension. I'd value your
> view on whether this direction fits your group.
>
> Amartay Kumar Dhar
> Jahangirnagar University

Send it only once the URL is live and you've clicked every page yourself.

One small thing: it's **Stony** Brook, not Stoney. Check every email before you
send it.
