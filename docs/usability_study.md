# Usability study protocol

The dashboard is the apparatus, not the finding. The finding comes from this.

Two phases. Phase 1 is formative and fixes the interface. Phase 2 is summative
and tests one hypothesis about uncertainty encoding.

---

## Hypothesis

> When district-level estimates carry heterogeneous precision, a
> value-suppressing uncertainty palette produces better *targeting decisions*
> than a conventional sequential palette, without a meaningful cost in reading
> speed.

"Better" is defined operationally below. The measure is deliberately not
comprehension — a reader can correctly report "Munshiganj is 25.9%" from either
map. The question is whether they then *choose* Munshiganj, whose interval runs
from 14.0 to 31.5 on a sample of 72 adults.

---

## Phase 1 — Formative (n = 5–6)

**Who.** Public health students or junior programme staff. Anyone who has read a
choropleth before and has not seen this tool.

**Format.** 30 minutes, think-aloud, screen recorded with permission. You do not
help. When they get stuck, note the timestamp and say "what would you try
next?"

**Tasks.**

1. Find the district with the highest diabetes prevalence.
2. Explain, in your own words, what the grey districts mean.
3. Find a district that does worse than districts similar to it.
4. Decide which single intervention you would fund in Dinajpur, and say why.

**What to record.** Time to first correct action, every point of hesitation over
five seconds, every misreading of the legend, and every question they ask aloud.

**Output.** A fix list, and before/after screenshots of anything you change.
Those screenshots are the evidence that you iterated — put them in the write-up.

Stop Phase 1 when two consecutive participants surface no new issue.

---

## Phase 2 — Summative (n = 12–16, between-subjects)

**Design.** Two conditions, random assignment, balanced.

- **Condition A** — value-suppressing palette (toggle on)
- **Condition B** — conventional sequential palette (toggle off)

Everything else is identical. Participants see only their own condition.

**Decision task.** *"You manage a diabetes prevention budget that covers eight
districts. Using this map, choose the eight districts you would fund, and for
each one write one sentence justifying it."* Ten minutes.

**Primary measure — unsupported selections.** The count of chosen districts
whose 95% interval overlaps the national mean. These are selections the survey
does not support. Prediction: Condition A produces fewer.

**Secondary measures.**

| Measure | How |
|---|---|
| Time on task | Screen recording |
| Precision-aware justifications | Count of the eight sentences that reference sample size, confidence, or reliability. Two coders, report Cohen's κ. |
| Confidence | Single 7-point item after the task |
| Self-reported difficulty | Single 7-point item |

**Exit questions.** "What did the colours tell you?" and "Was there anything you
felt unsure about?" Asked after the task, never before.

---

## Analysis plan, fixed in advance

Write this section before collecting anything, so the analysis cannot drift
toward whatever the data happens to show.

- Primary outcome: mean unsupported selections per condition, reported with a
  95% bootstrap CI on the difference and a Cliff's delta effect size.
- **No p-values.** At n=16 the study is a pilot, and a significance test on
  eight participants per cell would be theatre. Report the effect size, the
  interval, and say plainly that the sample supports a direction, not a claim.
- Report the null honestly if it appears. "The palette made no measurable
  difference at this sample size" is a publishable sentence and a more credible
  one than a marginal effect squeezed out of twelve people.
- Pre-register the primary measure by writing it down and timestamping it — a
  commit is sufficient — before the first session.

---

## Materials checklist

- [ ] Consent form, one page, plain language
- [ ] Deployed URL, one condition pinned per participant
- [ ] Task sheet, printed
- [ ] Recording setup tested end to end
- [ ] Response spreadsheet with the coding scheme already in it
- [ ] Two coders briefed on the justification-coding rubric

---

## Reporting into the portfolio

The README's Findings section should carry three things: the effect size with
its interval, one before/after screenshot pair from Phase 1, and one sentence
naming what you would do differently with a larger sample.

That last sentence matters more than the result. It is the difference between a
student who ran a study and a student who understands what their study could and
could not establish.
