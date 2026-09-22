# Why does choosing items lose to sampling them?

**Not shipped.** A one-time study on synthetic data, excluded from the sdist
like the rest of `studies/`. The write-up is
[`docs/validation.md` §5](../../docs/validation.md#5-choosing-items-against-sampling-them),
under *On synthetic data: removing the candidates one at a time* and the two
subsections after it.

`irtcheck validate` on the README's synthetic matrix had random sets beating
`select`'s at every size, on data drawn from a one-dimensional 2PL. Synthetic
data has ground truth, so each candidate cause is removed in turn by swapping
one piece of the unchanged `leave_one_model_out` loop:

| script               | swaps                                                  | answer                   |
| -------------------- | ------------------------------------------------------ | ------------------------ |
| `targets.py`         | the target: true ability, noise-free accuracy          | not the target           |
| `ability_scoring.py` | the scoring: MAP ability with the fit's item parameters | not the scoring          |
| `diagnose.py`        | the selector: `select` given the *true* `a`, `b`, theta | not the objective — the estimates |

`alternatives.py` then scores two fixes §5 listed as untried, through the same
loop and the same cached fits:

| selector        | how it chooses                                                        |
| --------------- | --------------------------------------------------------------------- |
| `filter+random` | drop `dead`, `inverted`, ceiling and floor; draw the rest at random    |
| `expected info` | information averaged over the interval on `a` (Gauss-Hermite)          |
| `lower bound`   | information at the lower end of the interval on `a`                    |

Share of random draws beaten (`alternatives*.txt`, one suite per column):

| n   | 10 models: select / filter / lower | 25 models        | 50 models        | 100 models        |
| --- | ---------------------------------- | ---------------- | ---------------- | ----------------- |
| 25  | 3% / 44% / 17%                     | 38% / 52% / 53%  | 98% / 58% / 100% | 100% / 58% / 100% |
| 50  | 0% / 41% / 0%                      | 15% / 56% / 9%   | 94% / 51% / 99%  | 100% / 54% / 100% |
| 100 | 0% / 55% / 0%                      | 46% / 56% / 33%  | 95% / 54% / 88%  | 92% / 54% / 92%   |
| 200 | 0% / 68% / 1%                      | 31% / 52% / 28%  | 75% / 51% / 52%  | 86% / 55% / 95%   |
| 400 | 22% / 54% / 70%                    | 16% / 54% / 47%  | 32% / 53% / 44%  | 58% / 59% / 84%   |

- **Filter, then sample, is a random set.** It beats 41-68% of plain random
  draws at every size and model count: the filter removes too few items to
  matter, as §4 found for ceiling and floor. So the README's advice below fifty
  models — drop the flagged items, draw the rest — is sound, and costs nothing
  against random; it is just not better than random.
- **Expected information is no fix.** Within a few points of `select` up to
  n=100 at every model count, and worse at n=200-400 from fifty models (47%
  against 75%, 36% against 58%). Averaging `a^2` over the interval adds `se^2`,
  which *favours* uncertain items — the opposite of what the winner's curse
  calls for. The fix §5 named is not one.
- **The lower bound is mixed.** It helps at the edges — 25 models n=25, 100
  models n=200-400 — and hurts at 25 models n=50-100. It does not rescue
  selection below fifty models. One seed per column, so the small differences
  are not established.

### Random from select's own pool, and scoring on fresh responses

`alternatives.py` also scores `usable+random`: `select`'s pool and padding
rule, drawn at random instead of by information. At 10 models it is *worse*
than a plain random set (3% of draws beaten at n=100) although its items carry
about 56% more true information per item than the suite's. At 25 models it beats
`select` at every size (81% of draws at n=25, falling to 30% at n=400); at 50
and 100 models it beats plain random in 62-90% of draws, behind `select` up to
n=100 and ahead of it at n=400.

That pointed at the loop rather than the estimates. A held-out fit chooses items
from every other model's responses, and `validate` scores those same models on
those same responses. An item looks informative when the others' answers to it
happen to line up with their order, so on chosen items the reference models are
spread out by their own noise, and the held-out model, whose answers played no
part in the choice, is pulled toward the middle. `fresh.py` removes only that:
it chooses from the original responses and scores every model on a fresh draw
from the same true parameters.

Share of random draws beaten, same responses -> fresh responses (`fresh*.txt`):

| n   | 10 models  | 25 models   | 50 models  | 100 models  |
| --- | ---------- | ----------- | ---------- | ----------- |
| 25  | 3% -> 94%  | 38% -> 100% | 98% -> 100% | 100% -> 100% |
| 50  | 0% -> 100% | 15% -> 92%  | 94% -> 100% | 100% -> 100% |
| 100 | 0% -> 32%  | 46% -> 100% | 95% -> 98%  | 92% -> 100%  |
| 200 | 0% -> 64%  | 31% -> 100% | 75% -> 94%  | 86% -> 100%  |
| 400 | 22% -> 80% | 16% -> 96%  | 32% -> 98%  | 58% -> 91%   |

**Most of the loss below fifty models is this, not estimation noise.** Scored
on fresh responses, `select` beats random at 25 models at every size and at 10
models up to n=50. The effect shrinks with more models (at 100 it moves
n=100-400 by 8-33 points and leaves n=25-50 at 100%), so on its own it does not
explain HELM at 95 models.

It is not only an artefact of the protocol. A user who places a new model by
its accuracy on the anchor set, against the reference models' accuracy on the
same set from the responses it was chosen from, gets the same pull toward the
middle — and with deterministic decoding there is no fresh draw to take. Not yet
tested: placing the new model by ability estimated from its anchor responses
against the reference models' *full-suite* ability, which never re-scores the
reference models on the chosen items. One suite and one fresh draw per column.

### Placing the new model by ability instead, against the fit you already have

`placement.py` follows that through. A user with a fit of the models they have
already run does not need to re-score those models on the anchor set: the fit
gives both the item parameters and each reference model's ability over the whole
suite. A new model can be placed by the ability estimated from its own answers
to the set,

    theta_hat(k) = argmax over theta of  log N(theta; 0, 1)
                   + sum over i in S of  log P(y_ki | a_i, b_i, theta)
    place(k)     = 1 + #{m != k : theta_m(fit) > theta_hat(k)}

so nothing selection touched is re-scored. Random sets are placed the same way.
Share of random sets beaten, and places off against the true ranking
(`placement*.txt`, 200 random sets per size):

| n   | 10 models   | 25 models   | 50 models   | 100 models  |
| --- | ----------- | ----------- | ----------- | ----------- |
| 25  | 0.50 · 81%  | 2.52 · 77%  | 3.08 · 100% | 5.04 · 100% |
| 50  | 0.30 · 73%  | 1.96 · 71%  | 2.88 · 93%  | 4.08 · 100% |
| 100 | 0.30 · 41%  | 1.40 · 76%  | 2.08 · 95%  | 3.64 · 99%  |
| 200 | 0.20 · 31%  | 1.08 · 80%  | 1.80 · 75%  | 3.00 · 97%  |
| 400 | 0.10 · 56%  | 0.88 · 81%  | 1.52 · 51%  | 2.64 · 83%  |

**Choosing wins at every model count, including ten.** Against `validate`'s
3%, 0%, 0% at 10 models and 38%, 15%, 46% at 25, this is 81%, 73%, 41% and
77%, 71%, 76%. Placement is also more accurate in absolute terms for every
selector, random included, because the reference models keep their full-suite
places instead of being re-scored on a short set.

So "below fifty models, do not choose, sample" is a statement about how
`validate` scores an anchor set, not about selection. What it costs to choose
on noisy estimates is real but much smaller than the protocol's own effect.
`usable+random` behaves differently again: best of all at 10 and 25 models,
behind `select` at 50 and 100.

Caveats: one suite per model count; the item parameters used to estimate the
new model's ability still come from the same fit that chose the set, which no
synthetic control here removes; and `validate` does not offer this placement,
so nothing in the shipped tool measures it yet.

`diagnose.py` also reports what the chosen sets contain — true and fitted `a`,
the spread of true difficulty, how many models answer every item right or
every item wrong, and true test information — and, with `--holdouts`, runs the
model-count sweep.

    # README matrix: 10 models x 2 variants, 600 items, seed 5
    PYTHONPATH=src .venv/bin/python studies/true_ability/targets.py \
        --models 10 --variants 2 --items 600 --seed 5 > studies/true_ability/targets10x2.txt
    PYTHONPATH=src .venv/bin/python studies/true_ability/ability_scoring.py \
        --models 10 --variants 2 --items 600 --seed 5 \
        --cache ~/.cache/irtcheck-ability/synth10x2 > studies/true_ability/ability10x2.txt
    PYTHONPATH=src .venv/bin/python studies/true_ability/diagnose.py \
        --models 10 --variants 2 --items 600 --seed 5 \
        --cache ~/.cache/irtcheck-ability/synth10x2 > studies/true_ability/diagnose10x2.txt

    # 25 models, 800 items, seed 0: the same three with --models 25 --items 800 --seed 0
    # and --cache ~/.cache/irtcheck-ability/synth25, into *25.txt

    # the sweep: 25 holdouts spread across the ability range, four fits at a time
    PYTHONPATH=src OMP_NUM_THREADS=2 .venv/bin/python studies/true_ability/diagnose.py \
        --models 50 --items 800 --seed 0 --holdouts 25 --workers 4 \
        --cache ~/.cache/irtcheck-ability/synth50 > studies/true_ability/diagnose50.txt
    # and --models 100 with --cache ~/.cache/irtcheck-ability/synth100 into diagnose100.txt

    # the untried fixes, and choosing vs scoring on fresh responses: same arguments and
    # --cache as diagnose.py, into alternatives*.txt and fresh*.txt (fresh.py needs no --workers)
    PYTHONPATH=src .venv/bin/python studies/true_ability/alternatives.py \
        --models 10 --variants 2 --items 600 --seed 5 \
        --cache ~/.cache/irtcheck-ability/synth10x2 > studies/true_ability/alternatives10x2.txt

`ability_scoring.py`, `diagnose.py` and `alternatives.py` share holdout fits through `--cache`
and fit any that are missing; `targets.py` fits in-process and caches nothing.
A 600-800 item holdout fit takes tens of seconds on CPU.

## Caveats

- One seed per configuration, and each model count draws its own suite, so
  the columns of the sweep are not the same 800 items.
- No point between 25 and 50 models. "About fifty" is the resolution.
- The 50- and 100-model runs hold out 25 models; every model is still scored
  on every set.
- Holdout fits are not bit-reproducible across machines or thread counts; the
  committed outputs come from one set of cached fits.
