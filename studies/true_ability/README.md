# Why does choosing items lose to sampling them?

**Not shipped.** A one-time study on synthetic data, excluded from the sdist
like the rest of `studies/`. The write-up is
[`docs/validation.md` §5](../../docs/validation.md#5-choosing-items-against-sampling-them),
under *On synthetic data: it is the estimates, and more models fix them*.

`irtcheck validate` on the README's synthetic matrix had random sets beating
`select`'s at every size, on data drawn from a one-dimensional 2PL. Synthetic
data has ground truth, so each candidate cause is removed in turn by swapping
one piece of the unchanged `leave_one_model_out` loop:

| script               | swaps                                                  | answer                   |
| -------------------- | ------------------------------------------------------ | ------------------------ |
| `targets.py`         | the target: true ability, noise-free accuracy          | not the target           |
| `ability_scoring.py` | the scoring: MAP ability with the fit's item parameters | not the scoring          |
| `diagnose.py`        | the selector: `select` given the *true* `a`, `b`, theta | not the objective — the estimates |

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

`ability_scoring.py` and `diagnose.py` share holdout fits through `--cache`
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
