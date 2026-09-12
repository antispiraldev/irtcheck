# Cross-check: irtcheck's 2PL against `mirt`, `py-irt`, and a from-scratch MML

**Not shipped.** This directory is excluded from the wheel and the sdist and must
stay that way — see `pyproject.toml`'s `[tool.hatch.build.targets.sdist] exclude`,
and CI's `package` job, which asserts it. It is a one-time validation exercise,
run by hand, and it is the answer to "how do you know your numbers are right?"

Everything below was measured on this machine on 2026-09-12. Where a check did
not run, it says so.

    bash crosscheck/setup.sh     # installs both references; needs no root
    bash crosscheck/run_all.sh   # ~15 minutes

---

## Which references actually ran

**Both did, plus two of our own.** Nothing here is simulated or asserted from
documentation.

| Reference | Version | Ran? | How it was obtained |
|---|---|---|---|
| R + `mirt` | R 4.3.3, mirt 1.41 | **yes** | conda-forge via micromamba, no root |
| `py-irt` | 0.7.1 on CPython 3.11.14 | **yes** | `uv python install 3.11`, then a venv, no root |
| scipy marginal ML | `run_mml.py`, written here | **yes** | project dependencies only |
| grid posterior | `run_gridpost.py`, written here | **yes** | project dependencies only |

Neither reference was installed on this machine at the start. `R` was absent and
`apt` needs root; `python3.11` was absent and only 3.12 was present. Both were
installed into `~/.cache/irtcheck-crosscheck/` without root, and `setup.sh`
reproduces exactly that. So the fallback the brief allowed for — harness only,
plus a self-check — was not needed.

## The datasets

Both from `irtcheck.synth`, the only place ground truth exists in this repo.

- **`dense`** — 300 respondents x 60 items, complete. Every item parameter is
  identified, so a disagreement between three independent implementations is a
  bug in one of them. This is the bug-detection regime.
- **`sparse`** — 12 respondents x 200 items, 10% missing. irtcheck's actual
  regime. Divergence is expected here, and the interesting question is not
  agreement but whether the intervals are honest.

---

## Conventions, reconciled

This is what the exercise is for. Four estimators, four different ways of
writing down the same model.

| | parameterisation | `a` constrained > 0? | theta metric |
|---|---|---|---|
| **ours** | `a*(theta - b)` | no | `N(0,1)` **fixed** |
| **mirt** | `a1*theta + d` | **no** | `N(0,1)` fixed, 61 equally spaced nodes on [-6, 6] |
| **py-irt, vague** | `a*(theta - b)` | **yes** (LogNormal) | `N(0,1)` fixed |
| **py-irt, hierarchical** | `a*(theta - b)` | **yes** (LogNormal) | `N(mu, 1/u)`, **both learned — not identified** |

Four things had to be handled, and each is a place a careless comparison would
have reported a disagreement that does not exist:

1. **mirt's slope-intercept form.** `d` is an intercept, not a difficulty;
   `b = -d/a1`. Verified against mirt as its own witness: our conversion
   reproduces mirt's `IRTpars=TRUE` output to **1.5e-14**. `run_mirt.py` treats a
   failure here as fatal, because everything downstream would be meaningless.

2. **py-irt's free theta metric** under hierarchical priors. Its fitted
   `sd(theta)` came out at **1.955** (dense) and **2.443** (sparse) rather than
   1, confirming the scale is genuinely unidentified. Standardising with
   `a' = a*c`, `b' = (b-d)/c`, `theta' = (theta-d)/c` is a likelihood-preserving
   identity, and `test_conventions.py` checks that against a probability surface
   that must not move.

3. **py-irt's exported `disc` is a LogNormal median, not a mean.** `export()`
   returns `exp(loc_slope)`; the posterior mean is `exp(loc + sigma^2/2)`. On
   the sparse dataset the two differ by up to **3.74x** on the worst item. A
   comparison treating py-irt's `disc` as a posterior mean would report a
   discrepancy that is entirely a choice of summary statistic.

4. **Reflection.** All six non-reference fits landed in the same mode as mirt on
   both datasets — checked on the sign of `corr(theta)`, not on `a`, because two
   of the four constrain `a` positive and would agree by construction. Our
   fitter's own `reflected` diagnostic was `False` on all four runs, so
   `canonical_sign()` never had to intervene.

**Correction to a note in this repo:** `run_mirt.py` originally recorded mirt as
constraining `a > 0`. It does not. mirt's default 2PL puts no positivity
constraint on the slope and returned negative ones here — 2 of 59 items on
dense, 22 of 183 on sparse. Only py-irt constrains `a` positive. This matters
because it makes mirt, not py-irt, the estimator whose identification convention
matches ours.

---

## The reference validates itself

Before comparing anything to us: `mirt` and `run_mml.py` are two independent
implementations of the same estimator — the same marginal likelihood over the
same quadrature grid — in different languages with different optimisers (EM vs
L-BFGS-B). On `dense`:

```
mirt logLik                        -9431.926306
mml-scipy-mirtquad                 -9431.926306   (delta = +0.000000)
```

and on parameters, `r = 1.000000` for `a`, `b` and `theta`, with
`rmse = 1.5e-05`, `1.8e-04`, `7.6e-07` respectively. Two implementations
agreeing to six decimals is what makes either one usable as a reference, and it
means the scipy version can stand in on a machine with no R.

Getting there turned up two numerics findings worth recording:

- **mirt's "quadrature" is a rectangular rule, not a Gauss rule.** Despite
  `quadpts = 61`, mirt lays its nodes **equally spaced on [-6, 6]** with
  normal-density weights renormalised to sum to 1. Read off a fitted object
  (`fit@Model$Theta`), not inferred. Gauss-Hermite on the same problem gives a
  log-likelihood differing by 0.018 on `dense` and by ~70 on `sparse`, where the
  blown-up slopes push mass to where the two node sets diverge. Both rules are
  run so that "we found different optima" and "we optimised slightly different
  objectives" can be told apart.

- **mirt's slope-intercept form is numerics, not notation.** Our MML first
  landed 0.28 log-likelihood *below* mirt and would not improve. The cause was
  one near-dead item: when `a -> 0` the likelihood depends on `a` and `b` only
  through their product, so the `(a, b)` surface has a flat valley — `b` slid
  down it into a box constraint at -25 while `a` crept toward 0, and L-BFGS-B
  correctly reported convergence because a bound was active. The two fits were
  never in disagreement: mirt had `(a=-0.081, b=+3.324)` and ours
  `(a=+0.011, b=-25.0)`, whose intercepts `-a*b` are +0.2687 and +0.2649. Same
  model, split differently along a direction the data do not constrain.
  Refitting in `(a, d)` — mirt's own choice — removed the valley entirely: the
  same problem now converges in **2 iterations** with no parameter at a bound.

---

## Agreement with our fitter, conventions reconciled

`dense`, against mirt, on the 59 items mirt fitted. `b` is compared only on the
50 items with `|a_mirt| >= 0.35`, because difficulty is not identified when the
slope is near zero.

| | `a` r | `a` rho | `a` slope | `b` rho | `b` slope | theta r | mean \|dP\| |
|---|---|---|---|---|---|---|---|
| `ours-hierarchical` | +0.962 | +0.949 | **0.926** | +0.9991 | **0.690** | +0.9993 | **0.0092** |
| `ours-vague` | +0.986 | +0.969 | 0.972 | +0.9997 | 0.782 | +0.9996 | 0.0070 |
| `pyirt-hierarchical` | +0.942 | +0.933 | 0.899 | +0.9975 | 0.707 | +0.9979 | 0.0160 |
| `pyirt-vague` | +0.682 | +0.756 | 0.818 | +0.9800 | **0.202** | +0.9733 | 0.1111 |

**The tolerance actually achieved** — stated after the fact, not chosen in
advance. Our default fit against mirt on `dense`:

- **rank order**: Spearman **0.949** on discrimination, **0.9991** on difficulty.
  This is what the tool acts on: `report` ranks by information and `select`
  takes a top slice.
- **levels**: `a` agrees to a regression slope of **0.926**, `b` to **0.690** —
  i.e. our estimates are shrunk toward zero and toward the difficulty mean
  respectively. Residual scatter after removing that slope is 0.18 on `a` and
  0.47 on `b`.
- **predictions**: the invariance-free comparison. Mean absolute difference in
  implied P(correct) over 59 items x 300 respondents is **0.0092**, median
  0.0067. Under 1 percentage point on average.

The slopes below 1 are the documented shrinkage from the hierarchical prior, and
`ours-vague` moving them to 0.972 and 0.782 is that prior being switched off. It
is doing what `fit/model.py` says it does. Against the *generating truth* on
`dense`, shrinkage helps: our `b` recovers truth at rmse **0.343** against
mirt's **0.713**.

`pyirt-vague`'s `b` slope of **0.202** is worth a note: py-irt's "vague" priors
are `b ~ N(0, 0.1)` and `a ~ LogNormal(0, 0.1)` — not vague but very tight. Its
fitted `b` spans [-1.01, +1.01] on dense and [-0.16, +0.18] on sparse against a
true range of about [-4.7, +4.6]. The rank order survives; the scale does not.

---

## The two findings I would want to know about

### 1. Unpenalised maximum likelihood has no answer at all in irtcheck's regime

On `sparse` (12 respondents), **mirt does not converge** — `converged=FALSE`
after 5000 EM cycles — and our independent MML converges only onto box
constraints, with 10-14 parameters pinned at `|a| = 100`. mirt's slopes reach
68.8. The two implementations that agree to six decimals on `dense` correlate at
only **r = 0.64** on `sparse` `a`.

This is not a bug in either. The unpenalised likelihood has no interior maximum
on such a matrix: several slopes diverge, and two correct implementations settle
in different places along a ridge. `compare.py` detects this from the
convergence flags and **excludes `sparse` from its canary** rather than reporting
it as a harness failure, because requiring agreement about a quantity that does
not exist would be incoherent.

It is also the strongest empirical argument in this directory for the choice
`fit/model.py` makes. Against the generating truth on `sparse` `a`, our
penalised fit gets `r = 0.588, rmse = 0.60`; mirt gets `r = 0.304, rmse = 15.6`.
The priors are not a convenience, they are what makes the estimate exist.

### 2. The SVI intervals are materially too narrow, and the direction of harm is the opposite of what `fitter.py` claims

This is the finding I would most want reviewed, because it touches the tool's
central claim rather than a number in a table. `run_gridpost.py` compares our
95% interval on `a` against the same posterior computed by brute-force numerical
integration over a 481x481 grid, using the fit's own *learned* hyperparameters.

```
dense   width ratio w_svi/w_grid: median 0.558, range [0.252, 0.864]
sparse  width ratio w_svi/w_grid: median 0.847, range [0.676, 0.997]
```

**The interval the `insufficient-data` flag is computed from is roughly half the
width it should be at 300 respondents, and about 85% of it at 12.** In the
regime the tool ships for the verdict held on 12 of 12 items checked; on `dense`
it flipped on 2 of 11:

| item | true `a` | grid 95% | SVI 95% | grid verdict | SVI verdict | artifact flag |
|---|---|---|---|---|---|---|
| `item_00031` | 0.163 | [-0.341, **+0.139**] | [-0.293, **-0.017**] | insufficient-data | *confident* | `dead` |
| `item_00032` | 1.553 | [**-1.822**, +2.795] | [**+1.133**, +2.297] | insufficient-data | *confident* | `floor` |

Two things make this robust rather than an artefact:

- **The direction cannot be a conditioning artefact.** The grid posterior
  conditions on `theta = theta_hat`, which *removes* uncertainty the true
  posterior has. So the grid width is a **lower bound** on the truth and the
  measured ratio is an **upper bound** — integrating theta properly could only
  make the gap larger, never smaller.
- **The refit is verified.** The hyperparameters come from a refit at the same
  seed, whose `a` means reproduce the artifact's to **0.0e+00**, so they are the
  hyperparameters those intervals actually came from.

Now the part that is a claim about the code. `fit/fitter.py`'s module docstring
says:

> Mean-field SVI understates posterior correlations, and the honest consequence
> is that intervals here are, if anything, a little narrow — which makes
> `insufficient-data` conservative, never over-eager.

`insufficient-data` fires when the interval **spans zero**. A narrower interval
is *less* likely to span zero, so narrowness makes the flag fire **less** often
— the tool becomes **quicker** to claim an item discriminates, not slower. On a
strict reading the sentence is true (narrowness indeed never makes the flag fire
*too often*), but it presents the harmless direction as the only one, and the
measured effect runs the other way. It also partially cancels the reassurance in
`fit/model.py`, which is about the point estimate and is correct on its own
terms:

> The cost is a downward bias on `a`, and it is the right direction to be wrong
> in: it makes the tool slower to claim an item discriminates, never quicker.

The point estimate is biased toward "does not discriminate"; the interval is
biased toward "we know this confidently". Those work against each other, and
only the first is documented as a tradeoff.

Neither item above produced a *wrong conclusion* — `item_00031`'s true `a` of
0.163 is genuinely below `DEAD_THRESHOLD`, and `item_00032` genuinely
discriminates. What is wrong is the confidence, and `item_00031` is in
`usable_items()` carrying a `dead` flag that the exact posterior says is not
earned. Per CLAUDE.md, `dead` and `insufficient-data` are different claims; here
the tool made the stronger one on evidence supporting only the weaker.

**I have not patched any of this.** `src/irtcheck/fit/` belongs to another
brief. Reproduce with `python crosscheck/run_gridpost.py`; per-item numbers are
in `results/grid_posterior.json`.

---

## How this harness guards against checking nothing

A comparison that reports agreement because it compared nothing is worse than no
comparison, because it would license a claim in the README that is not true.
Three defences, deliberately of different kinds:

1. **Structural.** `compare.assert_comparable` refuses to report on a comparison
   that is empty, misaligned, constant on either side, all-zero, or
   bit-identical (the shape a no-op conversion takes). `test_conventions.py`
   feeds it each of those and asserts it raises.

2. **A shuffled control on every figure.** Each comparison is re-run against 32
   permutations of its own reference, and the mean and worst |r| print beside the
   real figure. Every correlation quoted above sits next to a control: the `a`
   comparisons' controls ran 0.02-0.22, the `b` comparisons' 0.01-0.05. Where a
   real figure fails to beat its own control the line is marked **NO SIGNAL** —
   two comparisons are (py-irt's `a` on sparse, r=+0.09), and that is reported as
   a fact about those estimators rather than a harness fault, because they are
   different claims.

3. **A canary with an answer known in advance.** `mirt` vs
   `mml-scipy-mirtquad` must agree to near machine precision where the maximum
   exists. `compare.py` asserts `r >= 0.999` and **exits non-zero** otherwise, and
   also exits non-zero if that comparison is *missing*. This is the only defence
   that catches a harness which has silently stopped comparing, since such a
   harness would report agreement everywhere and look healthy. It is itself
   tested, including the case where it cannot run.

Confirming explicitly, because it was asked: every correlation in this README was
checked to have real data on both sides (the `n` is printed on every line of
`results/comparison.txt`), and none is trivially equal — the two figures that
*are* near-identical (`mirt` vs `mml-scipy-mirtquad`) are two separately written
programs in two languages, and `assert_comparable` would have refused them had
they been the same array.

---

## Also confirmed: why py-irt is not a dependency

`bash crosscheck/check_pyirt_resolution.sh` reproduces CLAUDE.md's claim
exactly. On CPython 3.12.3:

- an unpinned `pip install py-irt` resolves to **0.1.1** (`Requires-Python >=3.6`),
  with no warning that a 2020 release was substituted;
- `py-irt==0.7.1` cannot be installed at all — every release from 0.2.1 onward
  is excluded, leaving only 0.0.1-0.1.1 as candidates;
- and 0.1.1's 2PL `fit()` ends with
  `values = ['loc_diff', 'scale_diff', 'loc_ability', 'scale_ability']` on its
  last line. No return, no read of the param store. It trains the model and
  discards every estimate.

All three as documented.

---

## Files

| | |
|---|---|
| `setup.sh` | installs both references, no root |
| `run_all.sh` | the whole thing in order |
| `common.py` | the two datasets, paths, interchange format |
| `conventions.py` | **the reconciliation logic** — reflection, scale, slope-intercept |
| `export_inputs.py` | writes the shared matrix as JSON, wide CSV and JSON Lines |
| `run_ours.py` | our 2PL, both prior families, through the shipped `fit_2pl` |
| `fit_mirt.R` / `run_mirt.py` | mirt, and the conversion checked against its own output |
| `run_pyirt.py` | py-irt 0.7.1, re-execing itself under Python 3.11 |
| `run_mml.py` | marginal ML from scratch, scipy only, two quadrature rules |
| `run_gridpost.py` | SVI intervals vs a brute-force grid posterior |
| `compare.py` | reconcile, measure, control, canary |
| `test_conventions.py` | 22 tests: the invariances, and the guards |
| `check_pyirt_resolution.sh` | reproduces the CLAUDE.md py-irt finding |
| `results/` | committed output of the run described above |

`test_conventions.py` is not in the project's `testpaths`, so it runs only when
pointed at: `.venv/bin/python -m pytest crosscheck/test_conventions.py`.
