"""Reconcile every estimator's conventions, then measure and report agreement.

    python crosscheck/compare.py

Reads whatever `est-*.json` files the runners left in `crosscheck/work/<dataset>/`,
so it degrades honestly: a reference that did not run is listed as MISSING and
never silently skipped.

## Guarding against a cross-check that checks nothing

A comparison harness has a failure mode worse than reporting a bad number,
which is reporting a good one because it compared nothing — an empty array
correlating with an empty array, a conversion that returned its input, a
reference whose fit never ran. That would let the README claim a validation we
do not have.

Three defences, deliberately of different kinds:

1. **Structural.** No agreement figure is printed until it has passed
   `assert_comparable`, which refuses empty, constant, all-zero, misaligned and
   bit-identical comparisons rather than correlating them and printing a number.

2. **A shuffled control on every figure.** Each comparison is also run against
   32 permutations of its own reference, and the mean and worst |r| of those are
   printed beside the real one. A correlation of 0.98 means nothing on its own;
   it means something next to a control of 0.02. Where the real figure fails to
   beat its own control the line is marked NO SIGNAL — which is a statement
   about those two estimators, not about the harness. See `Checked.beats_control`
   for why that distinction is kept.

3. **A canary with a known answer.** `mirt` and `mml-scipy-mirtquad` are two
   independent implementations of the same estimator on the same grid, so they
   must agree to near machine precision. `main()` asserts they still do and
   fails the run if they do not. This is the defence that catches the failure
   the other two cannot: a harness that has silently stopped comparing anything
   would report agreement everywhere, and the only way to notice is to have one
   comparison whose correct answer was known before the run started.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from crosscheck.common import (
    DATASETS,
    Estimates,
    estimates_path,
    load_truth,
    work_dir,
    write_json,
)
from crosscheck.conventions import (
    B_IDENTIFIED_MIN_A,
    Agreement,
    agreement,
    align,
    probabilities,
    reflection_agrees,
    standardise_theta_metric,
)

# Every estimator this report knows how to read, in the order it prints them.
ESTIMATORS = (
    "mirt",
    "mml-scipy",
    "mml-scipy-mirtquad",
    "pyirt-vague",
    "pyirt-hierarchical",
    "ours-hierarchical",
    "ours-vague",
)
# Our default fit is what the tool ships, so it is the subject of every
# comparison; mirt is the reference it is compared against.
SUBJECT = "ours-hierarchical"
REFERENCE = "mirt"

MIN_COMPARABLE_N = 10

# The canary. `mml-scipy-mirtquad` and `mirt` are two independent
# implementations of one estimator on one grid, so their agreement is known in
# advance to be near-exact. If it ever is not, either a reference stopped
# running or the harness stopped comparing, and the whole report is void — so
# the run fails rather than printing numbers nobody should trust.
CANARY_PAIR = ("mml-scipy-mirtquad", "mirt")
CANARY_MIN_PEARSON = 0.999
CANARY_MAX_CONTROL = 0.60


class VacuousComparison(AssertionError):
    """Raised when a comparison would report agreement without comparing anything."""


def assert_comparable(name: str, ref: np.ndarray, other: np.ndarray) -> int:
    """Refuse to report on a comparison that is not actually comparing.

    Six ways a silent no-op gets through, each of which would otherwise print
    as a clean correlation or a benign NaN:

    1. the vectors are different lengths — alignment failed
    2. too few positions are finite on both sides — usually a failed id match
    3. one side is constant — `corrcoef` returns NaN and a reader skims past it
    4. the two sides are bit-identical — a conversion that returned its input,
       or the same file compared with itself
    5. either side is all zeros — a fit that never ran and wrote its initial values
    6. either side is non-finite throughout
    """
    if ref.shape != other.shape:
        raise VacuousComparison(f"{name}: shapes {ref.shape} vs {other.shape} — alignment failed")
    ok = np.isfinite(ref) & np.isfinite(other)
    n = int(ok.sum())
    if n < MIN_COMPARABLE_N:
        raise VacuousComparison(
            f"{name}: only {n} positions finite on both sides (need {MIN_COMPARABLE_N}); "
            "the item ids probably did not match"
        )
    r, o = ref[ok], other[ok]
    for label, vec in (("reference", r), ("subject", o)):
        if np.allclose(vec, vec[0]):
            raise VacuousComparison(f"{name}: the {label} is constant at {vec[0]:.4g}")
        if np.allclose(vec, 0.0):
            raise VacuousComparison(f"{name}: the {label} is all zeros — did that fit run?")
    if np.array_equal(r, o):
        raise VacuousComparison(
            f"{name}: the two sides are bit-identical — a conversion returned its input, "
            "or one file is being compared with itself"
        )
    return n


# The shuffled control is averaged over this many permutations. One permutation
# is far too noisy to interpret at n=12 — a single draw can correlate at 0.4 by
# chance, which would either raise a false vacuity alarm or hide a real one.
N_SHUFFLES = 32


@dataclass(slots=True)
class Checked:
    """An agreement figure together with the control that says whether it means anything."""

    label: str
    real: Agreement
    shuffled_pearson: float
    shuffled_worst: float

    @property
    def beats_control(self) -> bool:
        """Does the real comparison outscore every permutation of its reference?

        The scale-free reading of the control. If it does not, this comparison
        carries no information about which item is which — and that is reported
        as **no signal** rather than as vacuity, because the two are different
        claims and conflating them would be its own dishonesty:

          - *no signal* is a fact about these two estimators. py-irt's `a` on
            the sparse dataset genuinely does not track mirt's (r=+0.09), because
            its 'vague' prior crushed every slope to about 1. That is a real
            finding, and calling it a broken comparison would bury it.
          - *vacuity* is a fault in the harness, and it is caught structurally
            by `assert_comparable` plus the canary in `main()`, which knows one
            pair of estimators that must agree almost exactly and fails the run
            if they stop doing so.
        """
        return abs(self.real.pearson) > self.shuffled_worst

    def line(self) -> str:
        flag = "" if self.beats_control else "  << NO SIGNAL (does not beat its own shuffled control)"
        return (
            f"{self.real.line(self.label)}  [shuffled control: mean|r|="
            f"{self.shuffled_pearson:.3f} worst={self.shuffled_worst:.3f}]{flag}"
        )


def checked_agreement(
    label: str, ref: np.ndarray, other: np.ndarray, *, through_origin: bool = False, seed: int = 7
) -> Checked:
    """`agreement`, plus the same comparison against permuted references.

    The control is the point. A correlation of 0.98 against the reference means
    nothing on its own — it has to be paired with a correlation near zero
    against shuffles of that same reference, or the harness has not demonstrated
    it is sensitive to which item is which. Reported as the mean and the worst
    |r| over `N_SHUFFLES` permutations.
    """
    assert_comparable(label, ref, other)
    real = agreement(ref, other, through_origin=through_origin)
    rng = np.random.default_rng(seed)
    controls = [
        abs(agreement(rng.permutation(ref), other, through_origin=through_origin).pearson)
        for _ in range(N_SHUFFLES)
    ]
    controls = [c for c in controls if np.isfinite(c)]
    return Checked(
        label,
        real,
        float(np.mean(controls)) if controls else float("nan"),
        float(np.max(controls)) if controls else float("nan"),
    )


def load_available(dataset: str) -> dict[str, Estimates]:
    found = {}
    for name in ESTIMATORS:
        path = estimates_path(dataset, name)
        if path.exists():
            found[name] = Estimates.from_json(path)
    return found


def reconcile(est: Estimates, reference_ids: list[str], reference_theta_ids: list[str]) -> dict:
    """Put one estimator's numbers onto the reference's item order and metric.

    Returns the aligned (a, b, theta) plus a record of every transformation
    applied, so the report can state what was done rather than leaving a reader
    to trust that something was.
    """
    a, b = align(reference_ids, est.item_ids, est.a, est.b)
    (theta,) = align(reference_theta_ids, est.respondent_ids, est.theta)

    notes: list[str] = []
    scale, loc = 1.0, 0.0
    if "NOT identified" in est.theta_metric:
        # py-irt's hierarchical model learns both the mean and the variance of
        # the ability distribution, so its output is on no particular scale
        # until we put it on one.
        a, b, theta, scale, loc = standardise_theta_metric(a, b, theta)
        notes.append(
            f"theta metric was free; standardised with c=sd(theta)={scale:.4f}, "
            f"d=mean(theta)={loc:+.4f} (a*=c, b=(b-d)/c, theta=(theta-d)/c)"
        )
    if "b = -d/a" in est.parameterisation:
        notes.append("slope-intercept (a, d) converted to (a, b) with b = -d/a")
    return {"a": a, "b": b, "theta": theta, "scale": scale, "loc": loc, "notes": notes}


def report_dataset(dataset: str, out: list[str]) -> dict:
    spec = DATASETS[dataset]
    truth = load_truth(dataset)
    available = load_available(dataset)

    def say(line: str = "") -> None:
        out.append(line)
        print(line)

    say("=" * 118)
    say(f"DATASET {dataset}: {spec['n_models']} respondents x {spec['n_items']} items, "
        f"{spec['missing']:.0%} missing")
    say(f"  why: {spec['why']}")
    say("=" * 118)

    say("\nestimators:")
    for name in ESTIMATORS:
        if name not in available:
            say(f"  {name:<22} MISSING — did not run")
            continue
        est = available[name]
        detail = est.detail
        bits = [detail.get("estimator", "?")]
        if "converged" in detail:
            bits.append(f"converged={detail['converged']}")
        if "log_lik" in detail:
            bits.append(f"logLik={detail['log_lik']:.4f}")
        elif "logLik" in detail:
            bits.append(f"logLik={float(detail['logLik']):.4f}")
        if "max_abs_free_gradient" in detail:
            bits.append(f"max|free grad|={detail['max_abs_free_gradient']:.1e}")
        say(f"  {name:<22} {'; '.join(str(b) for b in bits)}")
        say(f"  {'':<22} a>0 constrained: {est.a_positive};  theta metric: {est.theta_metric}")

    if REFERENCE not in available or SUBJECT not in available:
        say(f"\ncannot compare: need both {REFERENCE} and {SUBJECT}")
        return {}

    reference = available[REFERENCE]
    # Everything is aligned onto the items the reference actually fitted, so a
    # comparison never silently includes an item one side declined to estimate.
    ref_ids = list(reference.item_ids)
    ref_theta_ids = list(reference.respondent_ids)
    say(f"\nall comparisons are on the {len(ref_ids)} items {REFERENCE} fitted "
        f"(of {spec['n_items']}); it drops single-category items and so does mml-scipy")

    reconciled = {
        name: reconcile(est, ref_ids, ref_theta_ids) for name, est in available.items()
    }
    (truth_a, truth_b) = align(ref_ids, truth["item_ids"], truth["a"], truth["b"])
    (truth_theta,) = align(ref_theta_ids, truth["respondent_ids"], truth["theta"])

    say("\nconventions applied:")
    for name in ESTIMATORS:
        if name in reconciled:
            notes = reconciled[name]["notes"] or ["none needed — already a*(theta-b) on theta~N(0,1)"]
            for note in notes:
                say(f"  {name:<22} {note}")

    say("\nreflection check (sign of corr(theta) against the reference):")
    for name in ESTIMATORS:
        if name == REFERENCE or name not in reconciled:
            continue
        agrees = reflection_agrees(reconciled[REFERENCE]["theta"], reconciled[name]["theta"])
        say(f"  {name:<22} {'same mode' if agrees else 'MIRRORED — (a,b,theta) -> (-a,-b,-theta)'}")

    results: dict = {"dataset": dataset, "n_items_compared": len(ref_ids), "comparisons": {}}

    # The b comparison is restricted to items whose slope the reference pins
    # down. See conventions.B_IDENTIFIED_MIN_A: `b` enters the likelihood only
    # through a*(theta-b), so on a near-dead item it is not a quantity any
    # estimator can recover and comparing it measures priors.
    identified = np.isfinite(reconciled[REFERENCE]["a"]) & (
        np.abs(reconciled[REFERENCE]["a"]) >= B_IDENTIFIED_MIN_A
    )
    say(f"\n`b` is compared only on the {int(identified.sum())} items where "
        f"|{REFERENCE} a| >= {B_IDENTIFIED_MIN_A} (difficulty is not identified when a ~ 0)")

    for kind, getter, origin in (
        ("a", lambda r: r["a"], True),
        ("b", lambda r: np.where(identified, r["b"], np.nan), False),
        ("theta", lambda r: r["theta"], False),
    ):
        say(f"\n--- {kind} vs {REFERENCE} " + "-" * 80)
        for name in ESTIMATORS:
            if name == REFERENCE or name not in reconciled:
                continue
            try:
                checked = checked_agreement(
                    f"{name}", getter(reconciled[REFERENCE]), getter(reconciled[name]),
                    through_origin=origin,
                )
            except VacuousComparison as exc:
                say(f"  SKIPPED {exc}")
                continue
            say("  " + checked.line())
            results["comparisons"][f"{kind}:{name}-vs-{REFERENCE}"] = {
                "n": checked.real.n,
                "pearson": checked.real.pearson,
                "spearman": checked.real.spearman,
                "slope": checked.real.slope,
                "rmse": checked.real.rmse,
                "max_abs_diff": checked.real.max_abs_diff,
                "rmse_after_slope": checked.real.rmse_after_slope,
                "shuffled_pearson_mean": checked.shuffled_pearson,
                "shuffled_pearson_worst": checked.shuffled_worst,
                "beats_shuffled_control": checked.beats_control,
            }

        say(f"\n  ... and against the generating truth ({kind}):")
        truth_vec = {"a": truth_a, "b": np.where(identified, truth_b, np.nan), "theta": truth_theta}[kind]
        for name in ESTIMATORS:
            if name not in reconciled:
                continue
            try:
                checked = checked_agreement(
                    f"{name} vs truth", truth_vec, getter(reconciled[name]), through_origin=origin
                )
            except VacuousComparison as exc:
                say(f"  SKIPPED {exc}")
                continue
            say("  " + checked.line())
            results["comparisons"][f"{kind}:{name}-vs-truth"] = {
                "n": checked.real.n,
                "pearson": checked.real.pearson,
                "spearman": checked.real.spearman,
                "slope": checked.real.slope,
                "rmse": checked.real.rmse,
                "shuffled_pearson_mean": checked.shuffled_pearson,
                "shuffled_pearson_worst": checked.shuffled_worst,
                "beats_shuffled_control": checked.beats_control,
            }

    results["probability_surface"] = report_probability_surface(dataset, reconciled, say)
    results["mml_vs_mirt"] = report_mml_vs_mirt(available, say)
    results["pyirt_median_vs_mean"] = report_pyirt_export(available, say)
    results["both_converged"] = report_convergence(available, say)
    return results


def report_convergence(available: dict, say) -> bool:
    """Did the two unpenalised reference implementations converge at all?

    Reported as a first-class result rather than a footnote, because when the
    answer is no it is the strongest empirical argument in this directory for
    why irtcheck uses priors. An estimator that does not converge is not a
    stricter standard than a penalised one; it has no answer to be strict about.
    """
    subject, reference = CANARY_PAIR
    say("\n--- did the unpenalised references converge? " + "-" * 50)
    flags: list[bool] = []
    for name in (reference, subject, "mml-scipy"):
        if name not in available:
            continue
        detail = available[name].detail
        raw = detail.get("converged")
        ok = str(raw).lower() == "true"
        extra = ""
        if "n_params_at_bound" in detail:
            extra = f", {detail['n_params_at_bound']} parameter(s) pinned at a bound"
        if "iterations" in detail:
            extra += f", {detail['iterations']} iterations"
        say(f"  {name:<22} converged={raw}{extra}")
        if name in (reference, subject):
            flags.append(ok)

    both = all(flags) and len(flags) == 2
    if not both:
        say("  => the unpenalised marginal likelihood has NO interior maximum on this matrix.")
        say("     Slopes diverge, the EM stops on its iteration cap, and two correct")
        say("     implementations of the same estimator settle in different places along a")
        say("     ridge. This is not a bug in either: it is what maximum likelihood does with")
        say("     12 respondents, and it is the case for estimating this model with priors.")
    return both


def report_probability_surface(dataset: str, reconciled: dict, say) -> dict:
    """Compare fits by what they predict, not by how they are written down.

    The invariance-proof comparison. Two parameter sets can differ by any
    reflection or rescaling and still be the same model; the response
    probabilities they imply cannot. So when reconciled parameters disagree,
    this says whether the fits are one model described two ways or genuinely
    two models — and it is the number to trust when the b comparison has been
    restricted and the a comparison has a slope in it.
    """
    say("\n--- implied P(correct) surface, mean |difference| against " + REFERENCE + " " + "-" * 30)
    ref = reconciled[REFERENCE]
    ok_items = np.isfinite(ref["a"]) & np.isfinite(ref["b"])
    out = {}
    for name in ESTIMATORS:
        if name == REFERENCE or name not in reconciled:
            continue
        est = reconciled[name]
        usable = ok_items & np.isfinite(est["a"]) & np.isfinite(est["b"])
        if usable.sum() < MIN_COMPARABLE_N or not np.isfinite(est["theta"]).all():
            say(f"  {name:<22} SKIPPED — {int(usable.sum())} usable items")
            continue
        mine = probabilities(est["a"][usable], est["b"][usable], est["theta"])
        theirs = probabilities(ref["a"][usable], ref["b"][usable], ref["theta"])
        diff = np.abs(mine - theirs)
        say(f"  {name:<22} mean|dP|={diff.mean():.4f}  median|dP|={np.median(diff):.4f}  "
            f"max|dP|={diff.max():.4f}  (over {usable.sum()} items x {mine.shape[0]} respondents)")
        out[name] = {
            "mean_abs_dp": float(diff.mean()),
            "median_abs_dp": float(np.median(diff)),
            "max_abs_dp": float(diff.max()),
        }
    return out


def report_mml_vs_mirt(available: dict, say) -> dict:
    """The check that makes the scipy reference trustworthy.

    mirt and `mml-scipy-mirtquad` are two independent implementations of one
    estimator — same objective, same quadrature rule, different optimisers (EM
    versus quasi-Newton) in different languages. If their log-likelihoods agree
    to several decimals then neither has a parameterisation or convergence bug,
    and the scipy one can stand in as the reference on a machine with no R.
    Without this line that module is just a fourth opinion.
    """
    say("\n--- mirt vs the independent scipy MML (same objective, same grid) " + "-" * 30)
    out: dict = {}
    if "mirt" not in available:
        say("  mirt MISSING — the scipy MML stands alone and is unvalidated against R")
        return out
    mirt_ll = float(available["mirt"].detail["logLik"])
    say(f"  mirt logLik                        {mirt_ll:.6f}")
    for name in ("mml-scipy-mirtquad", "mml-scipy"):
        if name not in available:
            continue
        ll = float(available[name].detail["log_lik"])
        rule = available[name].detail["quadrature_rule"]
        say(f"  {name:<34} {ll:.6f}   (rule={rule}, delta={ll - mirt_ll:+.6f})")
        out[name] = {"log_lik": ll, "rule": rule, "delta_vs_mirt": ll - mirt_ll}
    say("  the matching rule is the comparable one; Gauss-Hermite optimises a")
    say("  slightly different approximation to the same integral.")
    return out


def report_pyirt_export(available: dict, say) -> dict:
    """How much py-irt's exported discrimination understates its own posterior.

    `py_irt`'s `export()` returns `exp(loc_slope)` — the median of its fitted
    LogNormal, not the mean, which is `exp(loc + sigma^2/2)`. A user comparing
    py-irt's `disc` against a posterior mean from anywhere else is comparing two
    different summaries of the same distribution, and this says by how much.
    """
    out: dict = {}
    rows = [(n, e) for n, e in available.items() if n.startswith("pyirt-")]
    if not rows:
        return out
    say("\n--- py-irt's exported `disc` is a LogNormal MEDIAN, not a mean " + "-" * 30)
    for name, est in rows:
        ratio = est.detail["median_to_mean_ratio_max"]
        say(f"  {name:<22} posterior mean / reported median, worst item: {ratio:.4f}x")
        out[name] = {"max_mean_over_median": ratio}
    return out


def main() -> int:
    lines: list[str] = []
    summary = {}
    for dataset in DATASETS:
        if not (work_dir(dataset) / "truth.json").exists():
            print(f"{dataset}: no inputs; run `python crosscheck/export_inputs.py` first")
            continue
        summary[dataset] = report_dataset(dataset, lines)
        lines.append("")
        print()

    RESULTS_DIR = Path(__file__).with_name("results")
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "comparison.txt").write_text("\n".join(lines) + "\n")
    write_json(RESULTS_DIR / "comparison.json", summary)
    print(f"wrote {RESULTS_DIR / 'comparison.txt'} and comparison.json")

    return check_canary(summary)


def check_canary(summary: dict) -> int:
    """Fail the run unless the one comparison with a known answer still has it.

    `mirt` and `mml-scipy-mirtquad` maximise the same likelihood over the same
    quadrature grid, in different languages with different optimisers. Where
    that maximum exists they must agree to near machine precision, and if they
    do not then one of three things happened — a reference silently failed to
    run, the reconciliation broke, or the comparison stopped reading its inputs
    — and every other figure in the report is void.

    This is the only check here that could catch a harness reporting agreement
    it never measured, because it is the only one whose right answer was known
    before the run started.

    **Where that maximum exists** is the whole qualification, and it is checked
    rather than assumed. On a matrix with 12 respondents the unpenalised
    likelihood has no interior maximum: several slopes diverge, mirt stops on
    its iteration cap without converging, and the two implementations settle in
    different places along a ridge. Requiring them to agree there would be
    requiring agreement about a quantity that does not exist, so a dataset where
    either implementation reports non-convergence is **excluded from the canary
    and reported as a finding instead**. At least one dataset must still
    qualify, or the report has nothing validating it and says so.
    """
    subject, reference = CANARY_PAIR
    print("\n" + "=" * 118)
    print(f"CANARY: {subject} vs {reference} — two implementations of one estimator, "
          "so where the maximum exists they must agree near-exactly")
    checked_any = False
    failures: list[str] = []
    for dataset, result in summary.items():
        converged = result.get("both_converged")
        if converged is False:
            print(f"  {dataset:<7} EXCLUDED — the unpenalised maximum does not exist here "
                  "(see the divergence finding above); nothing to be exact about")
            continue
        for kind in ("a", "b", "theta"):
            key = f"{kind}:{subject}-vs-{reference}"
            entry = result.get("comparisons", {}).get(key)
            if entry is None:
                failures.append(f"{dataset} {key}: MISSING — the comparison never ran")
                continue
            checked_any = True
            ok = (
                entry["pearson"] >= CANARY_MIN_PEARSON
                and entry["shuffled_pearson_worst"] <= CANARY_MAX_CONTROL
                and entry["beats_shuffled_control"]
            )
            print(
                f"  {dataset:<7} {kind:<6} n={entry['n']:<4} r={entry['pearson']:+.6f} "
                f"rmse={entry['rmse']:.2e}  control worst={entry['shuffled_pearson_worst']:.3f}"
                f"  {'OK' if ok else 'FAILED'}"
            )
            if not ok:
                failures.append(
                    f"{dataset} {key}: r={entry['pearson']:+.6f} "
                    f"(need >= {CANARY_MIN_PEARSON}), control worst="
                    f"{entry['shuffled_pearson_worst']:.3f} (need <= {CANARY_MAX_CONTROL})"
                )

    if not checked_any:
        print("\nCANARY DID NOT RUN — no comparison in this report has a known answer, so")
        print("nothing in it is evidence of anything. Do not quote these numbers.")
        return 1
    if failures:
        print("\nCANARY FAILED — the whole report is void:")
        for line in failures:
            print(f"  {line}")
        return 1

    no_signal = [
        key
        for ds in summary.values()
        for key, value in ds.get("comparisons", {}).items()
        if not value.get("beats_shuffled_control", True)
    ]
    if no_signal:
        print(f"\n{len(no_signal)} comparison(s) carry no signal — a finding about those")
        print("estimators, not a harness fault (the canary passed). Listed in comparison.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
