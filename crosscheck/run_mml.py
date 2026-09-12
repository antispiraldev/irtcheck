"""A marginal-maximum-likelihood 2PL written from scratch with scipy alone.

    python crosscheck/run_mml.py

This is the third reference, and the only one that needs nothing beyond the
project's own declared dependencies. It exists for two reasons:

1. **It is the same estimator mirt computes.** mirt maximises the likelihood
   with each respondent's ability integrated out against a fixed standard
   normal, by EM over a quadrature grid. This module maximises the identical
   objective by quasi-Newton instead. Their agreement is checked in compare.py
   and it is load-bearing: without it this file is a fourth opinion rather
   than a reference, and on a machine with no R it is the *only* external
   check available.

2. **It separates our estimator from our prior.** Our fit differs from mirt in
   two ways at once — variational rather than EM, and penalised by a
   hierarchical prior rather than unpenalised. This module is unpenalised, so
   `ours-vague` against this isolates the optimiser, and this against
   `ours-hierarchical` isolates the shrinkage.

## The objective

For respondent j with observed items O_j,

    L_j = integral N(t; 0, 1) prod_{i in O_j} p_ij(t)^{x_ij} (1-p_ij(t))^{1-x_ij} dt

approximated on a fixed grid; the total log-likelihood is sum_j log L_j,
maximised over the item parameters. Ability is never a parameter, which is what
makes the estimator consistent as items grow. Gradients are analytic —
finite differences over 2 x n_items parameters would work, but would put a
step-size artefact into the one number this exercise exists to measure.

## Why it optimises (a, d) and not (a, b)

In mirt's slope-intercept form, `P = sigmoid(a*theta + d)`. This module fits
that form and converts to `b = -d/a` only to report, and that is not a
stylistic choice — it was forced by a measured failure.

Fitting (a, b) directly, this module landed 0.28 log-likelihood *below* mirt on
the dense dataset and would not improve however long it ran. The cause was one
item: item_00031, whose true discrimination is near zero. For a dead item the
likelihood depends on `a` and `b` only through the product `a*b`, so the
(a, b) surface has a perfectly flat valley — `b` slid down it until it hit the
box constraint at -25 while `a` crept toward 0, and the optimiser sat there with
a nonzero gradient in exactly one coordinate, correctly reporting convergence
because a bound was active.

The two fits were never in disagreement. At that item mirt returned
(a=-0.0808, b=+3.3237) and this returned (a=+0.0106, b=-25.0); the intercepts
`-a*b` are +0.2687 and +0.2649, agreeing to 0.004. They had found the same model
and split it differently along a direction the data do not constrain.

In (a, d) there is no flat valley — a dead item is simply a≈0 with a
well-determined d — so the problem is well conditioned and the bound is never
reached. That mirt chose this parameterisation for its own estimation is, on
this evidence, not notation but numerics.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from crosscheck.common import DATASETS, Estimates, estimates_path, load_matrix
from crosscheck.conventions import slope_intercept_to_difficulty

N_QUAD = 61
# Wide enough never to bind in the (a, d) parameterisation, and present only so
# a genuinely degenerate column cannot send the optimiser to infinity. Whether
# any parameter actually reached a bound is recorded in the output, because a
# bound that binds silently is how the (a, b) run above looked like a
# disagreement with mirt.
MAX_ABS_A = 100.0
MAX_ABS_D = 100.0

# mirt's grid, read off a fitted object's `fit@Model$Theta` rather than guessed:
# 61 equally spaced nodes on [-6, 6]. See `quadrature()`.
MIRT_GRID_LIMIT = 6.0

GAUSS_HERMITE = "gauss-hermite"
MIRT_RULE = "mirt"


def quadrature(rule: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights for integrating against a standard normal density.

    Two rules, and the difference between them is a finding rather than an
    implementation detail:

    - `gauss-hermite`: the textbook choice, and what you would write if nobody
      told you otherwise. numpy's `hermgauss` targets the weight exp(-x^2), so
      nodes scale by sqrt(2) and weights normalise by sqrt(pi) to become a
      probability measure. Its outermost nodes for n=61 reach about +/-10.5.

    - `mirt`: what mirt actually does. Despite the name "quadrature" and a
      default of 61 points, mirt lays them out **equally spaced on [-6, 6]** and
      weights them by the normal density renormalised to sum to one — a
      rectangular rule, not a Gauss rule. Read off a fitted object
      (`fit@Model$Theta`: 61 nodes, range exactly -6 to 6, spacing exactly 0.2),
      not inferred from documentation.

    On well-determined parameters the two agree to about 0.02 in total
    log-likelihood. On the sparse dataset, where unpenalised slopes blow up past
    60, they disagree by almost 70 — because the integrand then has most of its
    mass where the two node sets differ. Both runs are kept for that reason: it
    is the difference between "mirt and we found different optima" and "mirt and
    we optimised subtly different objectives".
    """
    if rule == GAUSS_HERMITE:
        nodes, weights = np.polynomial.hermite.hermgauss(n)
        return nodes * np.sqrt(2.0), weights / np.sqrt(np.pi)
    if rule == MIRT_RULE:
        nodes = np.linspace(-MIRT_GRID_LIMIT, MIRT_GRID_LIMIT, n)
        density = np.exp(-0.5 * nodes**2)
        return nodes, density / density.sum()
    raise ValueError(f"unknown quadrature rule {rule!r}")


class MarginalLikelihood:
    """The objective and its gradient in (a, d), over a respondents x items grid.

    `mask` is the observed-cell indicator; a missing cell contributes nothing to
    any respondent's integral, which is how a ragged matrix is handled without a
    special case anywhere.
    """

    def __init__(self, responses: np.ndarray, n_quad: int = N_QUAD, rule: str = GAUSS_HERMITE):
        self.x = np.nan_to_num(responses, nan=0.0)
        self.mask = np.isfinite(responses).astype(float)
        self.n_respondents, self.n_items = responses.shape
        self.rule = rule
        self.nodes, self.weights = quadrature(rule, n_quad)
        self.log_weights = np.log(self.weights)

    def unpack(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(a, d), the slope-intercept pair this module optimises."""
        return params[: self.n_items], params[self.n_items :]

    def neg_log_lik_and_grad(self, params: np.ndarray) -> tuple[float, np.ndarray]:
        a, d = self.unpack(params)
        # (Q, I): probability of a correct answer at each quadrature node.
        z = a[None, :] * self.nodes[:, None] + d[None, :]
        p = expit(z)
        log_p = -np.logaddexp(0.0, -z)
        log_q = -np.logaddexp(0.0, z)

        # (Q, J): log-likelihood of each respondent's whole response vector at
        # each node. Masked cells drop out of the sum by construction.
        per_cell = self.x * log_p[:, None, :] + (self.mask - self.x) * log_q[:, None, :]
        log_lik_qj = per_cell.sum(axis=2)

        joint = self.log_weights[:, None] + log_lik_qj
        marginal = logsumexp(joint, axis=0)  # (J,)
        total = float(marginal.sum())

        # Posterior weight of each node for each respondent — the E step.
        # Having it is what makes the gradient exact and cheap.
        post = np.exp(joint - marginal[None, :])  # (Q, J)

        # residual[q, j, i] = observed - expected, zeroed where unobserved.
        residual = (self.x - self.mask * p[:, None, :]) * post[:, :, None]
        summed = residual.sum(axis=1)  # (Q, I)

        # dz/da = theta_q, dz/dd = 1. Simpler than the (a, b) form, and without
        # its flat valley; see the module docstring.
        grad_a = (summed * self.nodes[:, None]).sum(axis=0)
        grad_d = summed.sum(axis=0)
        return -total, -np.concatenate([grad_a, grad_d])

    def classical_start(self) -> np.ndarray:
        """Start where mirt and our own fitter start: the classical estimates.

        Not a speed optimisation. The marginal likelihood of a 2PL is not
        concave, and from a=d=0 a quasi-Newton method can walk into the
        no-structure stationary point — the same failure fit/model.py's
        `classical_start` docstring records for the variational fit.
        `d = logit(p_i)` is the classical intercept, the slope-intercept
        counterpart of that function's `b = -logit(p_i)`.
        """
        counts = self.mask.sum(axis=0).clip(min=1.0)
        rate = np.clip(self.x.sum(axis=0) / counts, 0.02, 0.98)
        d0 = np.clip(np.log(rate / (1.0 - rate)), -6.0, 6.0)
        return np.concatenate([np.ones(self.n_items), d0])

    def bounds(self) -> list[tuple[float, float]]:
        return [(-MAX_ABS_A, MAX_ABS_A)] * self.n_items + [
            (-MAX_ABS_D, MAX_ABS_D)
        ] * self.n_items


def single_category_columns(responses: np.ndarray) -> np.ndarray:
    """Items whose observed responses are all 0 or all 1.

    Exactly mirt's own exclusion rule, applied here for a specific reason: it
    makes the two log-likelihoods *directly comparable numbers*. A column with
    one observed category is fitted by driving `a` to infinity, so the
    unpenalised maximum is approached and never attained, and its contribution
    depends entirely on where the optimiser's bound happened to stop it.
    Leaving such columns in would mean mirt's log-likelihood was computed over
    59 items and ours over 60, and the difference between the two would be an
    artefact of a bound rather than a check on either implementation.
    """
    observed = np.isfinite(responses)
    correct = np.nansum(np.nan_to_num(responses, nan=0.0), axis=0)
    counts = observed.sum(axis=0)
    return np.where((counts == 0) | (correct == 0) | (correct == counts))[0]


def solve(objective: MarginalLikelihood, start: np.ndarray) -> tuple:
    """L-BFGS-B, restarted from its own answer until the gradient stops falling.

    One call can stop on its line search rather than at a stationary point,
    leaving a gradient large enough to show up in the report as a disagreement
    with mirt that is really a disagreement with our own optimiser. Restarting
    resets the inverse-Hessian approximation, which is usually all it takes.

    Returns the result, the max gradient over **free** coordinates, and the
    number of coordinates sitting on a bound. The distinction matters: at an
    active bound a nonzero gradient is the correct KKT answer, not a failure, so
    a single max-over-everything figure cannot tell a converged fit from a stuck
    one. That conflation is exactly what disguised the (a, b) parameterisation's
    flat valley as a disagreement with mirt.
    """
    bounds = objective.bounds()
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    x = start
    best = None
    for _ in range(10):
        result = minimize(
            objective.neg_log_lik_and_grad,
            x,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": 50000,
                "maxfun": 200000,
                "ftol": 1e-16,
                "gtol": 1e-12,
                "maxcor": 50,
            },
        )
        _, grad = objective.neg_log_lik_and_grad(result.x)
        at_bound = (np.abs(result.x - lo) < 1e-8) | (np.abs(result.x - hi) < 1e-8)
        free = ~at_bound
        max_free_grad = float(np.max(np.abs(grad[free]))) if free.any() else 0.0
        if best is None or -result.fun > -best[0].fun:
            best = (result, max_free_grad, int(at_bound.sum()))
        if max_free_grad < 1e-6 or np.allclose(result.x, x, atol=1e-13):
            break
        x = result.x
    return best


def run(dataset: str, rule: str = GAUSS_HERMITE, name: str = "mml-scipy") -> Estimates:
    payload = load_matrix(dataset)
    responses = payload["responses"]
    all_item_ids = list(payload["item_ids"])

    dropped_idx = single_category_columns(responses)
    keep = np.setdiff1d(np.arange(len(all_item_ids)), dropped_idx)
    dropped_ids = [all_item_ids[i] for i in dropped_idx]
    fitted_ids = [all_item_ids[i] for i in keep]
    if dropped_ids:
        print(
            f"{dataset}/{name}: dropping {len(dropped_ids)} single-category item(s), "
            "the same rule mirt applies"
        )

    objective = MarginalLikelihood(responses[:, keep], rule=rule)
    result, max_free_grad, n_at_bound = solve(objective, objective.classical_start())
    a, d = objective.unpack(result.x)
    _, b = slope_intercept_to_difficulty(a, d)

    est = Estimates(
        name=name,
        dataset=dataset,
        item_ids=fitted_ids,
        a=a,
        b=b,
        respondent_ids=list(payload["respondent_ids"]),
        theta=eap_theta(objective, a, d),
        parameterisation="fitted as a*theta + d; converted here with b = -d/a",
        # mirt does not constrain the slope positive either, and on these
        # datasets it returns negative ones (2 of 59 on dense, 22 of 183 on
        # sparse). Neither does this.
        a_positive=False,
        theta_metric=f"theta ~ N(0, 1), integrated out on a {N_QUAD}-node '{rule}' grid",
        detail={
            "estimator": "marginal maximum likelihood, L-BFGS-B with analytic gradients",
            "implementation": "crosscheck/run_mml.py, scipy only",
            "optimised_in": "(a, d) slope-intercept; see module docstring",
            "quadrature_rule": rule,
            "n_quad": N_QUAD,
            "log_lik": float(-result.fun),
            "converged": bool(result.success),
            "message": str(result.message),
            "iterations": int(result.nit),
            "max_abs_free_gradient": max_free_grad,
            "n_params_at_bound": n_at_bound,
            "n_items_fitted": len(fitted_ids),
            "n_items_dropped": len(dropped_ids),
            "items_dropped": dropped_ids,
            "d": [float(x) for x in d],
            "theta_estimator": "EAP on the same quadrature grid",
        },
    )
    est.to_json(estimates_path(dataset, est.name))
    print(
        f"{dataset}/{name}: logLik={-result.fun:.4f} in {result.nit} iters, "
        f"converged={result.success}, max|free grad|={max_free_grad:.2e}, "
        f"{n_at_bound} params at a bound, {len(fitted_ids)} items"
    )
    return est


def eap_theta(objective: MarginalLikelihood, a: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Posterior-mean abilities on the quadrature grid.

    EAP rather than ML for the same reason the mirt script uses it: a respondent
    who answered everything correctly has no finite maximum-likelihood ability,
    and the sparse dataset contains such respondents.
    """
    z = a[None, :] * objective.nodes[:, None] + d[None, :]
    log_p = -np.logaddexp(0.0, -z)
    log_q = -np.logaddexp(0.0, z)
    per_cell = objective.x * log_p[:, None, :] + (objective.mask - objective.x) * log_q[:, None, :]
    joint = objective.log_weights[:, None] + per_cell.sum(axis=2)
    post = np.exp(joint - logsumexp(joint, axis=0)[None, :])
    return post.T @ objective.nodes


def main() -> None:
    for dataset in DATASETS:
        # Gauss-Hermite is the estimator on its own merits; the mirt-rule run
        # exists so compare.py can attribute a gap against mirt to the
        # quadrature rather than to the optimiser.
        run(dataset, rule=GAUSS_HERMITE, name="mml-scipy")
        run(dataset, rule=MIRT_RULE, name="mml-scipy-mirtquad")


if __name__ == "__main__":
    main()
