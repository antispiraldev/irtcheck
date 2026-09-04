"""Item flags for a fitted suite.

There is deliberately no flag logic in this file. `artifact.compute_flags()` is
the single implementation, and synth.py calls the same function to flag its
fabricated fits — so a report built against a synthetic artifact and a report
built against a real one are describing the same rules. A second, "obviously
equivalent" copy here is how `dead` and `insufficient-data` end up disagreeing
between the fitter and the fixtures that were supposed to be testing it, and
nothing would fail loudly when they did.

What this module adds is the counting that goes into `IrtFit.diagnostics`,
which is what makes the honest answer visible at the end of a fit: at 5-15
respondents most low-information items land in `insufficient-data` and almost
none reach `dead`, and a user who sees "412 items: cannot tell" is being told
to add respondents, not that their suite is broken.
"""

from __future__ import annotations

from collections import Counter

from irtcheck.artifact import ALL_FLAGS, Posterior, compute_flags


def item_flags(
    a: Posterior,
    b: Posterior,
    p_correct: list[float],
    theta_mean: list[float],
) -> list[list[str]]:
    """Flags for every item. A thin pass-through, on purpose — see above."""
    return compute_flags(a, b, p_correct, theta_mean)


def flag_counts(flags: list[list[str]]) -> dict[str, int]:
    """How many items carry each flag, including the zeros.

    Every flag is present in the mapping whether or not it fired, so a caller
    reading diagnostics does not have to distinguish "no dead items" from "this
    fitter never sets that flag".
    """
    counter: Counter[str] = Counter(f for item in flags for f in item)
    return {flag: int(counter.get(flag, 0)) for flag in ALL_FLAGS}
