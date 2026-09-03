"""`irtcheck fit` — owned by wave1/fit.

Wave 0 leaves this a stub so that cli.py can declare the whole command surface
up front and then stay frozen. Replace `run` with the implementation; do not
change its signature without changing cli.py, which is a single-owner file.
"""

from __future__ import annotations

from typing import Any


def run(**kwargs: Any) -> None:
    raise NotImplementedError(
        "`irtcheck fit` is not implemented yet — it belongs to the wave1/fit brief "
        "(see docs/build-plan.html). Expected arguments: responses, output, respondent_key, fmt, priors, epochs, seed, device, embed_responses."
    )
