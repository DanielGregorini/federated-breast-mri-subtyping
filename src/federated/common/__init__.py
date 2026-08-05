"""Pure machine learning. **No file in this package imports `nvflare`.**

That is the invariant the whole project layout rests on, and it is checked by
`scripts/verify_data.py --check-imports`. Three things follow from it:

1. The centralised baseline and every federated client run the same trainer, so the
   gap RQ1 measures is federation and not a difference in code.
2. A bug in the model is found by running this package alone, in seconds, instead of
   by starting a server, four clients and an admin session.
3. This code could be lifted onto a laptop with no NVFLARE installed and still train.

The modules mirror `docs/ARCHITECTURE.md`, as flat files rather than sub-packages —
each is a single responsibility and a single file, and a folder holding one module
would be structure for its own sake.

    thesis.py      the bridge to src/core/ — the only file that
                   knows where the classifier phase lives
    data.py        per-site loaders, class weights, the trivial baseline
    models.py      the shared network, freezing, and the architecture fingerprint
    training.py    one epoch at a time, with the FedProx fork
    evaluation.py  patient-level metrics
"""

from __future__ import annotations

from . import data, evaluation, models, training  # noqa: F401
from .thesis import THESIS_ROOT, build_config  # noqa: F401

__all__ = ["data", "evaluation", "models", "training", "build_config", "THESIS_ROOT"]
