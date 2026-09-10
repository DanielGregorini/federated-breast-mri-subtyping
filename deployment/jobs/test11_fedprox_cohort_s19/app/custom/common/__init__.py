"""Pure machine learning. **No file in this package imports `nvflare`.**

That is the invariant the whole project layout rests on, and it is checked by
No file here imports it. Three things follow from that:

1. The centralised baseline and every federated client run the same trainer, so the
   measured gap is federation and not a difference in code.
2. A bug in the model is found by running this package alone, in seconds, instead of
   by starting a server, four clients and an admin session.
3. This code could be lifted onto a laptop with no NVFLARE installed and still train.

Flat files rather than sub-packages. Each is one responsibility and one file.

    thesis.py      the bridge to core/, the only file that knows
                   where the shared trainer lives
    data.py        per-site loaders, class weights, the trivial baseline
    models.py      the shared network, freezing, and the architecture fingerprint
    training.py    one epoch at a time, with the FedProx fork
    evaluation.py  patient-level metrics
"""

from __future__ import annotations

from . import data, evaluation, models, training  # noqa: F401
from .thesis import THESIS_ROOT, build_config  # noqa: F401

__all__ = ["data", "evaluation", "models", "training", "build_config", "THESIS_ROOT"]
