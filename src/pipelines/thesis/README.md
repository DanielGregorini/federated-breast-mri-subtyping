# pipelines/thesis/

This thesis's preprocessing. Every deviation from the authors is argued in
`preprocessing.py` with the measurement that motivated it — including where the
measurement went against the proposal.

| rule | value | why |
|---|---|---|
| slices | 8 spread, trimming 15% each end | neighbouring slices are near-duplicates |
| crop | **80 mm physical**, constant 0.357 mm/px | a proportional crop erases tumour size |
| normalisation | min-max over the whole volume | preserves enhancement kinetics |
| cohorts | I-SPY2 only by default | the source probe reaches 0.9978 when pooled |

## Honest caveat

On the same 99 I-SPY2 test patients this preprocessing scored 0.5837 ± 0.011
against 0.6201 ± 0.024 for the older pipeline. The difference is inside the
0.067 noise floor, so the verdict is *no difference detected* — but it is not the
improvement that was expected. What it did buy is a validation-to-test gap of
+0.015 against +0.073.
