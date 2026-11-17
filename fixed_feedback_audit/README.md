# Fixed-budget feedback control builder

This directory provides a **local, data-agnostic construction utility**, not a new recommendation model, learning algorithm, or policy-selection method. `audit_controls.py` depends only on Python standard-library modules and NumPy; it imports no repository modules, fits no models, generates no teacher feedback, and reads no data files. 

`QueryEntry(pair, bucket, teacher_label)` describes a fixed queried pool Q. Pair IDs and bucket IDs can be hashable identifiers. Each Q pair must occur exactly once and have one binary cached teacher label. Supply an ordered template, an aligned boolean positive mask, and (optionally) the original target array. The fixed retained positive set **Pstar** is derived from the template's positive positions; these pairs must be teacher-positive entries of Q.

`build_control` changes only pairs in negative positions:

| Selector | Eligible pool within each fixed bucket | Ordering supplied by caller |
|---|---|---|
| `T` | Cached teacher-negative Q pairs | Explicit priorities |
| `R` | **Q minus retained Pstar** | Explicit priorities |
| `Hard` | Q minus retained Pstar | Frozen score descending, then explicit tie priority |

R deliberately includes teacher-positive pairs that were not retained in Pstar. Excluding *all* teacher positives would implement a different control. A retained synthetic negative target does not assert that the pair is a true human negative. The diagnostic counts this inclusion explicitly.

```python
import numpy as np
from audit_controls import QueryEntry, build_control

Q = [
    QueryEntry((0, 9), "item9", 1),  # retained positive
    QueryEntry((1, 9), "item9", 1),  # accepted but NOT retained
    QueryEntry((2, 9), "item9", 0),
]
template = [(0, 9), (2, 9)]
positive = np.array([True, False])
targets = np.array([1, 0], dtype=np.float32)
keys = {(0, 9): 2, (1, 9): 0, (2, 9): 1}

control = build_control(
    Q, template, positive, "R",
    priorities=keys, targets=targets,
    slot_quotas={"item9": 1},
    priority_contract="common_permutation",
)
assert control.pairs == ((0, 9), (1, 9))
assert np.array_equal(control.targets, targets)
arrays = control.arrays()  # integer-ID convenience: u, i, y
```

The builder never draws random numbers. For a uniform random control, the caller must provide a uniform-permutation priority law. Reuse the same map for common-permutation pairing, or pass separate maps with `priority_contract="selector_independent"`. Arbitrary deterministic priorities implement a deterministic construction rather than a claim of uniform sampling. Keys must be complete and mutually comparable within each used bucket; avoid NaN or other non-total order values. Optional `tie_priorities` supplies an explicit tie rule. Remaining ties fall back to fixed Q-entry order. The original R89 pair-ID tie rule and R90 fixed-pool tie rule are both replayed explicitly in the separate audit.

For `Hard`, supply `frozen_scores`, `tie_priorities`, and a nonempty `score_provenance` string, such as a frozen TRAIN-fitted checkpoint digest. Scores must be finite. The module does not call a ranker, prove when the scores were frozen, or authorize held-out labels for mining; that information contract remains the caller's responsibility.

Negative buckets default to the template pair's bucket. `slot_quotas` **validates** the exact negative-slot counts; it does not silently invent a new slot layout. If allocation changes, provide an explicit aligned `slot_buckets` sequence for the negative positions. Positive positions stay fixed. Targets are copied with their original dtype, shape, auxiliary columns, and row order. Context or other feature columns are caller-owned: replace pair columns only and keep those columns fixed; this module does not reconstruct contexts.

`capped_uniform_quotas(capacities, budget, tie_order)` implements capped integer max-min waterfilling with an explicit deterministic remainder order. The order must contain every capacity bucket exactly once. Independent direction budgets should be passed in separate calls. The function neither chooses the budget nor estimates capacities from validation/test outcomes.

Missing priorities, duplicate Q/template pairs, inconsistent Pstar/targets, infeasible capacities, and quota/layout mismatches raise `ValueError`. There is no silent fallback, positive-dropping behavior, or automatic quota relaxation. The result contains input/output diagnostics, including changed negative-pair counts and unretained teacher-positive inclusion. The integer-ID `.arrays()` convenience method preserves target bytes; for noninteger IDs, use `.pairs` or select an appropriate `id_dtype`.

Verification is retained locally:

- `test_audit_controls.py` / `SYNTHETIC_VERIFICATION.json`: positive/target/order invariants, duplicates, capacities, ineligible pairs, R-versus-Rprime, common/independent priorities, frozen-score hard ranking/ties, and 360 waterfill comparisons against a simple iterative implementation.
- `replay_prepared.py` / `PREPARED_INPUT_REPLAY.json`: independently uses this builder to reproduce **108 R89 + 24 R90 = 132 existing prepared T/R arrays**, exactly matching u/i/y dtype, shape and bytes. Original code is imported only for its priority functions; original redraw/fit/evaluate functions are not called. All 14 capped-uniform quota reconstructions also match. Used input/source hashes are unchanged.
- The replay reads only existing synthetic-feedback Q, prepared arrays and mapping metadata. It performs no model construction/forward pass, human VAL/TEST evaluation, GPU work or new experiment. Actual Hard arrays are not included in the 132-array claim; Hard is verified with synthetic frozen-score fixtures.

Run the synthetic test with `python -B test_audit_controls.py`. The prepared-input replay requires the original private checkout and is kept separate so the portable module can be copied without it. The portable module and synthetic test are provided in this directory; the prepared-input replay and its private inputs remain in the research checkout.

## 한국어 사용 범위

이 모듈은 고정된 Q·Pstar·negative quota에서 비교용 데이터를 만드는 도구입니다. 새 모델이나 새로운 학습 방법을 주장하지 않습니다. `audit_controls.py`만 복사하면 Python 표준 라이브러리와 NumPy로 사용할 수 있습니다.

T는 teacher가 negative로 판정한 후보에서 고르고, R은 **retained Pstar만 제외한 Q**에서 고릅니다. R에서 retained되지 않은 teacher-positive까지 제외하면 다른 대조군이 됩니다. Hard는 사용자가 미리 고정한 점수와 tie priority를 받으며, 이 모듈이 모델을 학습하거나 점수를 다시 계산하지 않습니다.

사용자가 priority map과 bucket별 quota를 명시합니다. 동일 map을 재사용하거나 selector별 독립 map을 사용할 수 있으며, 무작위성은 호출자가 정의합니다. Positive pair·위치·target dtype/shape/행 순서는 유지되고 negative pair만 바뀝니다. 새 quota를 적용할 때는 negative slot의 목표 bucket 배치도 명시해야 합니다. Context 등 다른 열은 호출자가 그대로 유지해야 합니다.

기존 R89/R90의 준비된 T/R 입력 132개를 원본 함수의 priority만 이용해 u/i/y byte 단위로 재현했습니다. 실제 TEST/VAL 성능, 새로운 학습, 모델 추론, GPU는 사용하지 않았습니다. 이는 기존 데이터 구성의 재사용성 검증이며 새 데이터 성능이나 온라인 서비스 제공을 의미하지 않습니다. 공개 디렉터리에는 독립 모듈과 synthetic 테스트가 포함되며, 실제 입력 재현 자료는 연구용 checkout에 보존되어 있습니다.
