"""Inter-annotator agreement metrics.

Cohen's kappa with 95% bootstrap CI is the headline metric.  We also expose
percent agreement and a simple confusion matrix.

Implemented without sklearn so the module is usable in lightweight
environments.  If you want exact match with sklearn.metrics.cohen_kappa_score
you should still get the same numbers up to floating-point error.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Sequence


@dataclass
class AgreementResult:
    n: int
    kappa: float
    percent_agreement: float
    kappa_ci_low: float
    kappa_ci_high: float
    confusion: dict[tuple[str, str], int]


def _confusion(a: Sequence, b: Sequence) -> dict[tuple, int]:
    if len(a) != len(b):
        raise ValueError(f"Mismatched lengths: {len(a)} vs {len(b)}")
    c: dict[tuple, int] = Counter()
    for x, y in zip(a, b):
        c[(x, y)] += 1
    return dict(c)


def _kappa_from_confusion(conf: dict[tuple, int]) -> tuple[float, float]:
    total = sum(conf.values())
    if total == 0:
        return float("nan"), float("nan")

    labels = sorted({lab for pair in conf for lab in pair})
    p_o = sum(conf.get((l, l), 0) for l in labels) / total

    a_marg = Counter()
    b_marg = Counter()
    for (x, y), n in conf.items():
        a_marg[x] += n
        b_marg[y] += n

    p_e = sum((a_marg[l] / total) * (b_marg[l] / total) for l in labels)
    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else float("nan"), p_o
    kappa = (p_o - p_e) / (1.0 - p_e)
    return kappa, p_o


def cohen_kappa(
    a: Sequence,
    b: Sequence,
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> AgreementResult:
    if len(a) != len(b):
        raise ValueError(f"Mismatched lengths: {len(a)} vs {len(b)}")
    n = len(a)

    conf = _confusion(a, b)
    kappa, p_o = _kappa_from_confusion(conf)

    rng = random.Random(seed)
    if n_bootstrap > 0 and n > 0:
        samples = []
        for _ in range(n_bootstrap):
            idx = [rng.randrange(n) for _ in range(n)]
            sub = _confusion([a[i] for i in idx], [b[i] for i in idx])
            k, _ = _kappa_from_confusion(sub)
            if not math.isnan(k):
                samples.append(k)
        samples.sort()
        if samples:
            lo = samples[int(0.025 * len(samples))]
            hi = samples[int(0.975 * len(samples)) - 1]
        else:
            lo = hi = float("nan")
    else:
        lo = hi = float("nan")

    return AgreementResult(
        n=n,
        kappa=kappa,
        percent_agreement=p_o,
        kappa_ci_low=lo,
        kappa_ci_high=hi,
        confusion=conf,
    )


def format_report(name: str, r: AgreementResult) -> str:
    lines = [
        f"=== {name} ===",
        f"  n                  : {r.n}",
        f"  Cohen's kappa      : {r.kappa:.3f}  "
        f"(95% CI [{r.kappa_ci_low:.3f}, {r.kappa_ci_high:.3f}])",
        f"  percent agreement  : {r.percent_agreement:.3f}",
        f"  confusion          : {r.confusion}",
    ]
    return "\n".join(lines)
