"""Legacy scoring entry point.

Substring and regular-expression matching are not valid semantic promotion
evidence. Callers must use the typed inference judge and statistical release
gate in semantic_judge.py and reliability.py.
"""

from __future__ import annotations

def score_transcript(*args, **kwargs):
    del args, kwargs
    raise RuntimeError(
        "heuristic transcript scoring is prohibited; use TypedSemanticJudge and evaluate_release_gate"
    )
