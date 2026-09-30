"""Measurement protocol v1: how a case that never produced a program is counted.

Synthesis and measurement are different protocols and they fail for different
reasons, so they get different version numbers.  ``agent_protocol_version = 5``
describes how a candidate is *generated* (demo-only feedback, per-case session,
frozen prompt); this module describes how a case is *counted* once generation
has stopped.  Keeping them apart lets checkpoints whose generation protocol has
not changed stay valid when the counting rule changes.

The rule:

  * A case gets at most ``MAX_CASE_ATTEMPTS`` case attempts.
  * Only ``RETRYABLE_FAILURE_KINDS`` -- transport failures with no evidence
    about the model's ability -- earn the second attempt, and it must be a
    fresh session.
  * Once that budget is spent the case is a legitimate benchmark outcome and
    scores CP/EX/P/R/F1 = 0.  It is not dropped: dropping it would silently
    shrink the denominator, and a system that cannot return a program within
    the budget has failed the task.
  * Everything else -- quota/402, disabled account, CLI usage or config
    errors, image/prompt/config fingerprint drift -- makes the *run* invalid,
    never a zero.  Those say something about our infrastructure, not about the
    agent, and scoring them would import our billing into the benchmark.
  * A candidate that exists when the clock runs out is frozen and scored as it
    stands; only a case with no candidate at all reaches terminal zero.

Terminal zeros must never be scored by *executing* what is on disk.  When the
agent returns nothing, the harness still writes the schema block, and a bare
schema compiles and produces an empty relation -- which earns a perfect
negative-example exclusion for emitting nothing at all, and counts as a
compile pass.  ``terminal_zero_detail`` therefore synthesises the zero row
instead of running Soufflé over it.
"""

MEASUREMENT_PROTOCOL_VERSION = 1

MAX_CASE_ATTEMPTS = 2

# Transport failures: the request never came back, so nothing was learned about
# the model.  Anything outside this set either says something about the agent
# (and is already scored through its artifact) or about our account (and
# invalidates the run).
RETRYABLE_FAILURE_KINDS = frozenset({"timeout", "empty_response"})

# Failure kinds that can never become a score, whatever the budget says.
INVALIDATING_FAILURE_KINDS = frozenset({
    "quota", "account_disabled", "cli_usage", "config_drift",
    "image_drift", "prompt_drift", "auth",
})

SCORE_POLICY_ARTIFACT = "artifact"
SCORE_POLICY_TERMINAL_ZERO = "terminal_zero"


def is_retryable_kind(kind) -> bool:
    return str(kind or "").strip() in RETRYABLE_FAILURE_KINDS


def is_invalidating_kind(kind) -> bool:
    """Unknown kinds invalidate rather than score.

    The default matters more than the list does.  A kind we have never seen is
    a kind whose meaning we have not established, and the safe reading of an
    unestablished failure is "this run is not measurable", not "the agent
    scored zero".  Adding a kind to ``RETRYABLE_FAILURE_KINDS`` is a decision
    someone has to make on purpose.
    """
    kind = str(kind or "").strip()
    if not kind:
        return False
    return not is_retryable_kind(kind)


def classify_case(row: dict, attempts: int) -> dict:
    """The four measurement fields for one case row.

    ``attempts`` is the number of *case* attempts spent, which is not the
    ``attempts`` field on the row: that one counts turns inside the final case
    attempt.  The two were conflated once and produced a retry budget that
    looked exhausted after a single two-turn session.
    """
    artifact_valid = bool(row.get("checkpoint_valid"))
    kind = row.get("failure_kind") or ""

    if artifact_valid:
        return {
            "artifact_valid": True,
            "measurement_complete": True,
            "score_policy": SCORE_POLICY_ARTIFACT,
            "retry_budget_exhausted": False,
            "case_attempts": attempts,
            "measurement_protocol_version": MEASUREMENT_PROTOCOL_VERSION,
        }

    exhausted = attempts >= MAX_CASE_ATTEMPTS
    scorable = is_retryable_kind(kind) and exhausted
    return {
        "artifact_valid": False,
        "measurement_complete": bool(scorable),
        "score_policy": SCORE_POLICY_TERMINAL_ZERO if scorable else "",
        "retry_budget_exhausted": exhausted,
        "case_attempts": attempts,
        "measurement_protocol_version": MEASUREMENT_PROTOCOL_VERSION,
    }


def is_terminal_zero(row: dict) -> bool:
    return (
        row.get("measurement_protocol_version") == MEASUREMENT_PROTOCOL_VERSION
        and row.get("score_policy") == SCORE_POLICY_TERMINAL_ZERO
        and row.get("measurement_complete") is True
    )


def terminal_zero_detail(case_id: str, row: dict, variant_count: int) -> dict:
    """A zero row built without executing anything on disk.

    ``neg_exclusion`` is ``None`` rather than ``0.0`` on purpose: the case
    contributes no negative-example observation at all, and a 0.0 would drag
    the mean down as if the agent had hit every counterexample.  ``None`` is
    already how the evaluator spells "no negatives to exclude", and the
    aggregate skips it.
    """
    return {
        "case_id": case_id,
        "compile_ok": False,
        "perfect_match": False,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "neg_total": 0,
        "neg_hit": 0,
        "neg_exclusion": None,
        "variant_count": variant_count,
        "latency_ms": 0.0,
        "iterations_used": row.get("iterations_used", 0),
        "best_iteration": row.get("best_iteration", 0),
        "session_turns": row.get("session_turns", 0),
        "error": (
            f"terminal inference failure: {row.get('failure_kind') or 'unknown'} "
            f"after {row.get('case_attempts', MAX_CASE_ATTEMPTS)} case attempt(s); "
            f"scored zero under measurement-v1 without executing the schema stub"
        ),
        "score_policy": SCORE_POLICY_TERMINAL_ZERO,
        "per_variant": [],
    }
