from superagent.codegen import NOTE_NO_BLOCKS, NOTE_NO_MATCH, PatchState, patch_result, patch_step
from superagent.patching import apply_patch
from superagent.planner import normalize_and_check, render_plan


def P(search, replace):
    return f"<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE\n"


def test_planner_components_string():
    plan, errs = normalize_and_check({"rationale": "r", "components": "model_def", "changes": {"model_def": "x"}}, False)
    assert plan["components"] == ["model_def"] and not errs


def test_planner_components_from_changes():
    plan, errs = normalize_and_check({"rationale": "r", "components": [], "changes": {"data_prep": "x"}}, False)
    assert plan["components"] == ["data_prep"] and not errs


def test_planner_changes_string():
    plan, errs = normalize_and_check({"rationale": "r", "components": ["data_prep", "model_def"], "changes": "do X"}, False)
    assert plan["rationale"] == "r\n\ndo X"
    assert plan["changes"] == {"data_prep": "do X", "model_def": "do X"} and not errs


def test_planner_empty_changes():
    _p, errs = normalize_and_check({"rationale": "r", "components": ["model_def"], "changes": {}}, False)
    assert errs
    _p, errs = normalize_and_check({"rationale": "r", "components": ["model_def"], "changes": {}}, True)
    assert errs
    _p, errs = normalize_and_check({"rationale": "r", "components": [], "changes": {}}, True)
    assert errs


def test_render_plan():
    text = render_plan({"rationale": "why", "components": ["model_def"], "changes": {}})
    assert "why" in text and "model_def" in text and "targeted improvements" in text


def test_patch_exact():
    code = "def f(x):\n    return x + 1\n"
    new, n = apply_patch(code, P("    return x + 1", "    return x + 2"))
    assert n == 1 and "    return x + 2" in new


def test_patch_indent_tolerant():
    code = "class M:\n    def f(self):\n        y = 1\n        return y\n"
    new, n = apply_patch(code, P("y = 1\nreturn y", "y = 2\nif y:\n    return y"))
    assert n == 1
    assert new == "class M:\n    def f(self):\n        y = 2\n        if y:\n            return y\n"


def test_patch_other_cases():
    code = "a = 1\n"
    new, n = apply_patch(code, P("", "b = 2"))
    assert n == 1 and new.endswith("\n\nb = 2\n")
    new, n = apply_patch(code, P("zzz", "b"))
    assert n == 0 and new == code
    new, n = apply_patch(code, P("a = 1", "a = 3") + P("nope", "x"))
    assert n == 1 and "a = 3" in new
    new, n = apply_patch(code, P("a = 1", "a = 1\nc = 5") + P("c = 5", "c = 6"))
    assert n == 2 and "c = 6" in new


def run_steps(parent_code, replies, max_attempts=3):
    """Feed reply texts to the patch loop's step function. Each reply is text or (text, truncated)."""
    state = PatchState(parent_code=parent_code, current=parent_code)
    seen = []
    for reply in replies[:max_attempts]:
        seen.append((state.current, state.retry_note))
        text, truncated = reply if isinstance(reply, tuple) else (reply, False)
        if patch_step(state, text, truncated):
            break
    return patch_result(state), seen


def test_cumulative_retry_no_blocks_then_ok():
    res, seen = run_steps("a = 1\n", ["no patch here", P("a = 1", "a = 2")])
    assert res.code == "a = 2\n" and res.attempts == 2 and not res.partial
    assert seen[1][1] == NOTE_NO_BLOCKS


def test_cumulative_retry_truncated():
    first = P("a = 1", "a = 2") + "<<<<<<< SEARCH\nb = 1\n=======\n"
    res, seen = run_steps("a = 1\nb = 1\n", [(first, True), P("b = 1", "b = 2")])
    assert res.code == "a = 2\nb = 2\n" and not res.partial
    assert seen[1][0] == "a = 2\nb = 1\n" and "cut off" in seen[1][1]


def test_cumulative_retry_truncation_detected_from_markers():
    first = P("a = 1", "a = 2") + "<<<<<<< SEARCH\nb = 1\n=======\n"
    res, seen = run_steps("a = 1\nb = 1\n", [first, P("b = 1", "b = 2")])
    assert res.code == "a = 2\nb = 2\n" and "cut off" in seen[1][1]


def test_cumulative_retry_all_fail():
    res, seen = run_steps("a = 1\n", [P("x", "y")] * 3)
    assert res.code is None and res.attempts == 3
    assert seen[1][1].startswith(NOTE_NO_MATCH)
    assert "SEARCH:\nx\nREPLACE:\ny" in seen[1][1]


def test_cumulative_retry_partial_accept():
    first = P("a = 1", "a = 2") + "<<<<<<< SEARCH\nb = 1\n"
    res, _ = run_steps("a = 1\nb = 1\n", [first, P("x", "y"), P("x", "y")])
    assert res.code == "a = 2\nb = 1\n" and res.applied_blocks == 1
