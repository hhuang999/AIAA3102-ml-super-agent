from helpers import mk, tree
from superagent.textfmt import attempts_summary, score_change, trim_output


def test_trim_keeps_marker():
    text = "a" * 4990 + "\nVALIDATION_SCORE: 0.5\n" + "b" * (10000 - 4990 - 23)
    assert len(text) == 10000
    out = trim_output(text)
    assert out.startswith(text[:2500]) and out.endswith(text[-2500:])
    assert "VALIDATION_SCORE: 0.5" in out and "5000 characters omitted" in out
    short = "x" * 5000
    assert trim_output(short) == short


def test_score_change():
    t = tree(True)
    p = mk(t, "draft", t.root, 0.80)
    n = mk(t, "refine", p, 0.82)
    assert score_change(n, t.direction) == "0.8000 → 0.8200 (+0.0200 ↑)"
    t2 = tree(False)
    p2 = mk(t2, "draft", t2.root, 0.50)
    n2 = mk(t2, "refine", p2, 0.45)
    assert score_change(n2, t2.direction) == "0.5000 → 0.4500 (+0.0500 ↑)"
    f = mk(t, "draft", t.root, failed=True)
    r = mk(t, "repair", f, 0.82)
    assert score_change(r, t.direction) == "0.8200"


def test_attempts_order():
    t = tree()
    mk(t, "draft", t.root, 0.80, plan="P80")
    mk(t, "draft", t.root, failed=True, plan="PFAIL")
    mk(t, "draft", t.root, failed=None, plan="PPEND")
    mk(t, "draft", t.root, 0.85, plan="P85")
    s = attempts_summary(t.root, t.direction)
    order = [s.index(x) for x in ("P85", "P80", "PFAIL", "PPEND")]
    assert order == sorted(order)
    first = s.splitlines()[0]
    assert "4 attempts" in first and "1 waiting" in first and "3 already run" in first and "2 successful" in first and "0.8500" in first
    assert attempts_summary(mk(t, "draft", t.root, 0.1), t.direction) == ""
