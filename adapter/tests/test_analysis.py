import json

from superagent_adapter.analysis import _spearman, analyze


def _write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


def test_analyze_minimal_output(tmp_path):
    o = tmp_path / "out"
    nodes = [
        {"id": "root", "kind": "root", "parent_id": None, "failed": None, "score": None, "failure_reasons": [], "error_type": None,
         "error_message": None, "code": "", "branch_id": None, "step": 0, "method_summary": ""},
        {"id": "a", "kind": "draft", "parent_id": "root", "failed": False, "score": 0.5, "failure_reasons": [], "error_type": None,
         "error_message": None, "code": "from_pretrained('m/x')", "branch_id": 1, "step": 1, "method_summary": "m"},
        {"id": "b", "kind": "refine", "parent_id": "a", "failed": False, "score": 0.6, "failure_reasons": [], "error_type": None,
         "error_message": None, "code": "", "branch_id": 1, "step": 2, "method_summary": "m"},
        {"id": "c", "kind": "refine", "parent_id": "b", "failed": True, "score": None, "failure_reasons": ["runtime_error"],
         "error_type": "OSError", "error_message": "x", "code": "", "branch_id": 1, "step": 3, "method_summary": ""},
    ]
    _write(o / "logs" / "search_tree.json", {"higher_is_better": True, "nodes": nodes})
    (o / "logs" / "decisions.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"selection_mode": "initial", "action": "draft", "reason": "root_has_capacity", "details": {}},
        {"selection_mode": "chain", "action": "refine", "reason": "branch_progressing", "details": {}},
        {"selection_mode": "elite_pick", "action": "refine", "reason": "branch_progressing", "details": {"left_chain_at": "b"}},
    ]))
    usage = {"calls": 3, "failed_calls": 0, "prompt_tokens": 10, "completion_tokens": 5, "reasoning_tokens": 1, "cost": 0.3,
             "currency": "USD", "by_agent": {"x": {"calls": 3, "cost": 0.3}}}
    _write(o / "report" / "run_summary.json", {"score_timeline": [], "best": {"path": []}, "key_events": [],
                                                "resources": {"llm": usage, "embedding": {"cost": 0.0}, "exec": {}},
                                                "final": {"kind": "single"}, "diagnostics": {"chain_share": 0.3}})
    _write(o / "status.json", {"best_node_id": "b", "best_score": 0.6})
    _write(o / "top" / "rank_01" / "score.json", {"node_id": "b", "score": 0.6})
    _write(o / "top" / "rank_02" / "score.json", {"node_id": "a", "score": 0.5})
    grades = tmp_path / "grades.json"
    rep = {"score": 0.61, "gold_medal": False, "silver_medal": True, "bronze_medal": False, "above_median": True,
           "gold_threshold": 0.9, "silver_threshold": 0.6, "bronze_threshold": 0.5, "median_threshold": 0.4}
    _write(grades, {"primary": "final", "grades": {"final": rep, "top_rank_01": rep, "top_rank_02": {**rep, "score": 0.52}}})
    res = analyze(o, grades)
    assert res["operators"]["refine"] == {"count": 2, "ok": 1, "ok_rate": 0.5}
    assert res["selection_modes"]["elite_pick"] == 1 and res["trigger_evidence"]["chains_left_for_elite"] == 1
    assert res["improvements_over_parent"]["improved"] == 1
    assert res["grades"]["final"]["medal"] == "silver" and res["primary"] == "final"
    assert len(res["validation_vs_test"]["rows"]) == 2
    assert res["pretrained"]["from_pretrained_literals"] == {"m/x": 1}
    assert res["pretrained"]["nodes_failed_with_model_loading_errors"] == ["c"]
    assert res["final"] == {"kind": "single"} and res["diagnostics"]["chain_share"] == 0.3


def test_spearman():
    assert _spearman([1, 2, 3], [10, 20, 30]) == 1.0
    assert _spearman([1, 2, 3], [30, 20, 10]) == -1.0
    assert _spearman([1, 2], [1, 2]) is None
