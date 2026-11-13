import json
import tempfile
import unittest
from pathlib import Path

from superagent_adapter.course.consistency import check_summary
from superagent_adapter.course.reporting import generate_report


class Core11ReportTests(unittest.TestCase):
    def setUp(self):
        self.status = {"status": "success", "best_node_id": "n1", "best_score": 0.7,
                       "final_kind": "ensemble", "final_method": "weighted", "final_metric": "accuracy", "final_validation_score": 0.8}
        self.summary = {"best": {"node_id": "n1", "score": 0.7},
                        "final": {"kind": "ensemble", "method": "weighted", "metric": "accuracy", "validation": {"score": 0.8}},
                        "task": {"data_preview": "PRIVATE SAMPLE", "higher_is_better": True},
                        "control": {"controller": "test"}, "diagnostics": {"smoke_runs": 2}}

    def evidence(self):
        return {"schema_version": "1", "task": {}, "policy": {}, "core_status": self.status,
                "core_summary": self.summary, "collection": {"final_choice": "final", "selected_result": {
                    "source": "final", "kind": "ensemble", "method": "weighted", "metric": "accuracy",
                    "validation_score": 0.8, "node_ids": ["n1", "n2"], "reason": "validated"},
                    "solution_mode": "full_member_scripts_plus_combine", "full_script_replay_verified": False,
                    "combination_replay_verified": False, "inference_only_verified": False},
                "evaluation": {"test_accuracy": 0.6, "status": "evaluated"}, "provenance": {}}

    def test_core11_and_missing_anchors(self):
        result = check_summary(self.status, self.summary)
        self.assertEqual(result["status"], "consistent")
        self.assertIn("/best/score", result["checked_fields"])
        self.assertIn("/final/validation/score", result["checked_fields"])
        self.assertEqual(check_summary({}, self.summary)["status"], "not_verifiable")
        self.assertEqual(check_summary({"final_validation_score": None}, {"final": {"validation": {"score": 0.8}}})["status"], "conflict")
        wrong = dict(self.summary, final={"kind": "single", "validation": {"score": 0.4}})
        self.assertEqual({x["path"] for x in check_summary(self.status, wrong)["conflicts"]}, {"/final/kind", "/final/validation/score"})

    def test_null_final_anchors_and_availability(self):
        status = {"final_available": True, "final_metric": None, "final_validation_score": None}
        summary = {"final": {"available": True, "metric": "accuracy", "validation": {"score": 0.99}}}
        result = check_summary(status, summary)
        self.assertEqual({x["path"] for x in result["conflicts"]}, {"/final/metric", "/final/validation/score"})
        self.assertEqual(check_summary({"final_available": True}, {"final": []})["status"], "conflict")
        self.assertEqual(check_summary({"final_available": True}, {"final": {"available": False}})["status"], "conflict")
        self.assertEqual(check_summary({"final_available": False}, {"final": None})["status"], "not_verifiable")
        self.assertEqual(check_summary({}, {"final": []})["status"], "not_verifiable")

    def test_final_members_checked_against_collection(self):
        selected = {"source": "final", "node_ids": ["n1", "n2"]}
        summary = {"final": {"members": [{"node_id": "n2"}, {"node_id": "n1"}]}}
        self.assertEqual(check_summary({}, summary, selected)["status"], "consistent")
        summary["final"]["members"][0]["node_id"] = "other"
        self.assertEqual(check_summary({}, summary, selected)["status"], "conflict")
        self.assertEqual(check_summary({}, summary, {"source": "best_partial", "node_ids": ["n1"]})["status"], "not_verifiable")
        evidence = self.evidence()
        evidence["core_summary"]["final"]["members"] = [{"node_id": "other"}]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "report"
            generate_report(evidence, out)
            context = json.loads((out / "writer_context.json").read_text(encoding="utf-8"))
            self.assertEqual(context["evidence"]["core_summary"], {})
            self.assertEqual(context["evidence"]["consistency"]["status"], "conflict")

    def test_report_separates_scores_and_removes_preview(self):
        seen = []
        def writer(payload):
            seen.append(payload)
            return {"sections": [{"title": "方法", "text": "待核对", "evidence_refs": ["/core_summary/final/kind"]}]}
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "report"
            generate_report(self.evidence(), out, writer)
            rendered = (out / "report.md").read_text(encoding="utf-8")
            self.assertIn("验证最佳分数（单模型）：0.7", rendered)
            self.assertIn("最终方案验证分数：0.8", rendered)
            self.assertIn("测试 accuracy：0.6", rendered)
            self.assertNotIn("PRIVATE SAMPLE", json.dumps(seen, ensure_ascii=False))
            self.assertNotIn("PRIVATE SAMPLE", (out / "evidence.json").read_text(encoding="utf-8"))
            self.assertIn("/core_summary/task/data_preview", json.loads((out / "review.json").read_text(encoding="utf-8"))["omitted_core_fields"])
            self.assertIn("control", seen[0]["evidence"]["core_summary"])
            self.assertIn("diagnostics", seen[0]["evidence"]["core_summary"])

    def test_conflict_excludes_summary_from_writer(self):
        evidence = self.evidence()
        evidence["core_summary"]["final"]["kind"] = "single"
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "report"
            generate_report(evidence, out)
            context = json.loads((out / "writer_context.json").read_text(encoding="utf-8"))
            self.assertEqual(context["evidence"]["core_summary"], {})
            self.assertEqual(context["evidence"]["consistency"]["status"], "conflict")


if __name__ == "__main__":
    unittest.main()
