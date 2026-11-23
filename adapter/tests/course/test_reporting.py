import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

from superagent_adapter.course.reporting import generate_report
from superagent_adapter.course.report_llm import make_chat_writer, ReportWriterError

def evidence():
    return {"schema_version": "1", "task": {"name": "fixture"}, "policy": {"validation": "pending"}, "core_status": {"status": "success", "best_metric": 0.75}, "core_summary": "full script", "collection": {}, "evaluation": {"test_score": {"status": "not_requested"}, "validation_accuracy": 0.75}, "provenance": {"kind": "fixture"}}

class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name) / "new"

    def test_deterministic_and_existing_directory(self):
        result = generate_report(evidence(), self.out)
        self.assertEqual(result["mode"], "deterministic_no_llm")
        text = (self.out / "report.md").read_text(encoding="utf-8")
        self.assertIn("未使用 LLM", text)
        self.assertIn("0.75", text)
        self.assertIn("完整脚本", text)
        with self.assertRaises(FileExistsError):
            generate_report(evidence(), self.out)

    def test_valid_writer_cannot_replace_authoritative_metrics(self):
        def writer(payload):
            self.assertIn("不可信", payload["instruction"])
            payload["evidence"]["evaluation"]["validation_accuracy"] = 1
            return {"sections": [{"title": "说明", "text": "请检查指标。", "evidence_refs": ["/evaluation/validation_accuracy"]}], "usage": {"prompt_tokens": 10, "completion_tokens": 3}, "model": "fixture", "call_id": "x", "latency_secs": 0.1, "evaluation": {"validation_accuracy": 1}}
        self.assertEqual(generate_report(evidence(), self.out, writer)["mode"], "writer_human_review_required")
        report = json.loads((self.out / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["authoritative_evidence"]["evaluation"]["validation_accuracy"], 0.75)
        ledger = json.loads((self.out / "llm_calls.jsonl").read_text())
        self.assertEqual(ledger["prompt_tokens"], 10)
        self.assertEqual(ledger["agent"], "report_writer")
        self.assertNotEqual(ledger["call_id"], ledger["provider_call_id"])
        self.assertTrue(ledger["ok"] and ledger["output_valid"])

    def test_bad_refs_and_malformed_fallback(self):
        for i, result in enumerate(({"sections": [{"title": "x", "text": "y", "evidence_refs": ["/evaluation/absent"]}], "usage": {"prompt_tokens": 4}}, {"sections": "bad"})):
            out = Path(self.temp.name) / str(i)
            self.assertEqual(generate_report(evidence(), out, lambda _: result)["mode"], "deterministic_after_rejection")
            self.assertTrue((out / "rejected_writer_output.json").exists())
        ledger = json.loads((Path(self.temp.name) / "0" / "llm_calls.jsonl").read_text())
        self.assertEqual(ledger["prompt_tokens"], 4)
        self.assertTrue(ledger["ok"])
        self.assertFalse(ledger["output_valid"])

    def test_invalid_input_rejected_before_directory_creation(self):
        for key in ("raw_predictions", "api_key", "test_labels", "source_code"):
            data = evidence()
            data["collection"][key] = "sensitive"
            with self.assertRaises(ValueError):
                generate_report(data, self.out)
            self.assertFalse(self.out.exists())
        data = evidence()
        data["provenance"]["prompt_tokens"] = 12
        generate_report(data, self.out)

    def test_malformed_response_retains_usage_without_key(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, size):
                return json.dumps({"id": "call", "usage": {"prompt_tokens": 42, "completion_tokens": 8}, "choices": [{"message": {"content": "not json"}}]}).encode()
        with patch.dict(os.environ, {"TEST_REPORT_KEY": "private-key"}), patch("urllib.request.build_opener", return_value=Mock(open=Mock(return_value=Response()))):
            writer = make_chat_writer(base_url="https://example.invalid/v1", model="fixture", api_key_env="TEST_REPORT_KEY")
            self.assertEqual(generate_report(evidence(), self.out, writer)["mode"], "deterministic_after_rejection")
        ledger = (self.out / "llm_calls.jsonl").read_text()
        self.assertEqual(json.loads(ledger)["prompt_tokens"], 42)
        self.assertNotIn("private-key", ledger)

    def test_real_schema_metadata_and_bounded_context(self):
        data = evidence()
        data["task"]["rows"] = {"train": 200, "val": 40, "test": 50}
        data["policy"]["test_label_boundary"] = {"mode": "separate_paths", "os_enforced": False}
        data["evaluation"].update({"rows": 50, "predictions_sha256": "a" * 64})
        data["collection"]["files_sha256"] = {"best_solution/artifacts/raw_predictions.csv": "b" * 64}
        data["core_status"]["num_nodes"] = 7
        data["core_summary"] = {"score_timeline": list(range(25)), "key_events": ["x" * 3000]}
        def writer(payload):
            self.assertEqual(len(payload["evidence"]["core_summary"]["score_timeline"]), 8)
            self.assertLess(len(payload["evidence"]["core_summary"]["key_events"][0]), 510)
            return {"sections": [{"title": "迭代解释", "text": "请核对轨迹。", "evidence_refs": ["/core_summary/score_timeline/0"]}]}
        generate_report(data, self.out, writer)
        review = json.loads((self.out / "review.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(review["context_truncation"]), 2)
        self.assertEqual(len(json.loads((self.out / "evidence.json").read_text(encoding="utf-8"))["core_summary"]["score_timeline"]), 25)
        text = (self.out / "report.md").read_text(encoding="utf-8")
        self.assertIn("验证最佳分数", text)
        self.assertIn("测试 accuracy", text)
        self.assertIn("节点数量：7", text)
        self.assertIn("测试标签访问边界", text)

    def test_failure_ledger_and_timeout(self):
        def fail(_): raise RuntimeError("secret text")
        generate_report(evidence(), self.out, fail)
        ledger = json.loads((self.out / "llm_calls.jsonl").read_text())
        self.assertFalse(ledger["ok"])
        self.assertIsNone(ledger["prompt_tokens"])
        self.assertNotIn("secret text", (self.out / "review.json").read_text(encoding="utf-8"))
        for timeout in (float("nan"), float("inf"), 121, 0):
            with self.assertRaises(ValueError):
                make_chat_writer(base_url="https://example.invalid/v1", model="fixture", api_key_env="KEY", timeout=timeout)

if __name__ == "__main__":
    unittest.main()
