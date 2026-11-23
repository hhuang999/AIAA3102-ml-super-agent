import copy
import json
import unittest

from superagent_adapter.course.consistency import check_summary


class ConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.status = {"best_node_id": "node-1", "best_metric": 0.75}

    def test_consistent_and_tolerance(self):
        result = check_summary(self.status, {"best": {"node_id": "node-1", "metric": 0.75 + 1e-13}})
        self.assertEqual(result["status"], "consistent")
        self.assertEqual(result["checked_fields"], ["/best/node_id", "/best/metric"])
        self.assertIn("全文", result["note"])

    def test_node_and_score_disagreement(self):
        result = check_summary(self.status, {"best": {"node_id": "other", "metric": 0.5}})
        self.assertEqual(result["status"], "conflict")
        self.assertEqual({x["path"] for x in result["conflicts"]}, {"/best/node_id", "/best/metric"})

    def test_every_present_alias_checked(self):
        summary = {"best": {"node_id": "node-1", "best_node_id": "bad", "metric": 0.75, "validation_score": 0.7, "best_metric": 0.75}}
        result = check_summary(self.status, summary)
        self.assertEqual(result["status"], "conflict")
        self.assertEqual(len(result["checked_fields"]), 5)
        self.assertEqual({x["path"] for x in result["conflicts"]}, {"/best/best_node_id", "/best/validation_score"})

    def test_missing_or_invalid_anchor(self):
        self.assertEqual(check_summary(self.status, {})["status"], "not_verifiable")
        self.assertEqual(check_summary(self.status, {"best": {"unrelated": 1}})["status"], "not_verifiable")
        for bad in ({}, {"best_node_id": "", "best_metric": 0.5}, {"best_node_id": "x", "best_metric": float("nan")}, {"best_node_id": "x", "best_metric": True}):
            self.assertEqual(check_summary(bad, {"best": {"node_id": "x"}})["status"], "not_verifiable")

    def test_bad_best_and_bad_values_are_safe_and_inputs_unchanged(self):
        data = {"best": {"node_id": ["not", "scalar"], "metric": float("nan"), "validation_score": True, "best_metric": float("inf")}}
        snapshot = copy.deepcopy(data)
        result = check_summary(self.status, data)
        self.assertEqual(result["status"], "conflict")
        self.assertEqual(len(result["conflicts"]), 4)
        json.dumps(result, allow_nan=False)
        self.assertEqual(data["best"]["node_id"], snapshot["best"]["node_id"])
        self.assertTrue(data["best"]["metric"] != data["best"]["metric"])
        result = check_summary(self.status, {"best": ["invalid"]})
        self.assertEqual(result["status"], "conflict")
        self.assertEqual(result["conflicts"][0]["path"], "/best")
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
