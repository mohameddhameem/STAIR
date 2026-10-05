"""Checks for feature leakage and stage composition, using synthetic data only."""
import unittest
from baselines import composed, numeric_features, text_features


class BaselineChecks(unittest.TestCase):
    def test_features_ignore_supervision(self):
        row = {"question": "Who wrote this?", "pool_text": "[1] A: A short passage",
               "evidence_W": "weak refs", "evidence_S": "strong refs"}
        poisoned = dict(row, answers=["secret"], supporting_titles=["secret"],
                        ret_label="S", qa_label_W="S", em_lenient=[1, 0, 1, 0],
                        overlap_jaccard=0.9, R=[5, 8, 3, 4], gold_in_pool=True)
        for stage in ["ret", "qa_W", "qa_S"]:
            self.assertEqual(text_features(row, stage), text_features(poisoned, stage))
            self.assertEqual(numeric_features(row, stage), numeric_features(poisoned, stage))

    def test_only_selected_qa_branch_is_used(self):
        rows = {0: {"em_lenient": [1, 0, 0, 0], "em_strict": [1, 0, 0, 0]},
                1: {"em_lenient": [0, 0, 0, 1], "em_strict": [0, 0, 0, 0]}}
        r = composed(rows, [0, 1], ["W", "S"], ["W", "W"], ["S", "S"])
        self.assertEqual(r["picks"], {"WW": 1, "SS": 1})
        self.assertEqual(r["lenient_accuracy"], 1)
        self.assertEqual(r["strict_accuracy"], 0.5)
        self.assertAlmostEqual(r["mean_pipeline_cost"], 5.7)

    def test_qa_features_follow_branch(self):
        row = {"question": "Question", "pool_text": "pool", "evidence_W": "W ONLY", "evidence_S": "S ONLY"}
        self.assertNotIn("S ONLY", text_features(row, "qa_W"))
        self.assertNotIn("W ONLY", text_features(row, "qa_S"))


if __name__ == "__main__":
    unittest.main()
