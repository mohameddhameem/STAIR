import unittest
from full_pipeline_eval import passages, selected_evidence


class RetrievalChecks(unittest.TestCase):
    def test_preserves_full_selected_passages(self):
        pool = "[1] First: full body\ncontinued text\n[2] Second: other body"
        self.assertEqual(set(passages(pool)), {1, 2})
        ids, evidence = selected_evidence("Question?", pool, "[2], [1], [2]")
        self.assertEqual(ids, [2, 1])
        self.assertIn("continued text", evidence)
        self.assertLess(evidence.index("Second:"), evidence.index("First:"))

    def test_invalid_selection_is_not_replaced_by_gold(self):
        with self.assertRaises(ValueError):
            selected_evidence("Question?", "[1] Only passage", "[100]")
        with self.assertRaises(ValueError):
            selected_evidence("Question?", "[1] Only passage", "")


if __name__ == "__main__":
    unittest.main()
