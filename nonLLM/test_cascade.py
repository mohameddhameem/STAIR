"""Check cascade composition and threshold search against direct enumeration."""
import itertools
import unittest

import numpy as np

from cascade import compose, costs, sweep, view


class CascadeTests(unittest.TestCase):
    def test_search_matches_brute_force_with_ties(self):
        em = np.array([[1, 0, 1, 1], [0, 1, 0, 1], [0, 0, 1, 1],
                       [1, 1, 0, 0], [0, 1, 1, 1]])
        pr = np.array([.2, .2, .6, .8, .8])
        qw = np.array([.3, .3, .5, .9, .5])
        qs = np.array([.7, .1, .7, .4, .1])
        prices, _, _ = costs(None)
        lock, _ = sweep(em, pr, qw, qs, prices)
        feasible = []
        for tr, tq in itertools.product(np.r_[np.unique(pr), np.inf],
                                        np.r_[np.unique(np.r_[qw, qs]), np.inf]):
            r = pr < tr
            q = np.where(r, qs, qw) < tq
            cell = r.astype(int) * 2 + q.astype(int)
            acc = em[np.arange(len(em)), cell].mean()
            if acc >= .95 * em[:, 3].mean() - 1e-12:
                feasible.append((prices[cell].mean(), -acc))
        self.assertEqual((lock["cost"], -lock["accuracy"]), min(feasible))

    def test_all_four_corners_include_sunk_weak_costs(self):
        prices, direct_ss, _ = costs(None)
        self.assertAlmostEqual(direct_ss, 9.4)
        self.assertAlmostEqual(prices[3], 11.4)
        grid = [{"em_lenient": [1, 0, 1, 0], "em_strict": [0, 0, 1, 0]}]
        for r, q in itertools.product((False, True), repeat=2):
            lock = {"theta_ret": np.inf if r else -np.inf,
                    "theta_qa": np.inf if q else -np.inf}
            result, cell = compose(grid, np.array([.5]), np.array([.5]),
                                   np.array([.5]), lock, prices, direct_ss)
            j = int(r) * 2 + int(q)
            self.assertEqual(cell[0], j)
            self.assertAlmostEqual(result["mean_cost"], prices[j])
            self.assertEqual(result["lenient_accuracy"], grid[0]["em_lenient"][j])

    def test_features_ignore_gold_and_outcomes(self):
        row = {"question": "Who wrote this?", "evidence_W": "[1] author", "evidence_S": "[2] book"}
        expected = view(row, "qa", "W", "Alice")
        row.update(answers=["secret"], em_lenient=[1, 1, 1, 1], qa_label_W="S", ret_label="S")
        self.assertEqual(view(row, "qa", "W", "Alice"), expected)
        self.assertNotEqual(view(row, "qa", "W", "Bob"), expected)


if __name__ == "__main__":
    unittest.main()
