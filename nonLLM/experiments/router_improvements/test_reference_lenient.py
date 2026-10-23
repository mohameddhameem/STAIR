import unittest
from reference_lenient import score


class ReferenceLenientTests(unittest.TestCase):
    def test_forward_containment_differs_from_exact(self):
        self.assertEqual(score('film director',['director']),(0,1))
        self.assertEqual(score('World War I and World War II',['World War II']),(0,1))

    def test_reverse_containment_and_partial_words_rejected(self):
        self.assertEqual(score('1936',['7 January 1936']),(0,0))
        self.assertEqual(score('Newcastle',['City of Newcastle']),(0,0))
        self.assertEqual(score('board games',['board game']),(0,0))

    def test_unicode_punctuation_and_numeric_boundaries(self):
        self.assertEqual(score('Armenian-American',['Armenian–American']),(0,1))
        self.assertEqual(score('2018-19 UEFA Europa League group stage',['2018–19 UEFA Europa League']),(0,1))
        self.assertEqual(score('c. 1700–500 BC',['500 BC']),(0,0))

    def test_boolean_conflict_and_explanations(self):
        self.assertEqual(score('Yes\nNo',['yes']),(0,0))
        self.assertEqual(score('Yes, Virgilia is found in Africa.',['yes']),(0,1))

    def test_reference_quantifier_examples_are_accepted(self):
        self.assertEqual(score('more than 27 languages',['27 languages']),(0,1))
        self.assertEqual(score('over 9,000 employees',['9,000']),(0,1))

    def test_normalized_exact_and_aliases(self):
        self.assertEqual(score('The Eiffel Tower!', ['Eiffel Tower']),(1,1))
        self.assertEqual(score('Paris', ['London','Paris']),(1,1))
        self.assertEqual(score('Paris','Paris'),(1,1))


if __name__=='__main__':unittest.main()
