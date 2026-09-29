import time
import unittest

import numpy as np

from tillsmith_ai.retriever import analyse, strip_negated
from tillsmith_ai.text import stem, tokenize, with_bigrams
from tillsmith_ai.utils import sub, top_k_indices, wilson_lower
from tests.helpers import engine


class TextTests(unittest.TestCase):
    def test_tokenize_removes_stopwords_and_stems(self):
        self.assertEqual(tokenize("The payments are being declined"), ["pay", "declin"])

    def test_stem_is_symmetric_for_common_forms(self):
        self.assertEqual(stem("migrated"), stem("migration"))
        self.assertEqual(stem("refused"), stem("refusing"))

    def test_bigrams(self):
        self.assertEqual(with_bigrams(["a", "b", "c"]), ["a", "b", "c", "a_b", "b_c"])


class QueryUnderstandingTests(unittest.TestCase):
    def test_negated_symptoms_are_removed(self):
        self.assertNotIn("traffic", strip_negated("orders collapsed but traffic looks normal"))
        self.assertNotIn("fraud", strip_negated("declines are up though fraud has not changed"))

    def test_context_detection(self):
        a = analyse("our shopify plus fashion store on amazon has checkout errors")
        self.assertEqual(a.platform, "Shopify Plus")
        self.assertEqual(a.vertical, "Fashion and Apparel")
        self.assertEqual(a.business_model, "Marketplace Seller")
        self.assertEqual(a.detected[0], "checkout_failure")


class UtilityTests(unittest.TestCase):
    def test_top_k_matches_full_sort(self):
        scores = np.random.default_rng(0).random(5000)
        self.assertEqual(list(top_k_indices(scores, 20)), list(np.flip(np.argsort(scores))[:20]))

    def test_wilson_prefers_more_evidence(self):
        small, large = wilson_lower([9, 160], [10, 200])
        self.assertGreater(large, small)
        self.assertEqual(float(wilson_lower(0, 0)), 0.0)


class IndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eng = engine()

    def test_bm25_and_dense_find_payment_incidents(self):
        idx = self.eng.index
        toks = tokenize("card payments declined by the gateway")
        for scores in (idx.bm25_scores(toks), idx.dense_scores(toks)):
            labels = [idx.label_of("incident_type", r) for r in top_k_indices(scores, 10)]
            self.assertGreaterEqual(labels.count("Payment decline spike"), 6)

    def test_unknown_words_score_zero(self):
        self.assertEqual(float(self.eng.index.bm25_scores(["zzzqqq"]).max()), 0.0)

    def test_retrieval_is_fast(self):
        start = time.perf_counter()
        for _ in range(50):
            self.eng.retriever.retrieve("pages are slow to load on mobile")
        self.assertLess(sub(time.perf_counter(), start) / 50 * 1000, 50.0)

    def test_filters_are_respected(self):
        res = self.eng.retriever.retrieve("checkout errors", filters={"platform": "Magento"})
        labels = {self.eng.index.label_of("platform", r) for r in res.candidates[:50]}
        self.assertEqual(labels, {"Magento"})


if __name__ == "__main__":
    unittest.main()
