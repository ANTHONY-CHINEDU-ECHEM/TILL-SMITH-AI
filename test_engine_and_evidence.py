import unittest

from tillsmith_ai.vocab import EFFICACY_MAP, FIXES, INCIDENTS
from tests.helpers import engine


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eng = engine()

    def test_top_fix_is_strong_for_every_incident(self):
        for key, inc in INCIDENTS.items():
            top = self.eng.recommend(inc["name"])["recommended"][0]["fix"]
            strong = [FIXES[k]["name"] for k in EFFICACY_MAP[key]["strong"]]
            self.assertIn(top, strong, "{} got {}".format(inc["name"], top))

    def test_harmful_fixes_never_recommended(self):
        for key, inc in INCIDENTS.items():
            recommended = {s["fix"] for s in self.eng.recommend(inc["name"])["recommended"]}
            harmful = {FIXES[k]["name"] for k in EFFICACY_MAP[key]["harmful"]}
            self.assertFalse(recommended & harmful)

    def test_reference_class_narrows_only_when_large_enough(self):
        block = self.eng.recommend("Mobile experience regression", vertical="Fashion and Apparel")
        count = int(block["reference_class"].split()[0])
        self.assertGreaterEqual(count, 200 if block["narrowed_by"] else 1)


class AskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eng = engine()
        cls.question = "Genuine customers on our Shopify store say their cards are being refused"
        cls.result = cls.eng.ask(cls.question)

    def test_diagnosis(self):
        self.assertEqual(self.result["diagnosis"][0]["incident"], "Payment decline spike")

    def test_answer_is_grounded(self):
        v = self.result["verification"]
        self.assertTrue(v["passed"])
        self.assertEqual(v["invalid_ids"], [])
        self.assertEqual(v["citation_precision"], 1.0)

    def test_answer_structure(self):
        for heading in ("### Diagnosis", "### Recommended fixes", "### What to expect", "### Confidence"):
            self.assertIn(heading, self.result["answer"])

    def test_cache(self):
        self.assertTrue(self.eng.ask(self.question)["cached"])

    def test_negation_changes_the_diagnosis(self):
        r = self.eng.ask("Orders collapsed right after our last deployment but traffic looks normal")
        self.assertEqual(r["diagnosis"][0]["incident"], "Checkout failure")

    def test_multi_incident_question(self):
        r = self.eng.ask("Our emails land in spam and the website has become painfully slow")
        names = {d["incident"] for d in r["diagnosis"]}
        self.assertTrue({"Email and CRM revenue drop", "Site speed degradation"} & names)

    def test_off_topic_question_is_handled(self):
        r = self.eng.ask("What is the capital of France?")
        self.assertTrue(not r["diagnosis"] or r["confidence"]["label"] == "Low")

    def test_assess(self):
        out = self.eng.assess({"platform": "WooCommerce", "monitoring_maturity": "Basic", "app_integration_count": 40})
        for risk in out["risks"].values():
            self.assertTrue(0.0 < risk["probability"] < 1.0)
        self.assertEqual(len(out["playbook"]), 3)


if __name__ == "__main__":
    unittest.main()
