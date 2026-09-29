import os
import unittest

import numpy as np
from fastapi.testclient import TestClient

from tillsmith_ai.api import app
from tillsmith_ai.config import settings
from tillsmith_ai.llm import generate_answer, verify
from tillsmith_ai.risk_model import auc
from tests.helpers import engine


class FabricatingProvider:
    name = "fake"

    def generate(self, question, result, pack):
        return "Run a flash sale. It worked in 97% of cases [INC999999]."


class BrokenProvider:
    name = "broken"

    def generate(self, question, result, pack):
        raise RuntimeError("network down")


class VerifierTests(unittest.TestCase):
    def test_detects_invented_ids_and_numbers(self):
        check = verify("See [INC000001] and [INC123456]; success 88%.", "evidence INC000001 71%", ["INC000001"])
        self.assertEqual(check["invalid_ids"], ["INC123456"])
        self.assertEqual(check["unsupported_percentages"], [88.0])
        self.assertFalse(check["passed"])

    def test_strict_mode_replaces_fabricated_answer(self):
        result = engine().ask("Our checkout throws errors after the release", provider="extractive")
        out = generate_answer("q", result, settings, provider=FabricatingProvider(), strict=True)
        self.assertEqual(out["provider"], "extractive")
        self.assertTrue(out["verification"]["passed"])
        self.assertTrue(out["notes"])

    def test_provider_failure_falls_back(self):
        result = engine().ask("Our checkout throws errors after the release", provider="extractive")
        self.assertEqual(generate_answer("q", result, settings, provider=BrokenProvider())["provider"], "extractive")

    def test_auc(self):
        self.assertAlmostEqual(auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.3, 0.4])), 1.0)


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.pop("TILLSMITH_API_KEY", None)
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_health(self):
        body = self.client.get("/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertGreaterEqual(body["incidents"], 16000)

    def test_ask(self):
        r = self.client.post("/ask", json={"question": "Our newsletters land in spam", "k": 5})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["diagnosis"][0]["incident"], "Email and CRM revenue drop")

    def test_validation(self):
        self.assertEqual(self.client.post("/ask", json={"question": ""}).status_code, 422)
        self.assertEqual(self.client.post("/search", json={"query": "x y", "mode": "magic"}).status_code, 422)

    def test_search_recommend_assess_incident(self):
        self.assertEqual(len(self.client.post("/search", json={"query": "bot traffic", "k": 4}).json()["results"]), 4)
        self.assertEqual(self.client.post("/recommend", json={"incident": "Checkout failure"}).status_code, 200)
        self.assertEqual(self.client.post("/recommend", json={"incident": "Unknown"}).status_code, 404)
        self.assertEqual(self.client.post("/assess", json={"app_integration_count": 30}).status_code, 200)
        self.assertEqual(self.client.get("/incidents/INC000010").json()["incident_id"], "INC000010")
        self.assertEqual(self.client.get("/incidents/INC999999").status_code, 404)

    def test_bearer_auth(self):
        os.environ["TILLSMITH_API_KEY"] = "secret"
        try:
            self.assertEqual(self.client.get("/stats").status_code, 401)
            self.assertEqual(self.client.get("/stats", headers={"Authorization": "Bearer secret"}).status_code, 200)
            self.assertEqual(self.client.get("/health").status_code, 200)
        finally:
            os.environ.pop("TILLSMITH_API_KEY", None)


if __name__ == "__main__":
    unittest.main()
