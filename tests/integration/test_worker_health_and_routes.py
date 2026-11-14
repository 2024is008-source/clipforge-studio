"""Integration verification for ClipForge Local Worker FastAPI routes."""
import unittest
from fastapi.testclient import TestClient
from local_worker.main import app


class TestWorkerHealthAndRoutes(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint_returns_structured_status(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertTrue(payload.get("ok"))
        self.assertIn("nativeRenderer", payload)
        self.assertIn("whisperInstalled", payload)

    def test_protected_routes_enforce_validation_with_token(self):
        res = self.client.post(
            "/v1/render/overlays",
            json={"fps": 30, "frameCount": 0, "width": 1080, "height": 1920},
            headers={"X-ClipForge-Token": "auto-local"},
        )
        self.assertEqual(res.status_code, 400)


if __name__ == "__main__":
    unittest.main()
