"""Example: Query ClipForge Local Worker health and hardware capabilities."""
import json
import urllib.request

WORKER_URL = "http://127.0.0.1:8765"


def check_worker_health() -> dict:
    with urllib.request.urlopen(f"{WORKER_URL}/health", timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":
    try:
        status = check_worker_health()
        print("ClipForge Worker Status:", json.dumps(status, indent=2))
    except Exception as exc:
        print(f"Worker not reachable at {WORKER_URL}: {exc}")
