import json
import urllib.request


def main() -> None:
    """Fetch and print the current latency summary from the demo server."""
    with urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))

    print("Realtime Speech Metrics")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
