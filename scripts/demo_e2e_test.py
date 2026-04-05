#!/usr/bin/env python3
"""End-to-end demo validation script for SpatialMind three-beat demo.

Validates all three demo beats against the running server:
  - Beat 1: Object highlighting ("chairs") via CLIP similarity
  - Beat 2: Spatial reasoning ("what's on the tables?") via query walker
  - Beat 3: Unrehearsed query ("where are the windows?") via query walker

Usage:
    python scripts/demo_e2e_test.py [--base-url http://localhost:8001]
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def make_request(url, data=None, timeout=30):
    """Send a GET or POST request and return (response_dict, elapsed_ms).

    Args:
        url: The URL to request.
        data: If provided, send as JSON POST body. Otherwise GET.
        timeout: Request timeout in seconds.

    Returns:
        Tuple of (parsed JSON dict, elapsed milliseconds).

    Raises:
        Exception: On network or parsing errors.
    """
    headers = {"Content-Type": "application/json"}
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    else:
        req = urllib.request.Request(url, headers=headers)

    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    elapsed_ms = (time.monotonic() - start) * 1000

    return json.loads(raw), elapsed_ms


def check_pipeline(base_url):
    """Pre-check: verify pipeline readiness via /api/health.

    Args:
        base_url: Server base URL.

    Returns:
        True if pipeline is ready, False otherwise.
    """
    url = f"{base_url}/api/health"
    print(f"\n{'='*60}")
    print("PRE-CHECK: Pipeline Readiness")
    print(f"{'='*60}")

    try:
        data, ms = make_request(url)
    except Exception as exc:
        print(f"FAIL: Could not reach {url}: {exc}")
        print("Is the server running? Start with: python scripts/demo_launch.py")
        return False

    status = data.get("status", "unknown")
    pipeline_ready = data.get("pipeline_ready", False)
    gaussian_count = data.get("gaussian_count", 0)
    scene_source = data.get("scene_source", "unknown")

    print(f"Pipeline: {gaussian_count} Gaussians, scene_source={scene_source}")
    print(f"Response time: {ms:.0f}ms")

    if status != "ok":
        print(f"FAIL: Health status is '{status}', expected 'ok'")
        return False

    if not pipeline_ready:
        missing = []
        if not data.get("ply_loaded", False):
            missing.append("PLY file not loaded")
        if not data.get("clip_ready", False):
            missing.append("CLIP encoder not ready")
        if not data.get("scene_graph_ready", False):
            missing.append("Scene graph not built")
        print(f"FAIL: Pipeline not ready. Missing: {', '.join(missing)}")
        return False

    print("PASS: Pipeline ready")
    return True


def beat_1_highlight(base_url):
    """Beat 1: Object highlighting -- POST 'chairs' to /api/clip/highlight.

    Args:
        base_url: Server base URL.

    Returns:
        True if beat passes, False otherwise.
    """
    url = f"{base_url}/api/clip/highlight"
    print(f"\n{'='*60}")
    print("BEAT 1: Object Highlighting ('chairs')")
    print(f"{'='*60}")

    try:
        data, ms = make_request(url, {"text": "chairs", "k": 100})
    except Exception as exc:
        print(f"FAIL: Request failed: {exc}")
        return False

    indices = data.get("indices", [])
    scores = data.get("scores", [])

    if not isinstance(indices, list) or len(indices) == 0:
        print(f"FAIL: Expected non-empty 'indices' list, got {type(indices).__name__} with {len(indices) if isinstance(indices, list) else 'N/A'} items")
        return False

    if not isinstance(scores, list) or len(scores) == 0:
        print(f"FAIL: Expected non-empty 'scores' list, got {type(scores).__name__} with {len(scores) if isinstance(scores, list) else 'N/A'} items")
        return False

    print(f"Beat 1 PASS: {len(indices)} Gaussians highlighted for 'chairs'")
    print(f"Response time: {ms:.0f}ms")
    return True


def beat_2_spatial(base_url):
    """Beat 2: Spatial reasoning -- POST "what's on the tables?" to /api/query.

    Args:
        base_url: Server base URL.

    Returns:
        True if beat passes, False otherwise.
    """
    url = f"{base_url}/api/query"
    print(f"\n{'='*60}")
    print("BEAT 2: Spatial Reasoning ('what's on the tables?')")
    print(f"{'='*60}")

    try:
        data, ms = make_request(
            url,
            {"query": "what's on the tables?", "session_id": "e2e-test"},
        )
    except Exception as exc:
        print(f"FAIL: Request failed: {exc}")
        return False

    answer = data.get("answer", "") or data.get("description", "")

    if not isinstance(answer, str) or len(answer) <= 10:
        print(f"FAIL: Expected answer string >10 chars, got {len(answer) if isinstance(answer, str) else 'non-string'} chars")
        return False

    preview = answer[:80]
    print(f"Beat 2 PASS: Got answer ({len(answer)} chars): '{preview}...'")
    print(f"Response time: {ms:.0f}ms")
    return True


def beat_3_unrehearsed(base_url):
    """Beat 3: Unrehearsed query -- POST "where are the windows?" to /api/query.

    Args:
        base_url: Server base URL.

    Returns:
        True if beat passes, False otherwise.
    """
    url = f"{base_url}/api/query"
    print(f"\n{'='*60}")
    print("BEAT 3: Unrehearsed Query ('where are the windows?')")
    print(f"{'='*60}")

    try:
        data, ms = make_request(
            url,
            {"query": "where are the windows?", "session_id": "e2e-test"},
        )
    except Exception as exc:
        print(f"FAIL: Request failed: {exc}")
        return False

    answer = data.get("answer", "") or data.get("description", "")

    if not isinstance(answer, str) or len(answer) <= 10:
        print(f"FAIL: Expected answer string >10 chars, got {len(answer) if isinstance(answer, str) else 'non-string'} chars")
        return False

    preview = answer[:80]
    print(f"Beat 3 PASS: Got answer ({len(answer)} chars): '{preview}...'")
    print(f"Response time: {ms:.0f}ms")
    return True


def main():
    """Run the full three-beat E2E demo validation."""
    parser = argparse.ArgumentParser(
        description="SpatialMind E2E demo validation script",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8001",
        help="Base URL of the SpatialMind server (default: http://localhost:8001)",
    )
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    print("SpatialMind E2E Demo Validation")
    print(f"Server: {base_url}")

    # Pre-check pipeline readiness
    if not check_pipeline(base_url):
        print(f"\n{'='*60}")
        print("DEMO NOT READY - pipeline not ready")
        print(f"{'='*60}")
        sys.exit(1)

    # Run beats
    results = {
        "Beat 1": beat_1_highlight(base_url),
        "Beat 2": beat_2_spatial(base_url),
        "Beat 3": beat_3_unrehearsed(base_url),
    }

    # Summary
    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")

    print()
    if failed == 0:
        print(f"DEMO READY - all {passed} beats passed")
        sys.exit(0)
    else:
        failed_names = [n for n, v in results.items() if not v]
        print(f"DEMO NOT READY - {failed} beat(s) failed: {', '.join(failed_names)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
