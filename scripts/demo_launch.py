#!/usr/bin/env python3
"""SpatialMind one-command demo launcher.

Starts the backend server with automatic scene fallback detection,
polls for pipeline readiness, and prints three-beat demo instructions.

Usage:
    python scripts/demo_launch.py
    python scripts/demo_launch.py --port 8001 --scene-dir ./data/scene
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


BANNER = """
=============================================
       SpatialMind Demo Launcher
=============================================

  An AI that understands 3D space as a
  structured graph, queryable with natural
  language, powered by semantically-embedded
  Gaussians.

  Three-Beat Demo:
    1. Semantic Search  -- "chairs" lights up
    2. Spatial QA       -- "what's on the tables?"
    3. Open Query       -- any unrehearsed query

=============================================
"""

DEMO_INSTRUCTIONS = """
=== DEMO READY ===

Frontend: http://localhost:5173
Backend:  http://localhost:{port}
Health:   http://localhost:{port}/api/health

Three-Beat Demo Sequence:
  Beat 1: Click "Begin Exploration", then type "chairs" in chat
          -> Chair Gaussians light up in <5s
  Beat 2: Type "what's on the tables?"
          -> Natural language answer with real objects
  Beat 3: Type any unrehearsed query (e.g., "where are the windows?")
          -> Spatially correct response

Press Ctrl+C to stop.
"""


def check_scene_availability(scene_dir, fallback_dir):
    """Check which scene artifacts are available.

    Args:
        scene_dir: Primary scene directory path.
        fallback_dir: Fallback scene directory path.

    Returns:
        Tuple of (scene_type, scene_path) or exits on failure.
    """
    live_ply = os.path.join(scene_dir, "artifacts", "point_cloud.ply")
    fallback_ply = os.path.join(fallback_dir, "artifacts", "point_cloud.ply")

    if os.path.exists(live_ply):
        print("[LIVE] Using JacHacks venue scene")
        print(f"  PLY: {live_ply}")
        return "live", scene_dir

    if os.path.exists(fallback_ply):
        print("[FALLBACK] Using pre-trained scene")
        print(f"  PLY: {fallback_ply}")
        return "fallback", fallback_dir

    print("ERROR: No scene data found.")
    print(f"  Checked live:     {live_ply}")
    print(f"  Checked fallback: {fallback_ply}")
    print()
    print(
        "Place PLY artifacts in ./data/scene/artifacts/ "
        "or ./data/fallback_scene/artifacts/"
    )
    sys.exit(1)


def poll_health(port, timeout=60, interval=2):
    """Poll /api/health until pipeline_ready is True.

    Args:
        port: Server port number.
        timeout: Maximum seconds to wait.
        interval: Seconds between poll attempts.

    Returns:
        True if pipeline became ready, False on timeout.
    """
    url = f"http://localhost:{port}/api/health"
    deadline = time.time() + timeout
    attempt = 0

    while time.time() < deadline:
        attempt += 1
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                pipeline_ready = data.get("pipeline_ready", False)
                scene_source = data.get("scene_source", "unknown")

                if pipeline_ready:
                    print(f"  Server ready! Pipeline: OK (source: {scene_source})")
                    return True

                print(
                    f"  [{attempt}] Waiting... "
                    f"ply={data.get('ply_loaded')}, "
                    f"clip={data.get('clip_ready')}, "
                    f"graph={data.get('scene_graph_ready')}"
                )
        except (urllib.error.URLError, ConnectionError, OSError):
            print(f"  [{attempt}] Waiting for server...")

        time.sleep(interval)

    print("ERROR: Server did not become ready within {timeout}s")
    return False


def start_server(port, scene_dir):
    """Start the uvicorn server as a subprocess.

    Args:
        port: Port to bind the server to.
        scene_dir: Scene directory to pass via SCENE_DIR env var.

    Returns:
        subprocess.Popen instance.
    """
    env = os.environ.copy()
    env["SCENE_DIR"] = scene_dir

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "server.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
    ]

    print(f"Starting server on port {port}...")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc


def main():
    """Run the SpatialMind demo launcher."""
    parser = argparse.ArgumentParser(
        description="SpatialMind one-command demo launcher"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Server port (default: 8001)",
    )
    parser.add_argument(
        "--scene-dir",
        type=str,
        default="./data/scene",
        help="Primary scene directory (default: ./data/scene)",
    )
    parser.add_argument(
        "--fallback-dir",
        type=str,
        default="./data/fallback_scene",
        help="Fallback scene directory (default: ./data/fallback_scene)",
    )
    args = parser.parse_args()

    print(BANNER)

    # 1. Check scene availability
    print("Checking scene availability...")
    scene_type, scene_path = check_scene_availability(
        args.scene_dir, args.fallback_dir
    )
    print()

    # 2. Start server
    server_proc = start_server(args.port, scene_path)

    try:
        # 3. Poll for readiness
        print("Polling for pipeline readiness...")
        ready = poll_health(args.port, timeout=60, interval=2)

        if not ready:
            print("Server failed to reach pipeline_ready state.")
            print("Check server logs for errors.")
            server_proc.terminate()
            sys.exit(1)

        # 4. Print demo instructions
        print(DEMO_INSTRUCTIONS.format(port=args.port))

        # 5. Wait for Ctrl+C
        print("Server running. Press Ctrl+C to stop.")
        server_proc.wait()

    except KeyboardInterrupt:
        print("\nShutting down...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        print("Server stopped.")


if __name__ == "__main__":
    main()
