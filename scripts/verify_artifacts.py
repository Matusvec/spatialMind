#!/usr/bin/env python3
"""Verify LangSplat training artifacts exist and are valid.

Usage:
    python verify_artifacts.py /path/to/scene/directory

Checks the consolidated artifacts/ directory first (co-located artifacts).
Falls back to checking scattered source locations if artifacts/ doesn't exist.

Checks:
    1. point_cloud.ply exists and is > 10 MB
    2. language_feature_dim3/*.npy files exist and are > 1 KB each
    3. autoencoder checkpoint exists and is > 100 KB
    4. PLY file contains expected Gaussian properties
"""

import argparse
import glob
import os
import sys


def verify_artifacts(scene_dir):
    """Verify LangSplat training artifacts in the given scene directory.

    Args:
        scene_dir: Path to the scene directory (e.g., /path/to/data/figurines)

    Returns:
        bool: True if all checks pass, False otherwise
    """
    results = []
    artifacts_dir = os.path.join(scene_dir, "artifacts")
    scene_name = os.path.basename(os.path.normpath(scene_dir))

    # Determine path mode
    if os.path.isdir(artifacts_dir):
        print("Checking consolidated artifacts/ directory")
        mode = "consolidated"
    else:
        print("Checking scattered source locations (artifacts/ not found)")
        mode = "scattered"

    print("=" * 60)

    # --- Check 1: point_cloud.ply ---
    if mode == "consolidated":
        ply_path = os.path.join(artifacts_dir, "point_cloud.ply")
    else:
        ply_path = os.path.join(
            scene_dir,
            "output",
            scene_name,
            "point_cloud",
            "iteration_30000",
            "point_cloud.ply",
        )

    ply_ok = False
    if os.path.exists(ply_path):
        size_mb = os.path.getsize(ply_path) / 1e6
        if size_mb > 10:
            print(f"  [PASS] point_cloud.ply: {size_mb:.1f} MB (> 10 MB)")
            ply_ok = True

            # Check PLY structure with plyfile
            try:
                from plyfile import PlyData

                ply = PlyData.read(ply_path)
                vertex = ply["vertex"]
                props = [p.name for p in vertex.properties]
                has_xyz = all(p in props for p in ["x", "y", "z"])
                if has_xyz:
                    print(
                        f"  [PASS] PLY structure: {vertex.count} vertices, "
                        f"has xyz properties, {len(props)} total properties"
                    )
                else:
                    print(
                        f"  [FAIL] PLY structure: missing xyz properties "
                        f"(found: {props[:10]}...)"
                    )
                    ply_ok = False
            except ImportError:
                print(
                    "  [WARN] plyfile not installed -- skipping PLY structure check"
                )
            except Exception as e:
                print(f"  [FAIL] PLY structure: error reading file -- {e}")
                ply_ok = False
        else:
            print(
                f"  [FAIL] point_cloud.ply: {size_mb:.1f} MB (expected > 10 MB)"
            )
    else:
        print(f"  [FAIL] point_cloud.ply: NOT FOUND at {ply_path}")
    results.append(("point_cloud.ply", ply_ok))

    # --- Check 2: autoencoder checkpoint ---
    if mode == "consolidated":
        ae_path = os.path.join(artifacts_dir, "autoencoder.pth")
    else:
        ae_path = os.path.join(scene_dir, "ae_ckpt", "best_ckpt.pth")
        if not os.path.exists(ae_path):
            ae_path = os.path.join(scene_dir, "ae_ckpt", "last_ckpt.pth")

    ae_ok = False
    if os.path.exists(ae_path):
        size_kb = os.path.getsize(ae_path) / 1e3
        if size_kb > 100:
            print(f"  [PASS] autoencoder: {size_kb:.0f} KB (> 100 KB)")
            ae_ok = True
        else:
            print(
                f"  [FAIL] autoencoder: {size_kb:.0f} KB (expected > 100 KB)"
            )
    else:
        print(f"  [FAIL] autoencoder: NOT FOUND at {ae_path}")
    results.append(("autoencoder", ae_ok))

    # --- Check 3: language_feature_dim3 .npy files ---
    if mode == "consolidated":
        dim3_dir = os.path.join(artifacts_dir, "language_feature_dim3")
    else:
        dim3_dir = os.path.join(scene_dir, "language_feature_dim3")

    npy_files = glob.glob(os.path.join(dim3_dir, "*_f.npy"))
    npy_ok = False
    if npy_files:
        small_files = [
            f for f in npy_files if os.path.getsize(f) <= 1024
        ]
        if not small_files:
            total_mb = sum(os.path.getsize(f) for f in npy_files) / 1e6
            print(
                f"  [PASS] language_feature_dim3: {len(npy_files)} files, "
                f"{total_mb:.1f} MB total (all > 1 KB)"
            )
            npy_ok = True
        else:
            print(
                f"  [FAIL] language_feature_dim3: {len(small_files)} files "
                f"are <= 1 KB"
            )
    else:
        print(f"  [FAIL] language_feature_dim3: NO .npy files in {dim3_dir}")
    results.append(("language_feature_dim3", npy_ok))

    # --- Summary ---
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n*** ALL CHECKS PASSED ***")
    else:
        print("\n*** SOME CHECKS FAILED ***")

    return all_pass


def main():
    """CLI entry point for artifact verification."""
    parser = argparse.ArgumentParser(
        description="Verify LangSplat training artifacts exist and are valid."
    )
    parser.add_argument(
        "scene_dir",
        help="Path to the scene directory (e.g., /path/to/data/figurines)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.scene_dir):
        print(f"Error: {args.scene_dir} is not a valid directory")
        sys.exit(1)

    passed = verify_artifacts(args.scene_dir)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
