#!/usr/bin/env python3
"""Cosine similarity smoke test for LangSplat trained artifacts.

Usage:
    python smoke_test.py /path/to/scene/directory [--queries "chair,table,wall"]

Requires: torch, open-clip-torch, numpy (GPU recommended)

Loads the trained autoencoder and compressed 3-dim features, decodes them to
512-dim CLIP vectors, then computes cosine similarity against text queries.
Semantically meaningful embeddings should show higher variance (std) for real
object queries compared to random gibberish.
"""

import argparse
import glob
import os
import sys

import numpy as np


def smoke_test(scene_dir, queries=None):
    """Run cosine similarity smoke test on LangSplat artifacts.

    Args:
        scene_dir: Path to the scene directory (e.g., /path/to/data/figurines)
        queries: List of text queries to test. Defaults to common objects + gibberish.

    Returns:
        bool: True if smoke test passes (>= 2 real queries with std > 0.05)
    """
    if queries is None:
        queries = ["chair", "table", "wall", "floor", "random gibberish xyz"]

    try:
        import torch
    except ImportError:
        print("Error: torch is required. Install with: pip install torch")
        return False

    try:
        import open_clip
    except ImportError:
        print(
            "Error: open-clip-torch is required. Install with: "
            "pip install open-clip-torch"
        )
        return False

    # Determine device
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("WARNING: No GPU detected. Using CPU (slower).")
    except Exception as e:
        print(f"WARNING: GPU detection failed ({e}). Using CPU.")
        device = torch.device("cpu")

    artifacts_dir = os.path.join(scene_dir, "artifacts")

    # --- Load autoencoder ---
    ae_path = None
    ae_candidates = [
        os.path.join(artifacts_dir, "autoencoder.pth"),
        os.path.join(scene_dir, "ae_ckpt", "best_ckpt.pth"),
        os.path.join(scene_dir, "ae_ckpt", "last_ckpt.pth"),
    ]
    for candidate in ae_candidates:
        if os.path.exists(candidate):
            ae_path = candidate
            break

    if ae_path is None:
        print(
            "Error: Autoencoder not found! Checked:\n  "
            + "\n  ".join(ae_candidates)
        )
        return False

    print(f"Loading autoencoder from: {ae_path}")
    try:
        autoencoder = torch.load(ae_path, map_location=device)
        autoencoder.eval()
    except Exception as e:
        print(f"Error loading autoencoder: {e}")
        return False

    # --- Load dim3 features ---
    dim3_candidates = [
        os.path.join(artifacts_dir, "language_feature_dim3", "*_f.npy"),
        os.path.join(scene_dir, "language_feature_dim3", "*_f.npy"),
    ]
    dim3_files = []
    for pattern in dim3_candidates:
        dim3_files = sorted(glob.glob(pattern))
        if dim3_files:
            break

    if not dim3_files:
        print(
            "Error: No dim3 feature files found! Checked:\n  "
            + "\n  ".join(dim3_candidates)
        )
        return False

    print(f"Loading features from: {dim3_files[0]}")
    try:
        features_3d = np.load(dim3_files[0])  # shape: (N_masks, 3)
        features_3d_tensor = torch.tensor(features_3d).to(device).float()
    except Exception as e:
        print(f"Error loading features: {e}")
        return False

    # --- Decode to 512-dim ---
    print("Decoding 3-dim features to 512-dim...")
    try:
        with torch.no_grad():
            features_512 = autoencoder.decode(features_3d_tensor)
            features_512 = features_512 / features_512.norm(
                dim=-1, keepdim=True
            )
    except Exception as e:
        print(f"Error decoding features: {e}")
        return False

    # --- Encode text queries with CLIP ---
    print("Loading CLIP ViT-B-16 model...")
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="laion2b_s34b_b88k"
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-16")
        model.eval().to(device)
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        return False

    print("Encoding text queries...")
    try:
        text_tokens = tokenizer(queries).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(
                dim=-1, keepdim=True
            )
    except Exception as e:
        print(f"Error encoding text: {e}")
        return False

    # --- Compute cosine similarity ---
    print("\n" + "=" * 60)
    print("SMOKE TEST: Cosine Similarity")
    print("=" * 60)

    pass_count = 0
    for i, text in enumerate(queries):
        try:
            sims = (features_512 @ text_features[i]).cpu().numpy()
            std = sims.std()
            print(
                f"  '{text}': min={sims.min():.3f}, max={sims.max():.3f}, "
                f"mean={sims.mean():.3f}, std={std:.3f}"
            )
            if text != "random gibberish xyz" and std > 0.05:
                pass_count += 1
        except Exception as e:
            print(f"  '{text}': ERROR -- {e}")

    non_gibberish = [q for q in queries if q != "random gibberish xyz"]
    total_real = len(non_gibberish)

    print(f"\nPASS criteria: std > 0.05 for real object queries")
    print(f"Result: {pass_count}/{total_real} real queries passed")

    passed = pass_count >= 2
    if passed:
        print(
            "\n*** SMOKE TEST PASSED -- embeddings are semantically meaningful ***"
        )
    else:
        print(
            "\n*** SMOKE TEST FAILED -- embeddings may be noise. "
            "Check training. ***"
        )

    return passed


def main():
    """CLI entry point for smoke test."""
    parser = argparse.ArgumentParser(
        description=(
            "Cosine similarity smoke test for LangSplat trained artifacts."
        )
    )
    parser.add_argument(
        "scene_dir",
        help="Path to the scene directory (e.g., /path/to/data/figurines)",
    )
    parser.add_argument(
        "--queries",
        type=str,
        default=None,
        help=(
            "Comma-separated query strings "
            '(default: "chair,table,wall,floor,random gibberish xyz")'
        ),
    )
    args = parser.parse_args()

    if not os.path.isdir(args.scene_dir):
        print(f"Error: {args.scene_dir} is not a valid directory")
        sys.exit(1)

    queries = None
    if args.queries:
        queries = [q.strip() for q in args.queries.split(",")]

    passed = smoke_test(args.scene_dir, queries)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
