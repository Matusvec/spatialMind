#!/usr/bin/env python3
"""Convert 3DGS PLY to .splat format — antimatter15 compatible.

Vectorized version of https://github.com/antimatter15/splat/blob/main/convert.py
Same output format, ~100x faster via numpy batch operations.

.splat = 32 bytes per Gaussian:
  position:   3 x float32 (12 bytes)
  scale:      3 x float32 (12 bytes)
  color+alpha: 4 x uint8   (4 bytes)
  rotation:   4 x uint8   (4 bytes)

Usage: python scripts/ply_to_splat.py input.ply -o output.splat [--no-sort]
"""

import argparse
import numpy as np
from plyfile import PlyData


def process_ply_to_splat(ply_file_path: str, *, sort_by_importance: bool = True) -> bytes:
    print(f"Reading {ply_file_path}...")
    plydata = PlyData.read(ply_file_path)
    vert = plydata["vertex"]
    n = len(vert)
    print(f"  {n:,} Gaussians")

    if sort_by_importance:
        # Sort by importance (same as antimatter15): scale * opacity descending
        print("  Sorting by importance...")
        sorted_indices = np.argsort(
            -np.exp(vert["scale_0"] + vert["scale_1"] + vert["scale_2"])
            / (1 + np.exp(-vert["opacity"]))
        )
    else:
        print("  Preserving original PLY order...")
        sorted_indices = np.arange(n)

    # Extract all fields as numpy arrays in sorted order
    print("  Extracting fields...")
    x = vert["x"][sorted_indices].astype(np.float32)
    y = vert["y"][sorted_indices].astype(np.float32)
    z = vert["z"][sorted_indices].astype(np.float32)

    scale_0 = np.exp(vert["scale_0"][sorted_indices]).astype(np.float32)
    scale_1 = np.exp(vert["scale_1"][sorted_indices]).astype(np.float32)
    scale_2 = np.exp(vert["scale_2"][sorted_indices]).astype(np.float32)

    SH_C0 = 0.28209479177387814
    r = np.clip((0.5 + SH_C0 * vert["f_dc_0"][sorted_indices]) * 255, 0, 255).astype(np.uint8)
    g = np.clip((0.5 + SH_C0 * vert["f_dc_1"][sorted_indices]) * 255, 0, 255).astype(np.uint8)
    b = np.clip((0.5 + SH_C0 * vert["f_dc_2"][sorted_indices]) * 255, 0, 255).astype(np.uint8)
    a = np.clip((1 / (1 + np.exp(-vert["opacity"][sorted_indices]))) * 255, 0, 255).astype(np.uint8)

    rot_0 = vert["rot_0"][sorted_indices].astype(np.float32)
    rot_1 = vert["rot_1"][sorted_indices].astype(np.float32)
    rot_2 = vert["rot_2"][sorted_indices].astype(np.float32)
    rot_3 = vert["rot_3"][sorted_indices].astype(np.float32)

    # Normalize quaternions
    qnorm = np.sqrt(rot_0**2 + rot_1**2 + rot_2**2 + rot_3**2)
    qnorm = np.maximum(qnorm, 1e-10)
    rot_0 /= qnorm; rot_1 /= qnorm; rot_2 /= qnorm; rot_3 /= qnorm

    # Quantize rotation to uint8: [-1,1] -> [0,255]
    rot_0_u8 = np.clip(rot_0 * 128 + 128, 0, 255).astype(np.uint8)
    rot_1_u8 = np.clip(rot_1 * 128 + 128, 0, 255).astype(np.uint8)
    rot_2_u8 = np.clip(rot_2 * 128 + 128, 0, 255).astype(np.uint8)
    rot_3_u8 = np.clip(rot_3 * 128 + 128, 0, 255).astype(np.uint8)

    # Build output buffer: 32 bytes per Gaussian
    print("  Building .splat buffer...")
    # Interleave: [x,y,z, sx,sy,sz, r,g,b,a, q0,q1,q2,q3] per row
    # Position block (12 bytes)
    pos = np.column_stack([x, y, z])  # (n, 3) float32
    # Scale block (12 bytes)
    scales = np.column_stack([scale_0, scale_1, scale_2])  # (n, 3) float32
    # Color block (4 bytes)
    rgba = np.column_stack([r, g, b, a])  # (n, 4) uint8
    # Rotation block (4 bytes)
    rot = np.column_stack([rot_0_u8, rot_1_u8, rot_2_u8, rot_3_u8])  # (n, 4) uint8

    # Pack into 32-byte records
    out = np.empty(n, dtype=np.dtype([
        ('pos', np.float32, 3),
        ('scale', np.float32, 3),
        ('rgba', np.uint8, 4),
        ('rot', np.uint8, 4),
    ]))
    out['pos'] = pos
    out['scale'] = scales
    out['rgba'] = rgba
    out['rot'] = rot

    result = out.tobytes()
    print(f"  Output: {len(result) / 1024 / 1024:.1f} MB")
    return result


def main():
    parser = argparse.ArgumentParser(description="Convert PLY to .splat (antimatter15 format)")
    parser.add_argument("input", help="Input .ply file")
    parser.add_argument("-o", "--output", default=None, help="Output .splat file")
    parser.add_argument(
        "--no-sort",
        action="store_true",
        help="Preserve original PLY Gaussian order instead of sorting by importance.",
    )
    args = parser.parse_args()

    output = args.output or args.input.replace(".ply", ".splat")
    splat_data = process_ply_to_splat(args.input, sort_by_importance=not args.no_sort)

    print(f"Writing {output}...")
    with open(output, "wb") as f:
        f.write(splat_data)

    import os
    in_size = os.path.getsize(args.input) / 1024 / 1024
    out_size = len(splat_data) / 1024 / 1024
    print(f"Done! {in_size:.0f} MB → {out_size:.0f} MB ({in_size/out_size:.1f}x smaller)")


if __name__ == "__main__":
    main()
