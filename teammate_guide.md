# Spatial Mind — Scene Production Pipeline Guide

**For:** Teammate (Frontend/Scene Producer)
**Project:** Spatial Mind @ JacHacks 2026
**Last Updated:** April 4, 2026

---

## Your Role

Matus is handling the semantic intelligence layer (SAM, CLIP, LangSplat, Jac walkers). **Your job is to produce beautiful, optimized Gaussian Splat scenes** that the semantic engine can query. This means:

1. Capture good video of scenes
2. Run the reconstruction pipeline (COLMAP or VGGT)
3. Train Gaussian Splats
4. Optimize/clean the output (prune, compress, bounding sphere)
5. Deliver `.ply` files ready for the web viewer + semantic training

---

## Table of Contents

1. [Environment Setup (Colab)](#1-environment-setup-colab)
2. [Capturing Good Video](#2-capturing-good-video)
3. [Option A: COLMAP Pipeline (Proven)](#3-option-a-colmap-pipeline-proven)
4. [Option B: VGGT Pipeline (Faster, Cutting-Edge)](#4-option-b-vggt-pipeline-faster-cutting-edge)
5. [Gaussian Splatting Training](#5-gaussian-splatting-training)
6. [Optimizing & Cleaning Gaussians](#6-optimizing--cleaning-gaussians)
7. [Web Viewer Testing](#7-web-viewer-testing)
8. [Recording Demo Videos](#8-recording-demo-videos)
9. [Delivering Files to Matus](#9-delivering-files-to-matus)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Environment Setup (Colab)

### Get an A100 GPU Runtime

1. Go to [Google Colab](https://colab.research.google.com)
2. Runtime → Change runtime type → **A100 GPU**
3. If A100 unavailable, use T4 (slower but works)

### Install Everything

Run this cell first — it installs all dependencies:

```python
# ============================================
# CELL 1: Full Environment Setup (~5 min)
# ============================================

# Mount Google Drive for saving outputs
from google.colab import drive
drive.mount('/content/drive')

# Install pycolmap with GPU support
!pip install pycolmap-cuda12 plyfile numpy Pillow --quiet

# Clone the official 3DGS repo (NOT LangSplat's fork — it has bugs)
!git clone https://github.com/graphdeco-inria/gaussian-splatting.git --recursive /content/gaussian-splatting

# Build the CUDA submodules
!cd /content/gaussian-splatting && pip install -e submodules/diff-gaussian-rasterization --no-build-isolation --quiet
!cd /content/gaussian-splatting && pip install submodules/simple-knn --no-build-isolation --quiet

# Fix simple-knn if it fails (missing include)
!grep -q 'cfloat' /content/gaussian-splatting/submodules/simple-knn/simple_knn.cu || \
  sed -i '1i #include <cfloat>' /content/gaussian-splatting/submodules/simple-knn/simple_knn.cu && \
  cd /content/gaussian-splatting && pip install submodules/simple-knn --no-build-isolation --quiet

# Create working directories
!mkdir -p /content/scenes

print("✅ Setup complete!")
```

### Install VGGT (Optional — for the faster pipeline)

```python
# ============================================
# CELL 1b: VGGT Setup (run only if using VGGT)
# ============================================

!git clone https://github.com/facebookresearch/vggt.git /content/vggt
!cd /content/vggt && pip install -r requirements.txt --quiet

# Also install gsplat (VGGT's recommended trainer, alternative to official 3DGS)
!pip install gsplat==1.3.0 --quiet

print("✅ VGGT setup complete!")
```

---

## 2. Capturing Good Video

This is the MOST important step. Bad video = bad reconstruction = ugly splats.

### Rules

| Do | Don't |
|----|-------|
| Move the camera SLOWLY and smoothly | Quick jerky movements |
| Walk AROUND the subject (full 360° if possible) | Stand in one spot and pan |
| Keep objects at consistent distance | Zoom in/out during capture |
| Film textured, colorful objects | White/shiny/reflective surfaces |
| Overlap! Every frame should share ~70% with the previous | Skip areas or make big jumps |
| Film in good, even lighting | Dark rooms or harsh directional light |
| 30-60 seconds is ideal | Too short (<15s) or too long (>2 min) |

### What Makes a Good Scene for Demo

- **Objects on surfaces** — books on tables, items on shelves, stuff on a desk
- **Mixed objects** — variety makes the semantic queries more interesting
- **Clear spatial relationships** — things on top of, next to, inside other things
- **NO PEOPLE in the scene** (they move between frames = ghosting)
- **Textured backgrounds** — bookshelves, posters, patterned rugs are great

### Recording Settings

- Use your phone camera (iPhone or Android)
- **1080p** resolution (4K works but is overkill and slower to process)
- **30 fps** is fine
- Save as `.mov` or `.mp4`

---

## 3. Option A: COLMAP Pipeline (Proven)

This is the pipeline Matus already validated. It works. Use this if VGGT gives issues.

### Step 1: Upload Video & Extract Frames

```python
# ============================================
# CELL 2: Frame Extraction
# ============================================

import os

# Upload your video to Colab or copy from Drive
# Example: copy from Drive
SCENE_NAME = "living_room"  # <-- CHANGE THIS for each scene
VIDEO_PATH = f"/content/drive/MyDrive/{SCENE_NAME}.mov"  # <-- CHANGE THIS

SCENE_DIR = f"/content/scenes/{SCENE_NAME}"
os.makedirs(f"{SCENE_DIR}/input", exist_ok=True)

# Extract frames at 2 fps (adjust: higher = more frames = better but slower)
# For a 45-second video at fps=2, you get ~90 frames (good balance)
# For a 60-second video at fps=1.5, you get ~90 frames
FPS = 2

!ffmpeg -i "{VIDEO_PATH}" -vf "fps={FPS}" -q:v 2 "{SCENE_DIR}/input/frame_%04d.jpg" -y

# Count frames
num_frames = len([f for f in os.listdir(f"{SCENE_DIR}/input") if f.endswith('.jpg')])
print(f"✅ Extracted {num_frames} frames at {FPS} fps")
print(f"   Target: 80-200 frames. {'👍 Good range!' if 80 <= num_frames <= 200 else '⚠️ Adjust FPS!'}")
```

### Step 2: COLMAP — Feature Extraction + Matching (GPU!)

```python
# ============================================
# CELL 3: COLMAP Reconstruction (GPU-accelerated)
# ============================================

import pycolmap
from pathlib import Path

database = f"{SCENE_DIR}/database.db"
images = f"{SCENE_DIR}/input"
sparse = Path(f"{SCENE_DIR}/sparse")
sparse.mkdir(exist_ok=True)

print("🔍 Extracting features (GPU)...")
pycolmap.extract_features(database, images)
print("✅ Features extracted!")

print("🔗 Matching features (GPU)...")
pycolmap.match_exhaustive(database)
print("✅ Matching complete!")

# Verify matching quality
import sqlite3
db = sqlite3.connect(database)
num_images = db.execute("SELECT COUNT(*) FROM images").fetchone()[0]
num_pairs = db.execute("SELECT COUNT(*) FROM two_view_geometries WHERE rows > 0").fetchone()[0]
db.close()

print(f"\n📊 Results:")
print(f"   Images: {num_images}")
print(f"   Verified pairs: {num_pairs}")
print(f"   {'👍 Looks good!' if num_pairs > num_images * 5 else '⚠️ Low pairs — video may have gaps'}")
```

### Step 3: Reconstruction

```python
# ============================================
# CELL 4: Incremental Mapping (~10-20 min)
# ============================================

print("🏗️ Running reconstruction... (this takes 10-20 min, no progress bar)")
print("   Check GPU with !nvidia-smi in another cell if worried")

maps = pycolmap.incremental_mapping(
    database_path=database,
    image_path=images,
    output_path=str(sparse)
)

print(f"\n✅ Reconstructed {len(maps)} model(s)")

# Load and verify
rec = pycolmap.Reconstruction(f"{SCENE_DIR}/sparse/0")
print(f"   Cameras: {rec.num_cameras()}")
print(f"   Images registered: {rec.num_reg_images()}")
print(f"   3D Points: {rec.num_points3D():,}")
print(f"   {'👍 Great reconstruction!' if rec.num_reg_images() > num_images * 0.8 else '⚠️ Many images failed to register'}")
```

### Step 4: Undistortion

```python
# ============================================
# CELL 5: Undistort Images
# ============================================

UNDISTORTED = f"{SCENE_DIR}_undistorted"

!colmap image_undistorter \
  --image_path {SCENE_DIR}/input \
  --input_path {SCENE_DIR}/sparse/0 \
  --output_path {UNDISTORTED} \
  --output_type COLMAP

# Fix directory structure (3DGS expects sparse/0/)
!mkdir -p {UNDISTORTED}/sparse/0
!mv {UNDISTORTED}/sparse/*.bin {UNDISTORTED}/sparse/0/ 2>/dev/null; true

# Verify camera model is PINHOLE
!python3 -c "
import pycolmap
rec = pycolmap.Reconstruction('{UNDISTORTED}/sparse/0')
for cam_id, cam in rec.cameras.items():
    print(f'Camera model: {cam.model_name}')
    assert cam.model_name == 'PINHOLE', f'ERROR: Expected PINHOLE, got {cam.model_name}'
print('✅ Undistortion verified — PINHOLE cameras')
"
```

---

## 4. Option B: VGGT Pipeline (Faster, Cutting-Edge)

VGGT is a CVPR 2025 Best Paper Award winner from Meta. It replaces COLMAP entirely — reconstructs a scene in **under 1 second** using a neural network instead of traditional SfM.

### Why VGGT?

- **Speed:** Seconds instead of 10-20 minutes for COLMAP
- **Quality:** Often better camera poses than COLMAP
- **Robustness:** Handles challenging scenes (reflections, low texture) better
- **Simplicity:** One command instead of extract → match → map → undistort

### Pipeline

```python
# ============================================
# CELL: VGGT Reconstruction
# ============================================

import os

SCENE_NAME = "kitchen"  # <-- CHANGE THIS
VIDEO_PATH = f"/content/drive/MyDrive/{SCENE_NAME}.mov"
SCENE_DIR = f"/content/scenes/{SCENE_NAME}"

# Extract frames (same as before)
os.makedirs(f"{SCENE_DIR}/images", exist_ok=True)  # NOTE: VGGT expects 'images/' not 'input/'
!ffmpeg -i "{VIDEO_PATH}" -vf "fps=2" -q:v 2 "{SCENE_DIR}/images/frame_%04d.jpg" -y

num_frames = len([f for f in os.listdir(f"{SCENE_DIR}/images") if f.endswith('.jpg')])
print(f"✅ Extracted {num_frames} frames")

# Run VGGT — outputs COLMAP-format files directly
!cd /content/vggt && python demo_colmap.py \
  --scene_dir={SCENE_DIR} \
  --use_ba

# This creates:
#   SCENE_DIR/sparse/cameras.bin
#   SCENE_DIR/sparse/images.bin  
#   SCENE_DIR/sparse/points3D.bin

print("✅ VGGT reconstruction complete!")
```

### Then Train with gsplat (VGGT's Recommended Trainer)

```python
# ============================================
# CELL: Train with gsplat
# ============================================

# gsplat is faster and more memory-efficient than the original 3DGS
!cd /content/gsplat && python examples/simple_trainer.py default \
  --data_factor 1 \
  --data_dir {SCENE_DIR} \
  --result_dir /content/scenes/{SCENE_NAME}_output

print("✅ Training complete!")
```

### OR Train with Official 3DGS (if gsplat has issues)

You may need to fix the sparse directory structure first:

```python
# Fix structure for official 3DGS
!mkdir -p {SCENE_DIR}/sparse/0
!mv {SCENE_DIR}/sparse/*.bin {SCENE_DIR}/sparse/0/ 2>/dev/null; true

# Rename images dir if needed (3DGS expects 'images/')
# VGGT already uses 'images/' so this should be fine

!cd /content/gaussian-splatting && python train.py \
  -s {SCENE_DIR} \
  -m /content/scenes/{SCENE_NAME}_output \
  --iterations 20000
```

### VGGT Notes

- VGGT needs **at least ~8GB VRAM** for ~100 images. A100 handles this easily.
- For very large image sets (200+), process in batches or reduce frame count.
- The `--use_ba` flag adds bundle adjustment for better accuracy (recommended).
- VGGT outputs cameras as PINHOLE by default — no undistortion step needed!

---

## 5. Gaussian Splatting Training

If using the COLMAP pipeline, train with the official repo:

```python
# ============================================
# CELL 6: Train Gaussians (~25-35 min on A100)
# ============================================

OUTPUT_DIR = f"/content/scenes/{SCENE_NAME}_output"

!cd /content/gaussian-splatting && python train.py \
  -s {UNDISTORTED} \
  -m {OUTPUT_DIR} \
  --iterations 20000

# You'll see progress like:
#   Training progress: 35% 7000/20000 [08:27, Loss=0.0416]
#   [ITER 7000] Evaluating train: L1 0.018 PSNR 31.2
#   Training progress: 100% 20000/20000 [34:59, Loss=0.035]

print("✅ Training complete!")
```

### What the Numbers Mean

| Metric | Good | Bad |
|--------|------|-----|
| PSNR | > 28 dB | < 25 dB |
| L1 Loss | < 0.03 | > 0.05 |
| Final Loss | < 0.04 | > 0.06 |

### Back Up Immediately

```python
!cp -r {OUTPUT_DIR} /content/drive/MyDrive/spatial_mind_{SCENE_NAME}/
print(f"✅ Backed up to Drive!")
```

---

## 6. Optimizing & Cleaning Gaussians

The raw output will be ~1-1.5 GB. For the web demo, we need to shrink it.

### Step 1: Check Current Size

```python
# ============================================
# CELL 7: Analyze Gaussians
# ============================================

from plyfile import PlyData
import numpy as np
import os

PLY_PATH = f"{OUTPUT_DIR}/point_cloud/iteration_20000/point_cloud.ply"
file_size = os.path.getsize(PLY_PATH) / (1024**2)

ply = PlyData.read(PLY_PATH)
data = ply.elements[0].data
print(f"📊 Gaussian Stats:")
print(f"   Count: {len(data):,}")
print(f"   File size: {file_size:.0f} MB")

# Analyze opacity distribution
opacity = 1.0 / (1.0 + np.exp(-np.array(data['opacity'])))
print(f"   Opacity < 0.05: {np.sum(opacity < 0.05):,} ({np.sum(opacity < 0.05)/len(data)*100:.1f}%)")
print(f"   Opacity < 0.1:  {np.sum(opacity < 0.1):,} ({np.sum(opacity < 0.1)/len(data)*100:.1f}%)")

# Analyze scale distribution  
scales = np.stack([
    np.exp(np.array(data['scale_0'])),
    np.exp(np.array(data['scale_1'])),
    np.exp(np.array(data['scale_2']))
], axis=1)
max_scale = np.max(scales, axis=1)
print(f"   Huge Gaussians (scale > 1.0): {np.sum(max_scale > 1.0):,}")
print(f"   Tiny Gaussians (scale < 0.001): {np.sum(max_scale < 0.001):,}")
```

### Step 2: Prune Invisible & Oversized Gaussians

```python
# ============================================
# CELL 8: Prune Gaussians
# ============================================

from plyfile import PlyData, PlyElement
import numpy as np

ply = PlyData.read(PLY_PATH)
data = ply.elements[0].data

# Compute opacity (stored as inverse sigmoid)
opacity = 1.0 / (1.0 + np.exp(-np.array(data['opacity'])))

# Compute scale
scales = np.stack([
    np.exp(np.array(data['scale_0'])),
    np.exp(np.array(data['scale_1'])),
    np.exp(np.array(data['scale_2']))
], axis=1)
max_scale = np.max(scales, axis=1)

# Remove:
# 1. Nearly transparent Gaussians (opacity < 0.05)
# 2. Gigantic Gaussians (likely floaters/artifacts)
# 3. Microscopic Gaussians (contribute nothing visually)
mask = (opacity > 0.05) & (max_scale < 2.0) & (max_scale > 0.0001)

print(f"Before: {len(data):,} Gaussians")
filtered = data[mask]
print(f"After:  {len(filtered):,} Gaussians ({len(filtered)/len(data)*100:.1f}% kept)")

# Save pruned version
PRUNED_PATH = PLY_PATH.replace('.ply', '_pruned.ply')
el = PlyElement.describe(filtered, 'vertex')
PlyData([el]).write(PRUNED_PATH)

pruned_size = os.path.getsize(PRUNED_PATH) / (1024**2)
print(f"File size: {file_size:.0f} MB → {pruned_size:.0f} MB ({(1-pruned_size/file_size)*100:.0f}% reduction)")
```

### Step 3: Bounding Sphere (Remove Floaters)

Gaussians that fly off into space (far from the scene center) are artifacts. Remove them:

```python
# ============================================
# CELL 9: Bounding Sphere Crop
# ============================================

from plyfile import PlyData, PlyElement
import numpy as np

ply = PlyData.read(PRUNED_PATH)
data = ply.elements[0].data

# Get positions
x = np.array(data['x'])
y = np.array(data['y'])
z = np.array(data['z'])
positions = np.stack([x, y, z], axis=1)

# Compute scene center (median is more robust than mean to outliers)
center = np.median(positions, axis=0)
distances = np.linalg.norm(positions - center, axis=1)

# Keep only Gaussians within a reasonable radius
# Use percentile to auto-determine radius
radius = np.percentile(distances, 99)  # keep 99% of points
mask = distances < radius

print(f"Scene center: {center}")
print(f"Auto radius (99th percentile): {radius:.2f}")
print(f"Before: {len(data):,} → After: {np.sum(mask):,} ({np.sum(~mask):,} floaters removed)")

filtered = data[mask]
FINAL_PATH = PLY_PATH.replace('.ply', '_final.ply')
el = PlyElement.describe(filtered, 'vertex')
PlyData([el]).write(FINAL_PATH)

final_size = os.path.getsize(FINAL_PATH) / (1024**2)
print(f"\n📦 Final file: {final_size:.0f} MB (down from {file_size:.0f} MB)")
```

### Step 4: Strip Higher-Order Spherical Harmonics (Biggest Size Win)

The default 3DGS stores 48 SH coefficients per Gaussian (for view-dependent color). Most web viewers only use the first 3 (DC term). Stripping the rest cuts file size by ~60%:

```python
# ============================================
# CELL 10: Strip SH Coefficients for Web
# ============================================

from plyfile import PlyData, PlyElement
import numpy as np

ply = PlyData.read(FINAL_PATH)
data = ply.elements[0].data

# Keep only essential properties
# Core: x, y, z, opacity, scale_0/1/2, rot_0/1/2/3
# Color: f_dc_0, f_dc_1, f_dc_2 (just the base color, not higher SH bands)
keep_props = ['x', 'y', 'z', 'opacity',
              'scale_0', 'scale_1', 'scale_2',
              'rot_0', 'rot_1', 'rot_2', 'rot_3',
              'f_dc_0', 'f_dc_1', 'f_dc_2']

# Build new structured array with only kept properties
dtypes = [(name, data.dtype[name]) for name in keep_props if name in data.dtype.names]
new_data = np.empty(len(data), dtype=dtypes)
for name, _ in dtypes:
    new_data[name] = data[name]

WEB_PATH = PLY_PATH.replace('.ply', '_web.ply')
el = PlyElement.describe(new_data, 'vertex')
PlyData([el]).write(WEB_PATH)

web_size = os.path.getsize(WEB_PATH) / (1024**2)
print(f"📦 Web-optimized: {web_size:.0f} MB (down from {file_size:.0f} MB original)")
print(f"   {(1-web_size/file_size)*100:.0f}% total reduction!")
```

---

## 7. Web Viewer Testing

### Quick Test: SuperSplat (No Setup Required)

1. Download your `_web.ply` file from Drive
2. Go to [playcanvas.com/supersplat/editor](https://playcanvas.com/supersplat/editor)
3. Drag and drop the file
4. Navigate around — check for artifacts, floaters, blurry areas

### Quick Test: antimatter15 Viewer

1. Go to [antimatter15.com/splat](https://antimatter15.com/splat/)
2. Upload your PLY file
3. This viewer is simpler but faster for large files

### What to Look For

| Good | Bad (needs more capture/cleanup) |
|------|----------------------------------|
| Sharp edges on objects | Blurry/cloudy areas |
| Consistent color | Color bleeding between objects |
| Clean background | Floaters (random blobs in air) |
| Smooth surfaces | Holes or missing areas |
| Objects clearly distinguishable | Merged/blobby objects |

---

## 8. Recording Demo Videos

For the hackathon presentation, pre-record a smooth fly-through of your best scene.

### Using SuperSplat

1. Load your scene in SuperSplat
2. Set up camera keyframes for a smooth path
3. Use the built-in recording feature
4. Export as MP4

### Using the Built-in Renderer

```python
# Render all training views (quick check)
!cd /content/gaussian-splatting && python render.py \
  -s {UNDISTORTED} \
  -m {OUTPUT_DIR}

# View renders
from IPython.display import display, Image
import glob

renders = sorted(glob.glob(f"{OUTPUT_DIR}/train/ours_20000/renders/*.png"))
for img in renders[:10]:
    display(Image(filename=img, width=400))
```

### Making a Video from Renders

```python
# Stitch renders into a video
!ffmpeg -framerate 30 \
  -pattern_type glob -i '{OUTPUT_DIR}/train/ours_20000/renders/*.png' \
  -c:v libx264 -pix_fmt yuv420p \
  /content/drive/MyDrive/{SCENE_NAME}_flythrough.mp4

print("✅ Video saved to Drive!")
```

---

## 9. Delivering Files to Matus

### What Matus Needs from You

For EACH scene, deliver these to Google Drive:

```
spatial_mind_{scene_name}/
├── point_cloud.ply              # Full training output (for LangSplat semantic training)
├── point_cloud_web.ply          # Optimized for web viewer
├── chkpnt20000.pth              # Training checkpoint (for LangSplat to continue from)
├── cameras.json                 # Camera parameters (auto-generated by train.py)
├── input/                       # Original extracted frames
│   ├── frame_0001.jpg
│   ├── frame_0002.jpg
│   └── ...
└── sparse/0/                    # COLMAP reconstruction
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin
```

### Copy Command

```python
# Copy everything Matus needs
import shutil

DRIVE_OUT = f"/content/drive/MyDrive/spatial_mind_{SCENE_NAME}"
os.makedirs(DRIVE_OUT, exist_ok=True)

# Full PLY (for semantic training)
shutil.copy2(PLY_PATH, f"{DRIVE_OUT}/point_cloud.ply")

# Web PLY (for viewer)
shutil.copy2(WEB_PATH, f"{DRIVE_OUT}/point_cloud_web.ply")

# Checkpoint
ckpt = f"{OUTPUT_DIR}/chkpnt20000.pth"
if os.path.exists(ckpt):
    shutil.copy2(ckpt, f"{DRIVE_OUT}/chkpnt20000.pth")

# Cameras
cam_json = f"{OUTPUT_DIR}/cameras.json"
if os.path.exists(cam_json):
    shutil.copy2(cam_json, f"{DRIVE_OUT}/cameras.json")

# Frames
shutil.copytree(f"{SCENE_DIR}/input", f"{DRIVE_OUT}/input", dirs_exist_ok=True)

# Sparse reconstruction
sparse_src = f"{UNDISTORTED}/sparse/0" if os.path.exists(f"{UNDISTORTED}/sparse/0") else f"{SCENE_DIR}/sparse/0"
shutil.copytree(sparse_src, f"{DRIVE_OUT}/sparse/0", dirs_exist_ok=True)

print(f"✅ All files delivered to {DRIVE_OUT}")
```

---

## 10. Troubleshooting

### COLMAP: "No such file: images.bin"
The sparse files are in `sparse/` not `sparse/0/`. Fix:
```bash
!mkdir -p {SCENE_DIR}/sparse/0
!mv {SCENE_DIR}/sparse/*.bin {SCENE_DIR}/sparse/0/
```

### COLMAP: "camera model not handled: only PINHOLE supported"
You skipped undistortion or pointed train.py at the wrong directory. Use the `_undistorted` path.

### 3DGS: "CUDA error: illegal memory access"
The rasterizer buffer overflow bug. Solutions:
1. Reduce `--densify_grad_threshold 0.001` (default 0.0002)
2. Use `--resolution 2` to halve image resolution
3. Try fewer iterations: `--iterations 15000`
4. Reduce frame count (re-extract at lower fps)

### Training Loss Doesn't Decrease
- Bad COLMAP reconstruction — check that most images registered
- Video has motion blur — re-capture with steadier hands
- Scene is too reflective/transparent — add more textured objects

### PLY File Won't Load in Web Viewer
- File too large — run the optimization steps (Section 6)
- Wrong format — make sure you're using the `_web.ply` version
- Browser out of memory — try a smaller scene or fewer Gaussians

### VGGT: Out of Memory
- Reduce number of input frames (aim for ~80-100)
- Use `--max_images 100` if available
- Switch to COLMAP pipeline as fallback

---

## Quick Reference: Full Pipeline Commands

```python
# === ONE-SHOT PIPELINE (copy-paste friendly) ===

SCENE_NAME = "my_scene"
VIDEO_PATH = f"/content/drive/MyDrive/{SCENE_NAME}.mov"
SCENE_DIR = f"/content/scenes/{SCENE_NAME}"
UNDISTORTED = f"{SCENE_DIR}_undistorted"
OUTPUT_DIR = f"/content/scenes/{SCENE_NAME}_output"

# 1. Extract frames
!mkdir -p {SCENE_DIR}/input
!ffmpeg -i {VIDEO_PATH} -vf "fps=2" -q:v 2 {SCENE_DIR}/input/frame_%04d.jpg -y

# 2. COLMAP (GPU)
import pycolmap
pycolmap.extract_features(f"{SCENE_DIR}/database.db", f"{SCENE_DIR}/input")
pycolmap.match_exhaustive(f"{SCENE_DIR}/database.db")
import pathlib; pathlib.Path(f"{SCENE_DIR}/sparse").mkdir(exist_ok=True)
pycolmap.incremental_mapping(f"{SCENE_DIR}/database.db", f"{SCENE_DIR}/input", f"{SCENE_DIR}/sparse")

# 3. Undistort
!colmap image_undistorter --image_path {SCENE_DIR}/input --input_path {SCENE_DIR}/sparse/0 --output_path {UNDISTORTED} --output_type COLMAP
!mkdir -p {UNDISTORTED}/sparse/0 && mv {UNDISTORTED}/sparse/*.bin {UNDISTORTED}/sparse/0/

# 4. Train (20K iterations, ~30 min)
!cd /content/gaussian-splatting && python train.py -s {UNDISTORTED} -m {OUTPUT_DIR} --iterations 20000

# 5. Backup to Drive
!cp -r {OUTPUT_DIR} /content/drive/MyDrive/spatial_mind_{SCENE_NAME}/
```

---

## Scene Ideas for Demo

Capture 2-3 of these if time permits:

1. **Hackathon desk** — laptops, water bottles, snacks, notebooks (great for "what's on the table?")
2. **Room corner** — chair, bookshelf, plant, lamp (great for spatial queries like "what's next to the bookshelf?")
3. **Kitchen counter** — mugs, fruit, cutting board (great for "what's near the sink?")

**Priority: Quality > Quantity.** One amazing scene beats three mediocre ones.
