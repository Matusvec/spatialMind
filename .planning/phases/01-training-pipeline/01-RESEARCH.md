# Phase 1: Training Pipeline - Research

**Researched:** 2026-04-04
**Domain:** LangSplat 3D Language Gaussian Splatting training pipeline on Google Colab A100
**Confidence:** MEDIUM-HIGH

## Summary

Phase 1 produces the three co-located LangSplat artifacts (point_cloud.ply, language_feature_dim3/*.npy, autoencoder.pth) on a Google Colab A100 and verifies them with a cosine similarity smoke test. The training pipeline is a strict 6-stage sequential process: COLMAP SfM, SAM mask generation + CLIP feature extraction, autoencoder training, base 3DGS RGB training (30K iterations), then LangSplat training (three feature levels). A pre-trained fallback scene (lerf_figurines from the LERF dataset) runs in parallel from minute 0 as insurance.

The biggest risks are (1) a critical version mismatch between Colab's default runtime (Python 3.12, CUDA 12.4) and LangSplat's pinned dependencies (Python 3.7, PyTorch 1.12.1, CUDA 11.6 in environment.yml), (2) Colab session timeouts killing long-running training, and (3) COLMAP failing on featureless indoor surfaces. All three are mitigable with specific notebook setup patterns documented below.

**Primary recommendation:** Build the Colab notebook as a pip-only setup (no conda) targeting Python 3.10 with PyTorch 2.0.1+cu118, which satisfies LangSplat's CUDA extension compilation requirements while working within Colab's constraints. Start the lerf_figurines fallback at minute 0, venue training after scan upload at ~30 min.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Scan the actual hackathon room at UMich -- judges will be in that space, recognition adds demo impact
- **D-02:** Capture 50-80 photos with Scaniverse -- good reconstruction quality while keeping training under 3 hours
- **D-03:** Use Scaniverse on iPhone, export photos for COLMAP processing
- **D-04:** Fallback scene is `lerf_figurines` from the public LangSplat dataset -- tabletop scene with varied objects, good for spatial queries
- **D-05:** Run fallback and live training in parallel from the start -- download and pre-process figurines immediately at minute 0, start venue training after scan at ~30 min. Use whichever produces valid output first.
- **D-06:** Staggered start -- kick off figurines (fallback) training at minute 0 of hackathon, start venue training after Scaniverse scan completes (~30 min in)
- **D-07:** Checkpoint every 5K iterations to Google Drive -- if Colab session dies, resume from last checkpoint on a new session
- **D-08:** Mount Google Drive as the very first Colab cell -- all artifacts and checkpoints survive session timeouts
- **D-09:** All-in-one notebook with sequential cells: setup -> COLMAP -> SAM -> autoencoder -> 3DGS RGB (30K iter) -> LangSplat training. Single notebook, run top-to-bottom.
- **D-10:** Final artifacts auto-upload to InsForge S3 after training completes (Drive + InsForge dual storage)
- **D-11:** Monitor via Colab output tab -- just check the notebook occasionally for training loss prints. No elaborate monitoring system.

### Claude's Discretion
- Training hyperparameters (iterations, learning rate, etc.) -- use LangSplat defaults unless research suggests otherwise
- COLMAP configuration -- standard defaults for indoor scenes
- CLIP backbone choice -- use whatever LangSplat's default is (confirmed: ViT-B/16 with laion2b_s34b_b88k)
- Exact Scaniverse export settings -- standard photo export

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| TRAIN-01 | LangSplat 5-stage pipeline runs on Colab A100 (COLMAP -> SAM -> autoencoder -> 3DGS RGB -> LangSplat training) | Full pipeline commands documented in Architecture Patterns; version pinning strategy verified; COLMAP install path confirmed |
| TRAIN-02 | Training output produces PLY + .npy latent features + autoencoder.pth as three co-located artifacts | Output directory structure confirmed from LangSplat README; artifact paths mapped in Code Examples |
| TRAIN-03 | Pre-trained fallback scene stored on Google Drive as insurance against training failures | LERF dataset Google Drive download link confirmed; lerf_figurines structure documented; pre-trained model download link confirmed |
| TRAIN-04 | Live scan of JacHacks venue at UMich captured and trained during the hackathon | Scaniverse photo export -> COLMAP -> LangSplat pipeline documented; timing estimates provided |
</phase_requirements>

## Standard Stack

### Core (Training Environment)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python | 3.10.x (Colab default or condacolab) | Training runtime | LangSplat env.yml pins 3.7 but this is outdated; PyTorch 2.0.x requires >=3.8; 3.10 is the sweet spot for CUDA 11.8 compat |
| PyTorch | 2.0.1+cu118 | Tensor ops, CUDA extension compilation | Must use cu118 build to match CUDA 11.8 SDK; LangSplat env.yml pins 1.12.1 but newer works if CUDA version matches |
| CUDA SDK | 11.8 (via PyTorch wheel, not system CUDA) | diff-gaussian-rasterization compilation | LangSplat README states "we used 11.8"; custom CUDA ops require matching SDK at compile time |
| COLMAP | apt binary or pycolmap | SfM camera poses from photos | LangSplat's convert.py wraps COLMAP; apt version lacks GPU SIFT but works; pycolmap-cuda12 alternative |
| SAM (vit_h) | 1.0 (LangSplat fork) | Per-image hierarchical mask generation | Must use segment-anything-langsplat fork, NOT stock facebook/segment-anything |
| open-clip-torch | latest compatible | CLIP ViT-B/16 image encoding | LangSplat preprocess.py loads ViT-B-16 with laion2b_s34b_b88k weights; produces 512-dim features |
| graphdeco-inria/gaussian-splatting | from source (submodule) | Base 3DGS RGB training | LangSplat submodule; trains RGB checkpoint needed as --start_checkpoint for LangSplat |
| langsplat-rasterization | from source (submodule) | Custom CUDA Gaussian rasterizer with language features | Compiled during setup; requires CUDA 11.8 matching PyTorch |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| condacolab | latest | Install conda/mamba on Colab | Only if pip-only setup fails; adds ~2 min setup + kernel restart |
| plyfile | 0.8.1+ | PLY file parsing | Reading output PLY to verify Gaussian count and properties |
| numpy | 1.x or 2.x | Feature array manipulation | Smoke test: load .npy files, compute cosine similarity |
| tqdm | latest | Progress bars | Visual feedback during long training cells |
| gdown | latest | Google Drive file download | Download LERF dataset and SAM checkpoint from Drive links |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| pip-only install on Colab | condacolab + environment.yml | Conda adds kernel restart + 2-3 min overhead; pip is faster but requires manual version pinning |
| COLMAP via apt | pycolmap (pip) | pycolmap is easier to install but GPU SIFT requires pycolmap-cuda12; apt COLMAP binary also lacks GPU SIFT on Colab |
| Python 3.10 | Python 3.7 (env.yml default) | 3.7 is EOL; Colab no longer supports it; 3.10 works with all LangSplat deps when versions are adjusted |

**Installation (Colab notebook cell 2, after Drive mount):**
```python
# Force PyTorch with CUDA 11.8
!pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
  --index-url https://download.pytorch.org/whl/cu118

# Clone LangSplat with submodules
!git clone --recursive https://github.com/minghanqin/LangSplat.git /content/drive/MyDrive/spatialMind/LangSplat

# Install CUDA extensions (compiles against cu118)
!pip install /content/drive/MyDrive/spatialMind/LangSplat/submodules/langsplat-rasterization
!pip install /content/drive/MyDrive/spatialMind/LangSplat/submodules/simple-knn

# Install SAM (LangSplat fork)
!pip install git+https://github.com/minghanqin/segment-anything-langsplat.git

# Install remaining deps
!pip install open-clip-torch plyfile tqdm tensorboard opencv-python

# Install COLMAP
!apt-get install -y colmap

# Download SAM checkpoint
!wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
  -P /content/drive/MyDrive/spatialMind/ckpts/
```

**Version verification notes:**
- PyTorch 2.0.1+cu118: confirmed available at https://download.pytorch.org/whl/cu118
- COLMAP via apt: confirmed available on Colab's Ubuntu base (no GPU SIFT, but CPU SIFT works for 50-80 images)
- SAM vit_h checkpoint: 2.4 GB download, save to Drive to avoid re-downloading on session restart

## Architecture Patterns

### Recommended Notebook Structure

```
Colab Notebook (sequential cells, top-to-bottom):
  Cell 0: Mount Google Drive
  Cell 1: Environment setup (pip install, clone repo, compile CUDA extensions)
  Cell 2: Download SAM checkpoint + LERF fallback dataset (to Drive)
  Cell 3: COLMAP SfM (either on fallback or on uploaded venue photos)
  Cell 4: SAM + CLIP preprocessing (preprocess.py)
  Cell 5: Autoencoder training (autoencoder/train.py)
  Cell 6: Base 3DGS RGB training (30K iterations with 5K checkpoints)
  Cell 7: LangSplat training (feature levels 0, 1, 2)
  Cell 8: Smoke test (cosine similarity verification)
  Cell 9: Upload to InsForge S3
```

### Output Directory Structure (confirmed from LangSplat README)

```
/content/drive/MyDrive/spatialMind/data/<scene_name>/
  images/                              # Input photos (from Scaniverse)
  input/                               # Raw photos (convert.py reads from here)
  sparse/0/                            # COLMAP output
    cameras.bin
    images.bin
    points3D.bin
  language_features/                   # Per-image: {name}_f.npy (512-dim CLIP), {name}_s.npy (segmentation maps)
  language_feature_dim3/               # Compressed 3-dim features (after autoencoder test.py)
  output/<scene_name>/
    point_cloud/iteration_30000/
      point_cloud.ply                  # Final Gaussians: XYZ + color + opacity + covariance
    chkpnt30000.pth                    # RGB model checkpoint (input to LangSplat training)
    autoencoder.pth                    # Decoder: 3-dim latent -> 512-dim CLIP
```

**Critical co-location requirement:** The three artifacts that Phase 2 consumes are:
1. `output/<scene>/point_cloud/iteration_30000/point_cloud.ply` -- Gaussian positions + visual properties + 3-dim language latents
2. `language_feature_dim3/*.npy` -- compressed per-image language features
3. `output/<scene>/autoencoder.pth` -- scene-specific decoder (3-dim -> 512-dim)

All three must travel together. If the autoencoder is from a different training run than the PLY, decoded embeddings will be garbage.

### Pattern 1: Staggered Parallel Training

**What:** Run two training pipelines simultaneously -- fallback (lerf_figurines) and live venue scan.
**When to use:** Always during hackathon. Fallback starts at minute 0, venue starts after photos are uploaded (~30 min).
**Implementation:**
- Download LERF dataset to Drive in Cell 2
- Start COLMAP + full pipeline on figurines immediately (Cells 3-7 for figurines path)
- When Scaniverse photos are uploaded to Drive, duplicate the notebook or re-run Cells 3-7 with the venue path
- Use whichever completes first with valid smoke test results

### Pattern 2: Checkpoint-and-Resume

**What:** Save 3DGS checkpoints every 5K iterations to Google Drive so training survives session timeouts.
**When to use:** Always. Colab sessions can die at any time.
**Implementation:**
```python
# In 3DGS training cell, pass --save_iterations
!python train.py -s $DATASET_PATH \
  -m /content/drive/MyDrive/spatialMind/data/$SCENE/output/$SCENE \
  --iterations 30000 \
  --save_iterations 5000 10000 15000 20000 25000 30000
```
On resume after timeout:
```python
# Resume from last checkpoint
!python train.py -s $DATASET_PATH \
  -m /content/drive/MyDrive/spatialMind/data/$SCENE/output/$SCENE \
  --start_checkpoint /content/drive/MyDrive/spatialMind/data/$SCENE/output/$SCENE/chkpnt25000.pth \
  --iterations 30000
```

### Pattern 3: Cosine Similarity Smoke Test

**What:** After training, verify embeddings are semantically meaningful (not random noise).
**When to use:** After every completed training run before declaring success.
**Implementation:**
```python
import numpy as np
import torch
import open_clip

# Load autoencoder
autoencoder = torch.load('autoencoder.pth')
autoencoder.eval()

# Load a sample of compressed features
features_3d = np.load('language_feature_dim3/00000_f.npy')  # shape: (N_masks, 3)
features_3d_tensor = torch.tensor(features_3d).cuda().float()

# Decode to 512-dim
with torch.no_grad():
    features_512 = autoencoder.decode(features_3d_tensor)  # shape: (N_masks, 512)
    features_512 = features_512 / features_512.norm(dim=-1, keepdim=True)

# Encode text queries
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
tokenizer = open_clip.get_tokenizer('ViT-B-16')
model.eval().cuda()

texts = ["chair", "table", "wall", "floor", "random gibberish xyz"]
text_tokens = tokenizer(texts).cuda()
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# Compute cosine similarity
for i, text in enumerate(texts):
    sims = (features_512 @ text_features[i]).cpu().numpy()
    print(f"'{text}': min={sims.min():.3f}, max={sims.max():.3f}, mean={sims.mean():.3f}, std={sims.std():.3f}")

# PASS criteria: std > 0.05 for real object queries, max(chair) != max(table)
# FAIL criteria: all scores in 0.3-0.5 range with std < 0.02 = embeddings are noise
```

### Anti-Patterns to Avoid

- **Using conda on Colab without condacolab:** Conda is not pre-installed. Raw `conda install` commands will fail. Use pip or condacolab.
- **Running LangSplat without the RGB checkpoint:** Stage 3 (LangSplat training) requires `--start_checkpoint` pointing to a completed 3DGS RGB model. Without it, training starts from random init and produces garbage embeddings.
- **Saving artifacts to /content/ instead of /content/drive/:** Colab's ephemeral filesystem is wiped on session timeout. ALL artifacts must go to mounted Google Drive.
- **Using SAM 2 or SAM 3:** LangSplat's preprocessing fork is based on SAM 1 (vit_h). SAM 2/3 have different APIs and will break the mask generation pipeline.
- **Skipping the autoencoder step:** Without autoencoder training (Step 2), the 512-dim CLIP features cannot be compressed to 3-dim for per-Gaussian storage. The LangSplat training will fail or produce meaningless latents.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| SfM camera poses from photos | Custom pose estimation | COLMAP via convert.py | COLMAP handles lens distortion, bundle adjustment, outlier rejection. Custom SfM takes weeks to get right. |
| Per-image semantic masks | Manual annotation or simple thresholding | SAM vit_h (LangSplat fork) | SAM generates hierarchical masks at multiple scales automatically. Manual masks for 80 images would take hours. |
| CLIP feature extraction | Custom feature extractor | LangSplat preprocess.py | Handles multi-scale mask cropping, CLIP encoding, NMS deduplication, and proper output format. |
| Autoencoder 512->3 dim compression | PCA or random projection | LangSplat autoencoder/train.py | Scene-specific learned compression preserves semantic structure better than linear methods. |
| 3DGS training loop | Custom Gaussian optimization | graphdeco-inria train.py | Adaptive density control, pruning, learning rate scheduling are all critical and already implemented. |
| Google Drive mounting | Manual file copying | `from google.colab import drive; drive.mount()` | One line, handles auth, persists across cells. |

**Key insight:** The entire LangSplat training pipeline is a sequence of existing scripts with specific arguments. The notebook's job is to orchestrate them in the right order with the right paths, not to reimplement any training logic.

## Common Pitfalls

### Pitfall 1: Version Mismatch Between Colab Default and LangSplat Requirements

**What goes wrong:** Colab defaults to Python 3.12 and CUDA 12.4 as of 2026. LangSplat's environment.yml pins Python 3.7.13 and PyTorch 1.12.1 with CUDA 11.6. The LangSplat README says "we used CUDA 11.8". The custom CUDA extensions (langsplat-rasterization, simple-knn) must be compiled against the same CUDA version as PyTorch's runtime. Mismatches cause cryptic compilation errors or silent runtime failures.

**Why it happens:** LangSplat was published in 2024 with an outdated environment.yml that was never updated. Colab runtime versions advance independently.

**How to avoid:** Do NOT use the environment.yml. Install PyTorch 2.0.1+cu118 via pip from the cu118 index URL. This provides CUDA 11.8 runtime libraries bundled with PyTorch, which is sufficient for compiling the CUDA extensions even though Colab's system CUDA is 12.x. The pip wheel bundles its own CUDA runtime.

**Warning signs:** `RuntimeError: CUDA error: no kernel image is available for execution` or compilation errors mentioning `nvcc` version mismatch.

### Pitfall 2: COLMAP GPU SIFT Not Available on Colab apt Binary

**What goes wrong:** The COLMAP binary installed via `apt-get install colmap` on Colab does not include GPU-accelerated SIFT feature extraction. It falls back to CPU SIFT, which is slower but functional. For 50-80 images, CPU SIFT takes 5-15 minutes instead of 1-2 minutes.

**Why it happens:** Ubuntu apt packages for COLMAP are built without CUDA support to keep the package universal.

**How to avoid:** Accept the CPU SIFT slowdown (15 min is acceptable). Alternatively, install pycolmap-cuda12 via pip for GPU SIFT, but this introduces a CUDA 12 dependency that may conflict with the cu118 PyTorch install. For a hackathon, the CPU path is the safer choice.

**Warning signs:** COLMAP output shows "Using CPU SIFT extraction" in logs. Not a failure, just slower.

### Pitfall 3: Scaniverse Does Not Export COLMAP Format Directly

**What goes wrong:** Scaniverse exports photos, PLY meshes, SPZ splats, and various 3D formats, but does NOT export COLMAP-format sparse reconstruction (cameras.bin, images.bin, points3D.bin). You cannot skip COLMAP by using Scaniverse's reconstruction.

**Why it happens:** Scaniverse's internal SfM is proprietary and outputs to its own format. The "Save raw data" option preserves RGB frames and LiDAR depth but not in COLMAP's binary format.

**How to avoid:** Export photos only from Scaniverse. Upload to Google Drive. Run COLMAP on the photos in Colab using LangSplat's convert.py, which wraps COLMAP's full pipeline (feature extraction -> matching -> mapper -> undistorter).

**Warning signs:** Looking for cameras.bin in Scaniverse export and not finding it.

### Pitfall 4: Colab Session Timeout Kills Multi-Hour Training

**What goes wrong:** Colab Pro has a ~12-hour session limit and a ~90-minute idle timeout. The full LangSplat pipeline (COLMAP + SAM preprocessing + autoencoder + 3DGS 30K iter + LangSplat 3 levels) can take 3-5 hours on A100. If the session dies between stages, intermediate results on ephemeral storage are lost.

**Why it happens:** Browser tab losing focus triggers the idle timer. Session limits are hard caps regardless of activity.

**How to avoid:**
1. Mount Google Drive in Cell 0. ALL paths point to Drive.
2. Save 3DGS checkpoints every 5K iterations to Drive.
3. Keep the Colab tab focused (or use a keep-alive snippet in browser console).
4. Pre-download SAM checkpoint and LERF dataset to Drive so they survive restarts.
5. The autoencoder training (100 epochs, ~10-15 min) and LangSplat training (~30-60 min per level) each produce checkpoints that survive on Drive.

**Warning signs:** Colab shows "Disconnected" or "Runtime disconnected" badge.

### Pitfall 5: Missing RGB Checkpoint Causes Silent Embedding Failure

**What goes wrong:** LangSplat training (train.py) requires `--start_checkpoint` pointing to a completed 3DGS RGB model (chkpnt30000.pth). If this flag is omitted or points to a nonexistent file, training either crashes with a non-obvious error or silently trains from random initialization. The resulting embeddings look plausible (the PLY file has the right structure) but the CLIP latents are random noise.

**Why it happens:** The process.sh script assumes a pre-trained RGB model already exists. New users skip the 3DGS RGB training step thinking LangSplat is end-to-end.

**How to avoid:** Always verify the RGB checkpoint exists before running LangSplat training:
```python
import os
ckpt_path = f"/content/drive/MyDrive/spatialMind/data/{scene}/output/{scene}/chkpnt30000.pth"
assert os.path.exists(ckpt_path), f"RGB checkpoint missing at {ckpt_path}! Run 3DGS training first."
```

**Warning signs:** LangSplat train.py output shows loss starting from a very high value (>10) instead of starting relatively low if loading from a valid checkpoint.

### Pitfall 6: Autoencoder Trains on Wrong Feature Directory

**What goes wrong:** The autoencoder's train.py expects features in `<dataset_path>/language_features/` (512-dim per-image .npy files from preprocess.py). If the path is wrong or the directory is empty, training silently produces a random autoencoder that maps 3-dim latents to garbage 512-dim vectors.

**Why it happens:** The `--dataset_path` argument must match exactly between preprocess.py (which writes language_features/) and autoencoder/train.py (which reads from it).

**How to avoid:** After preprocessing, verify the output exists:
```python
import os, glob
feat_dir = f"/content/drive/MyDrive/spatialMind/data/{scene}/language_features"
npy_files = glob.glob(f"{feat_dir}/*_f.npy")
assert len(npy_files) > 0, f"No language features found in {feat_dir}! Run preprocess.py first."
print(f"Found {len(npy_files)} feature files")
```

**Warning signs:** Autoencoder training loss does not decrease from initial value.

## Code Examples

### Complete Pipeline Commands (in Colab cell execution order)

```python
# === Cell 0: Mount Drive ===
from google.colab import drive
drive.mount('/content/drive')

BASE = "/content/drive/MyDrive/spatialMind"
SCENE = "figurines"  # or "jachacks_venue"
DATASET_PATH = f"{BASE}/data/{SCENE}"
```

```python
# === Cell 1: Setup ===
!pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
  --index-url https://download.pytorch.org/whl/cu118

# Clone LangSplat if not already on Drive
import os
if not os.path.exists(f"{BASE}/LangSplat"):
    !git clone --recursive https://github.com/minghanqin/LangSplat.git {BASE}/LangSplat

# Install CUDA extensions
!pip install {BASE}/LangSplat/submodules/langsplat-rasterization
!pip install {BASE}/LangSplat/submodules/simple-knn
!pip install git+https://github.com/minghanqin/segment-anything-langsplat.git
!pip install open-clip-torch plyfile tqdm tensorboard opencv-python-headless

# COLMAP
!apt-get install -y colmap
```

```python
# === Cell 2: Download SAM + LERF fallback ===
SAM_CKPT = f"{BASE}/ckpts/sam_vit_h_4b8939.pth"
if not os.path.exists(SAM_CKPT):
    !wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P {BASE}/ckpts/

# Download LERF dataset (contains figurines, ramen, teatime, waldo_kitchen)
# Link from LangSplat README: https://drive.google.com/file/d/1QF1Po5p5DwTjFHu6tnTeYs_G0egMVmHt/view
LERF_PATH = f"{BASE}/data/lerf_ovs"
if not os.path.exists(LERF_PATH):
    !pip install gdown
    !gdown 1QF1Po5p5DwTjFHu6tnTeYs_G0egMVmHt -O {BASE}/data/lerf_ovs.tar
    !tar -xf {BASE}/data/lerf_ovs.tar -C {BASE}/data/
```

```python
# === Cell 3: COLMAP (for custom scene only -- LERF already has sparse/) ===
%cd {BASE}/LangSplat
!python convert.py -s {DATASET_PATH}
# Verify COLMAP success
import os
sparse_path = f"{DATASET_PATH}/sparse/0/cameras.bin"
assert os.path.exists(sparse_path), "COLMAP failed! Check images in input/ directory."
```

```python
# === Cell 4: SAM + CLIP Preprocessing ===
%cd {BASE}/LangSplat
!python preprocess.py --dataset_path {DATASET_PATH}
# Verify output
import glob
npy_files = glob.glob(f"{DATASET_PATH}/language_features/*_f.npy")
print(f"Generated {len(npy_files)} language feature files")
assert len(npy_files) > 0, "Preprocessing failed!"
```

```python
# === Cell 5: Autoencoder Training ===
%cd {BASE}/LangSplat/autoencoder
!python train.py \
  --dataset_path {DATASET_PATH} \
  --encoder_dims 256 128 64 32 3 \
  --decoder_dims 16 32 64 128 256 256 512 \
  --lr 0.0007 \
  --output {DATASET_PATH}/ae_ckpt

# Generate compressed 3-dim features
!python test.py --dataset_path {DATASET_PATH} --output {DATASET_PATH}
```

```python
# === Cell 6: Base 3DGS RGB Training (30K iterations) ===
%cd {BASE}/LangSplat
OUTPUT_DIR = f"{DATASET_PATH}/output/{SCENE}"
!python train.py -s {DATASET_PATH} \
  -m {OUTPUT_DIR} \
  --iterations 30000 \
  --save_iterations 5000 10000 15000 20000 25000 30000
# Verify checkpoint
assert os.path.exists(f"{OUTPUT_DIR}/chkpnt30000.pth"), "3DGS RGB training incomplete!"
```

```python
# === Cell 7: LangSplat Training (3 feature levels) ===
%cd {BASE}/LangSplat
for level in [0, 1, 2]:
    !python train.py -s {DATASET_PATH} \
      -m {OUTPUT_DIR}_{level} \
      --start_checkpoint {OUTPUT_DIR}/chkpnt30000.pth \
      --feature_level {level}
```

```python
# === Cell 8: Smoke Test ===
# [See Pattern 3 in Architecture Patterns above]
```

### LERF Dataset Download Link (confirmed)

- **LERF dataset (with COLMAP data):** https://drive.google.com/file/d/1QF1Po5p5DwTjFHu6tnTeYs_G0egMVmHt/view
- **Pre-trained models:** https://drive.google.com/drive/folders/1ASFXWOwaXP_aSXV2iMDmEfILaDXQXlrE
- **3D-OVS dataset:** https://drive.google.com/drive/folders/1kdV14Gu5nZX6WOPbccG7t7obP_aXkOuC

### SAM Checkpoint Download

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# Size: ~2.4 GB
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| LangSplat environment.yml (Python 3.7, PyTorch 1.12.1, CUDA 11.6) | pip install PyTorch 2.0.1+cu118 on Colab | 2024-2025 (community workaround) | env.yml is outdated; pip approach is more reliable on modern Colab |
| COLMAP build from source on Colab | apt-get install colmap (CPU SIFT) | COLMAP 3.9+ (apt packages) | Building from source was fragile; apt is reliable if GPU SIFT is not needed |
| Conda env on Colab | pip-only OR condacolab | 2023-2024 (condacolab matured) | Conda on Colab requires condacolab + kernel restart; pip is simpler |
| LangSplat V1 (original, CVPR 2024) | LangSplat V2 (NeurIPS 2025, 450+ FPS) | 2025 | V2 is faster but less documented; V1 is safer for hackathon |

**Deprecated/outdated:**
- LangSplat environment.yml: Pins Python 3.7.13 and PyTorch 1.12.1. Both are incompatible with current Colab. Do NOT use `conda env create --file environment.yml`.
- SAM 2/3: LangSplat's segment-anything-langsplat fork is SAM 1 only. Do NOT substitute SAM 2.

## Timing Estimates

| Stage | Estimated Time (A100) | Notes |
|-------|----------------------|-------|
| Environment setup (pip install + compile CUDA extensions) | 5-10 min | Compilation of langsplat-rasterization and simple-knn |
| SAM checkpoint download (to Drive, first time) | 2-3 min | 2.4 GB; skip on subsequent runs |
| LERF dataset download + extract | 5-10 min | Download to Drive; skip on subsequent runs |
| COLMAP on 50-80 images (CPU SIFT) | 10-20 min | CPU feature extraction is the bottleneck |
| SAM + CLIP preprocessing (50-80 images) | 15-30 min | SAM mask generation + CLIP encoding per image |
| Autoencoder training (100 epochs) | 10-15 min | Small network, fast convergence |
| 3DGS RGB training (30K iterations) | 30-40 min | Well-established timing for room-scale scenes |
| LangSplat training (per feature level) | 20-40 min | 3 levels = 60-120 min total |
| **Total pipeline (one scene)** | **~2-4 hours** | Matches D-02 constraint |

**Staggered timeline for hackathon:**
- Minute 0: Start Cell 0-2 (setup + LERF download)
- Minute 10: Start figurines COLMAP + full pipeline
- Minute 30: Scaniverse scan complete, upload photos to Drive
- Minute 35: Start venue COLMAP + full pipeline (parallel notebook or sequential after figurines)
- Hour 2-3: First completed scene with valid smoke test
- Hour 3-4: Second scene complete (if both run to completion)

## Open Questions

1. **Exact LangSplat output PLY format for Phase 2 consumption**
   - What we know: PLY contains XYZ, color, opacity, covariance, and 3-dim language latent per Gaussian
   - What's unclear: Exact property names in the PLY header for the language latents (are they `f_language_0`, `f_language_1`, `f_language_2`?)
   - Recommendation: After first successful training, inspect PLY header with `plyfile` to confirm property names. This feeds directly into Phase 2's PLY loader.

2. **Scaniverse photo export resolution and format**
   - What we know: Scaniverse saves raw frames as photos; resolution depends on iPhone camera
   - What's unclear: Whether exported photos include EXIF data that COLMAP needs for camera intrinsics, or if COLMAP will need `--ImageReader.single_camera 1`
   - Recommendation: Test Scaniverse export -> COLMAP flow BEFORE the hackathon on a test scene. This is flagged as a blocker in STATE.md.

3. **autoencoder.pth location relative to PLY**
   - What we know: autoencoder.pth is saved in the output directory; Phase 2 needs it to decode 3-dim latents to 512-dim
   - What's unclear: Whether LangSplat training copies autoencoder.pth into the same output directory as the PLY or if it stays in the ae_ckpt directory
   - Recommendation: After training, verify autoencoder.pth location and include explicit copy step in upload cell if needed.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Python assert statements + numpy (inline in Colab cells) |
| Config file | None -- validation is embedded in notebook cells |
| Quick run command | Run Cell 8 (smoke test) |
| Full suite command | Run Cells 0-8 end-to-end |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TRAIN-01 | 5-stage pipeline completes on Colab A100 | smoke | Assert statements after each cell: sparse/ exists, language_features/ populated, chkpnt30000.pth exists, autoencoder.pth exists | No -- Wave 0 |
| TRAIN-02 | PLY + .npy + autoencoder.pth co-located | smoke | `os.path.exists()` checks for all three artifact paths | No -- Wave 0 |
| TRAIN-03 | Fallback scene on Google Drive | smoke | `os.path.exists(LERF_PATH + "/figurines/sparse/0/cameras.bin")` | No -- Wave 0 |
| TRAIN-04 | Live scan trained during hackathon | manual | Visual verification of Scaniverse photos uploaded + pipeline started | N/A -- manual |

### Sampling Rate
- **Per cell execution:** Assert statements verify stage output before proceeding to next stage
- **Per complete training run:** Full smoke test (Cell 8) with cosine similarity verification
- **Phase gate:** Smoke test returns non-uniform cosine similarity scores (std > 0.05) for at least two distinct object queries

### Wave 0 Gaps
- [ ] Assert statements after each pipeline stage (Cells 3-7) verifying intermediate artifacts exist and are non-empty
- [ ] Cell 8 smoke test script: load autoencoder, decode sample features, encode test queries, compute and display cosine similarity matrix
- [ ] Artifact size sanity checks: PLY file > 10 MB, .npy files > 1 KB each, autoencoder.pth > 100 KB

## Project Constraints (from CLAUDE.md)

- **Package manager:** pnpm for frontend (not applicable to Colab notebook)
- **Error handling:** Always handle errors with try/catch -- apply to notebook cells with assert + clear error messages
- **No hardcoded API keys:** InsForge S3 upload must use environment variables
- **Conventional commits:** `feat:`, `fix:`, `chore:`, `docs:` prefixes
- **Keep files under 300 lines:** Notebook cells should be focused; split complex cells
- **Git:** commit_docs is enabled; commit this research file

## Sources

### Primary (HIGH confidence)
- [LangSplat README](https://github.com/minghanqin/LangSplat/blob/main/README.md) -- pipeline stages, dataset links, command-line arguments
- [LangSplat process.sh](https://github.com/minghanqin/LangSplat/blob/main/process.sh) -- exact training commands for all stages
- [LangSplat preprocess.py](https://github.com/minghanqin/LangSplat/blob/main/preprocess.py) -- confirmed CLIP model (ViT-B-16, laion2b_s34b_b88k), SAM vit_h, output format
- [LangSplat autoencoder/train.py](https://github.com/minghanqin/LangSplat/blob/main/autoencoder/train.py) -- encoder dims [256,128,64,32,3], decoder dims [16,32,64,128,256,256,512], 100 epochs, Adam optimizer
- [LangSplat environment.yml](https://github.com/minghanqin/LangSplat/blob/main/environment.yml) -- Python 3.7.13, PyTorch 1.12.1, CUDA 11.6 (OUTDATED but confirms base requirements)
- [COLMAP installation docs](https://colmap.github.io/install.html) -- apt package, GPU SIFT limitations
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html) -- session limits, idle timeout
- [SAM checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) -- 2.4 GB, direct download

### Secondary (MEDIUM confidence)
- [Gaussian Splatting Colab fork](https://github.com/yuneko1127/gaussian-splatting-colab) -- Colab-specific setup patterns without conda
- [COLMAP on Colab gist](https://gist.github.com/kwea123/f0e8f38ff2aa94495dbfe7ae9219f75c) -- apt-get dependencies, build instructions
- [condacolab](https://github.com/conda-incubator/condacolab) -- conda installation on Colab if pip-only approach fails
- [3DGS training time benchmarks](https://github.com/graphdeco-inria/gaussian-splatting) -- 30-40 min for 30K iterations confirmed
- [Scaniverse support](https://scaniverse.com/support) -- export formats, raw data option
- [Scaniverse raw data blog](https://scaniverse.com/news/save-raw-data-for-future-processing) -- RGB frames preserved in raw data

### Tertiary (LOW confidence)
- Training time for SAM + CLIP preprocessing: estimated 15-30 min for 50-80 images based on SAM mask generation speed (~5-10 sec/image) + CLIP encoding. Not directly benchmarked for LangSplat on A100.
- Scaniverse photo export -> COLMAP compatibility: inferred from Scaniverse exporting standard JPEG photos with EXIF data. Not confirmed end-to-end. **Flagged as pre-hackathon validation item.**

## Metadata

**Confidence breakdown:**
- Standard stack: MEDIUM-HIGH -- PyTorch cu118 on Colab is a well-tested community pattern; LangSplat-specific version requirements confirmed from environment.yml and README
- Architecture: HIGH -- Pipeline stages, commands, and output structure confirmed from official repo (README, process.sh, preprocess.py, autoencoder/train.py)
- Pitfalls: HIGH -- Version mismatch, COLMAP limitations, session timeouts are all well-documented in GitHub issues and community posts
- Timing estimates: MEDIUM -- 3DGS training time is well-benchmarked; SAM preprocessing and LangSplat training times are extrapolated

**Research date:** 2026-04-04
**Valid until:** 2026-04-11 (hackathon is April 4-5; research is time-critical)
