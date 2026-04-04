# SpatialMind Hackathon Runbook -- JacHacks April 4-5, 2026

This runbook provides the exact steps and timing for the LangSplat training phase during the hackathon. Follow it top-to-bottom. Every minute counts.

---

## Pre-Hackathon Checklist

- [ ] Google Colab Pro session confirmed (A100 GPU access)
- [ ] Scaniverse installed on iPhone
- [ ] Google Drive has at least 20 GB free space
- [ ] Test Scaniverse -> photo export -> COLMAP flow on a small test scene (5-10 photos)
- [ ] Notebook `colab/spatialMind_training.ipynb` uploaded to Google Drive
- [ ] LERF dataset pre-downloaded to Drive (run Cells 0-2 before hackathon if possible)
- [ ] SAM checkpoint pre-downloaded to Drive

---

## Minute-by-Minute Timeline

```
Minute 0:    Open Colab notebook, connect to A100 runtime
             Run Cell 0 (Mount Drive)
             Run Cell 1 (Environment setup -- ~5-10 min)
             Run Cell 2 (SAM + LERF download -- skip if pre-downloaded)

Minute 10:   Set SCENE = "figurines"
             Run Cells 3-7 for figurines (fallback training -- ~2 hours total)
             (per D-05, D-06: fallback starts at minute 0, runs in parallel)

Minute 15:   Walk to hackathon venue area
             Select area with clearly distinct objects at different heights
             (per D-01: scan the actual room -- judges will be there)
             (per D-02: capture 50-80 photos with Scaniverse)

Minute 30:   Scaniverse scan complete
             Export photos from Scaniverse (standard photo export)
             Upload photos to Google Drive: /spatialMind/data/jachacks_venue/input/

Minute 35:   In Colab (new tab or after figurines completes):
             Set SCENE = "jachacks_venue"
             Update DATASET_PATH
             Run Cells 3-7 for venue scene

Hour 2-3:    First scene (figurines) should complete
             Run Cell 8 (smoke test) on figurines
             If smoke test passes: figurines is ready as fallback

Hour 3-4:    Venue scene should complete
             Run Cell 8 (smoke test) on venue
             If passes: switch to venue for demo
             If fails: use figurines fallback

Hour 4+:     Training phase complete -- move to Phase 2 (Semantic Query Server)
```

---

## Scaniverse Capture Guide (per D-01, D-02, D-03)

- Use iPhone with Scaniverse app
- Select "Photo Mode" (not LiDAR scan mode -- we need raw photos for COLMAP)
- Walk slowly around the target area, capturing 50-80 photos
- Overlap between photos: ~60-70% (needed for COLMAP feature matching)
- Include objects at multiple heights: floor items, table items, wall items
- Avoid: featureless walls, reflective surfaces, moving people
- Good subjects: tables with items on them, chairs, monitors, backpacks, water bottles
- Export: Share -> Save Photos -> Upload to Google Drive `/spatialMind/data/jachacks_venue/input/`

### What Makes a Good Scan Area

The three-beat demo needs objects with clear spatial relationships. Look for:

1. **Vertical stacking** -- items on top of tables, monitors on desks (enables "on_top_of")
2. **Lateral proximity** -- items next to each other (enables "next_to")
3. **Height variety** -- floor items, table items, wall items (enables "above" / "below")
4. **Distinct objects** -- not just flat walls (enables interesting queries like "find the backpack")

---

## Scene Switching in Notebook

```python
# To switch from figurines to venue:
SCENE = "jachacks_venue"  # was "figurines"
DATASET_PATH = f"{BASE}/data/{SCENE}"
# Then re-run Cells 3-7 (Cell 2 doesn't need re-running)
```

**Important:** When switching scenes, you must re-run Cells 3 through 7 in order. Cell 0 (Drive mount) and Cell 1 (environment setup) do NOT need re-running. Cell 2 (SAM + LERF download) does NOT need re-running either -- it only fetches shared resources.

---

## Cell Reference

| Cell | Name | Duration | Notes |
|------|------|----------|-------|
| 0 | Mount Drive | ~5 sec | Must be first. All paths depend on it. |
| 1 | Environment Setup | ~5-10 min | PyTorch 2.0.1+cu118, LangSplat clone, CUDA extensions |
| 2 | SAM + LERF Download | ~5-15 min | 2.4 GB SAM checkpoint + LERF dataset. Skip if pre-downloaded. |
| 3 | COLMAP SfM | ~5-20 min | Camera pose extraction. Skips for pre-computed LERF scenes. |
| 4 | SAM + CLIP Preprocessing | ~15-30 min | Hierarchical masks + 512-dim CLIP features per image. |
| 5 | Autoencoder Training | ~10-20 min | 512-dim to 3-dim compression. |
| 6 | 3DGS RGB Training | ~45-60 min | 30K iterations, checkpoints every 5K to Drive. |
| 7 | LangSplat Training | ~30-45 min | 3 feature levels on top of RGB checkpoint. |
| 8 | Smoke Test | ~1 min | Cosine similarity check. std > 0.05 = pass. |
| 9 | Consolidate Artifacts | ~10 sec | Copies PLY, autoencoder, dim3 features to artifacts/. |

---

## Troubleshooting

### Colab Disconnects Mid-Training

Reconnect, run Cell 0 (remount Drive), check for existing checkpoints in Cell 6, re-run from where it stopped. The notebook has skip-if-exists guards and checkpoint resume logic built in.

### COLMAP Fails (no cameras.bin)

Check that photos are in `{DATASET_PATH}/input/` directory. Verify photos have EXIF data (Scaniverse photos should). If feature matching fails:

```bash
# Try forcing single camera model
colmap feature_extractor --ImageReader.single_camera 1 ...
```

This tells COLMAP all images came from the same camera, which is true for Scaniverse exports.

### Smoke Test Fails (std < 0.05)

Embeddings are noise. Check in this order:

1. Was the RGB checkpoint from Cell 6 used as `--start_checkpoint` for LangSplat training in Cell 7?
2. Did the autoencoder (Cell 5) train on the correct `language_features/` directory?
3. Try increasing 3DGS iterations from 30K to 40K (edit Cell 6)
4. Verify CLIP model used in smoke test matches training (both should use ViT-B-16)

### SAM Preprocessing Fails

Verify SAM checkpoint path matches `{BASE}/ckpts/sam_vit_h_4b8939.pth`. Check GPU memory -- SAM vit_h needs ~8 GB VRAM (A100 has 40+ GB, so this should not be an issue on Colab).

### "CUDA error: no kernel image"

PyTorch CUDA version mismatch. Re-run Cell 1 with the exact `torch==2.0.1+cu118` command. Do NOT let Colab auto-install a different PyTorch version.

### Out of Disk Space on Drive

The full pipeline for one scene needs ~5-10 GB. For two scenes running in parallel, budget 20 GB. Delete intermediate files if needed:

```python
# Free space by removing non-essential intermediate files
!rm -rf {DATASET_PATH}/output/{SCENE}/train/  # TensorBoard logs
```

---

## Success Criteria

The training phase is DONE when:

1. At least one scene (figurines OR venue) passes the smoke test in Cell 8
2. The passing scene's artifacts are on Google Drive at known paths
3. You can state: "PLY at [path], autoencoder at [path], dim3 features at [path]"

**Specifically, the artifacts directory should contain:**

```
{BASE}/data/{SCENE}/artifacts/
  point_cloud.ply          # Gaussians with XYZ, color, opacity, covariance
  autoencoder.pth          # Trained 512-dim to 3-dim autoencoder
  language_feature_dim3/   # Compressed 3-dim CLIP features per image
    *_f.npy
```

These artifacts feed directly into Phase 2 (Semantic Query Server), Phase 3 (JAC Spatial Graph), and Phase 5 (Frontend Rendering).

---

## Decision Quick Reference

| ID | Decision | Impact |
|----|----------|--------|
| D-01 | Scan actual hackathon room at UMich | Judges in the scene = demo impact |
| D-02 | Capture 50-80 photos | Good quality, training under 3 hours |
| D-03 | Scaniverse on iPhone, export photos | COLMAP-compatible pipeline |
| D-04 | Fallback scene: lerf_figurines | Pre-computed COLMAP, diverse objects |
| D-05 | Run fallback + live in parallel | Use whichever finishes first with valid output |
| D-06 | Staggered start: figurines at t=0, venue at t=30m | Maximize chance of having trained scene |
| D-07 | Checkpoint every 5K iterations to Drive | Crash recovery across Colab sessions |
| D-08 | Mount Drive as first cell | All artifacts survive session timeouts |
| D-09 | All-in-one sequential notebook | Single notebook, run top-to-bottom |
