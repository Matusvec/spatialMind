---
phase: 1
slug: training-pipeline
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-04-04
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Python assert statements + numpy (inline in Colab cells) + standalone scripts |
| **Config file** | None — validation embedded in notebook cells and scripts |
| **Quick run command** | `python3 scripts/verify_artifacts.py <scene_dir>` |
| **Full suite command** | `python3 scripts/smoke_test.py <scene_dir>` |
| **Estimated runtime** | ~30 seconds (smoke test on pre-trained artifacts) |

---

## Sampling Rate

- **After every task commit:** Run `python3 -c "import json; nb=json.load(open('colab/spatialMind_training.ipynb')); assert len(nb['cells']) >= 10"`
- **After every plan wave:** Run `python3 scripts/verify_artifacts.py <scene_dir>`
- **Before `/gsd:verify-work`:** Full smoke test must pass
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 1 | TRAIN-01, TRAIN-02, TRAIN-03 | smoke | `python3 -c "import json; nb=json.load(open('colab/spatialMind_training.ipynb')); assert len(nb['cells']) >= 10"` | ❌ W0 | ⬜ pending |
| 01-01-02 | 01 | 1 | TRAIN-01, TRAIN-02 | smoke | `python3 -c "import ast; ast.parse(open('scripts/smoke_test.py').read())"` | ❌ W0 | ⬜ pending |
| 01-02-01 | 02 | 2 | TRAIN-04 | manual | Visual verification of runbook content | ❌ W0 | ⬜ pending |
| 01-02-02 | 02 | 2 | TRAIN-01 | manual | Human verifies Cells 0-1 run on Colab | N/A | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `colab/spatialMind_training.ipynb` — 10-cell notebook with inline asserts after each stage
- [ ] `scripts/smoke_test.py` — Loads autoencoder, decodes sample features, computes cosine similarity
- [ ] `scripts/verify_artifacts.py` — Checks PLY, .npy, and autoencoder.pth exist and pass size checks

*All created in Wave 1 (Plan 01-01).*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Scaniverse capture + upload | TRAIN-04 | Requires physical iPhone + Colab browser | Follow hackathon-runbook.md Section 1-3 |
| Colab Cells 0-1 execute | TRAIN-01 | Requires Colab A100 runtime | Open notebook in Colab, run Cells 0-1, verify no errors |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 30s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
