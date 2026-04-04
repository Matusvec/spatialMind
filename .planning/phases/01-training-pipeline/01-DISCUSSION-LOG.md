# Phase 1: Training Pipeline - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-04
**Phase:** 1-training-pipeline
**Areas discussed:** Scene capture, Fallback plan, Timing & risk, Colab notebook

---

## Scene Capture

### Venue Space

| Option | Description | Selected |
|--------|-------------|----------|
| Hackathon room | The actual hacking space — tables, chairs, laptops, backpacks. Judges will be there. | ✓ |
| Lobby/common area | A furnished common area with varied furniture — couches, shelves, lamps | |
| Not sure yet | Decide when arriving based on what looks good | |

**User's choice:** Hackathon room
**Notes:** Judges will physically be in the space — recognition adds demo impact.

### Photo Count

| Option | Description | Selected |
|--------|-------------|----------|
| 50-80 photos | Good balance — solid reconstruction, training under 3 hours | ✓ |
| 100-150 photos | Better quality but training could push 4-5 hours | |
| You decide | Claude picks based on scene complexity | |

**User's choice:** 50-80 photos

---

## Fallback Plan

### Backup Scene

| Option | Description | Selected |
|--------|-------------|----------|
| figurines | lerf_figurines — tabletop scene with varied small objects | ✓ |
| teatime | Tea set scene — clear on_top_of relationships | |
| You decide | Claude picks whichever has best spatial relationships | |

**User's choice:** figurines (lerf_figurines)

### Switch Point

| Option | Description | Selected |
|--------|-------------|----------|
| Hour 8 | Switch to fallback if no output by 1/3 of hackathon | |
| Hour 12 | Give it until halfway | |
| Parallel from start | Download fallback AND start live training. Use whichever finishes first. | ✓ |

**User's choice:** Parallel from start

---

## Timing & Risk

### Start Time

| Option | Description | Selected |
|--------|-------------|----------|
| Minute 0 | First thing — use pre-captured scene photos | |
| After scan | Capture room first (~30 min), then start | |
| Both staggered | Start fallback at minute 0, venue after scan at ~30 min | ✓ |

**User's choice:** Both staggered

### Session Death Recovery

| Option | Description | Selected |
|--------|-------------|----------|
| Checkpoint resume | Save checkpoints to Drive every 5K iterations — resume from last | ✓ |
| Restart fresh | Just restart with fewer iterations if time is short | |
| Switch to fallback | Abandon live training and use figurines dataset | |

**User's choice:** Checkpoint resume

---

## Colab Notebook

### Structure

| Option | Description | Selected |
|--------|-------------|----------|
| All-in-one | One notebook with sequential cells for each stage | ✓ |
| Modular scripts | Separate .py scripts, notebook orchestrates | |
| You decide | Claude structures for fastest execution and debugging | |

**User's choice:** All-in-one notebook

### Output Destination

| Option | Description | Selected |
|--------|-------------|----------|
| Drive only | Stay on Drive, download manually | |
| Drive + InsForge | Auto-upload to InsForge S3 after completion | ✓ |
| Drive + local | Auto-download via Drive sync | |

**User's choice:** Drive + InsForge

### Monitoring

| Option | Description | Selected |
|--------|-------------|----------|
| Colab output | Check Colab tab occasionally for training loss | ✓ |
| Drive markers | Write status file to Drive after each stage | |
| You decide | Claude picks simplest monitoring | |

**User's choice:** Colab output

---

## Claude's Discretion

- Training hyperparameters (iterations, learning rate)
- COLMAP configuration for indoor scenes
- CLIP backbone choice
- Scaniverse export settings

## Deferred Ideas

None — discussion stayed within phase scope
