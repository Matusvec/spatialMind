---
phase: 01-training-pipeline
plan: 02
subsystem: training
tags: [colab, langsplat, runbook, hackathon]

requires:
  - phase: 01-01
    provides: Colab notebook and verification scripts
provides:
  - Hackathon execution runbook with minute-by-minute timing
  - Human-verified Colab notebook ready for live use
affects: [phase-02-semantic-query-server]

tech-stack:
  added: []
  patterns: [runbook-driven hackathon execution]

key-files:
  created:
    - docs/hackathon-runbook.md
  modified: []

key-decisions:
  - "Staggered start: figurines fallback at minute 0, venue training after scan at ~30 min"
  - "Checkpoint every 5K iterations to Google Drive for session recovery"

patterns-established:
  - "Runbook pattern: step-by-step with exact timing, commands, and troubleshooting"
---

# Plan 01-02: Hackathon Runbook + Verification

## What Was Built

Created the hackathon execution runbook (`docs/hackathon-runbook.md`) with:
- Minute-by-minute timeline from hackathon start to submission
- Scaniverse capture guide (50-80 photos, coverage pattern)
- Scene switching instructions (figurines → venue)
- Checkpoint recovery procedures (5K iteration saves to Drive)
- Troubleshooting guide for 6 common failure modes (COLMAP, OOM, timeout, etc.)
- Pre-hackathon checklist

## Checkpoint Resolution

**Type:** human-verify
**User response:** Approved — user is already ahead of this checkpoint, running COLMAP on Colab A100 with frames extracted and about to start RGB training.

## Self-Check

- [x] docs/hackathon-runbook.md exists and contains all required sections
- [x] Runbook references Colab notebook cells correctly
- [x] Human verified notebook readiness (user is already running it live)

## Self-Check: PASSED

## Deviations

None — plan executed as written. User is ahead of schedule on actual training.
