# PARAKH pilot — SIH 2026, PS SIH26228 (Team Mavericks)

Small, reproducible pilot behind the numbers on slide 4 of our Stage 1 deck.
It runs fully offline: the only data is scikit-learn's bundled `digits` set.

**Scope, stated honestly:** this pilots the *statistical machinery* (per-contributor
permutation test + Benjamini–Hochberg) with a label-poisoning attack on a toy dataset,
and the *inference seal* (Ed25519 over RFC 8785 JSON, nonce store, sequence numbers,
signed checkpoints). Trigger/backdoor attacks and GTSRB / VisDrone runs come next.

## Run

```bash
pip install -r requirements.txt
python pilot.py        # seal tests + power curve + 200 all-clean runs  (~30 s)
python clean1000.py    # 1,000 all-clean runs for a tighter false-alarm interval (~30 s)
```

Seeds are fixed, so the output should match `results_pilot.txt` and `results_clean1000.txt`.

## What it measures

| Part | Test | Result |
|---|---|---|
| A. Seal | clean sealed records accepted | 1,000 / 1,000 |
| A. Seal | one-field edits (label, score, box, model digest, preprocessing config, one image bit) rejected | 1,000 / 1,000 |
| A. Seal | replays rejected by the nonce store | 200 / 200 |
| A. Seal | tail deletions caught by the signed checkpoint | 100 / 100 |
| C. Seal | edit image + recompute hashes + re-sign with attacker's own key, rejected | 1,000 / 1,000 |
| B. Contributor test | poisoned contributor flagged at 2 / 5 / 10 / 15 / 20 % label poison | 1.0 / 5.5 / 57.5 / 95.5 / 100 % |
| B. Contributor test | all-clean runs with any honest contributor flagged | 27 / 1,000 = 2.7 % (95 % CI 1.9–3.9 %) |

Setup for B: 10 simulated contributors × 89 images; k-NN (k = 7) label-disagreement
detector, threshold set on a separate calibration split; false-positive rate re-measured on
a held-out clean split; one contributor has a dimmer camera (honest drift) to test fairness.

## Files
- `pilot.py`: parts A, B, C
- `clean1000.py`: extra 1,000 all-clean runs
- `results_*.txt`: captured outputs
