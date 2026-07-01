# Redrob Candidate Ranking System — GenSonicImpact

This repository contains the candidate ranking pipeline for the Redrob hackathon. The system retrieves and ranks candidate profiles against a set of 7 key Job Description requirements using sentence embeddings, activity signals, and rule-based disqualifiers.

## 1. Setup Instructions
To install all required dependencies (CPU-only, offline-compatible), execute:
```bash
pip install -r requirements.txt
```

## 2. Precomputation Pipeline
To regenerate features, perform logical honeypot checks, and precompute candidate embeddings (caches stored under `cache/`), run these scripts in order:
```bash
python precompute/extract_features.py
python precompute/honeypot_checks.py
python precompute/embed_candidates.py
```
*Note: The candidate embedding step runs locally using SentenceTransformers `all-MiniLM-L6-v2` and takes approximately 41 minutes on an 8-core CPU.*

## 3. Reproduction Command
To run the fast ranking and reasoning engine and output the top 100 candidate ranking CSV, execute:
```bash
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
# Expected output: submission.csv — 100 rows, ranked 1-100, runs in ~64 seconds on CPU
```

## 4. Local Sandbox Dashboard
To launch the interactive Streamlit sandbox locally, run:
```bash
streamlit run sandbox_app.py
```
