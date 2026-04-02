# CONVEX — Source Code Overview

Code supporting **"The Synthetic Shift: Tracking the Rise, Virality, and Detectability of AI-Generated Multimodal Misinformation"**.

CONVEX (_Community Notes for Visual Misinformation on X_) is a large-scale dataset of multimodal misinformation — miscaptioned, edited, and AI-generated images and videos — collected from X's Community Notes. The codebase covers the full lifecycle: dataset construction, misinformation annotation, longitudinal analysis, and benchmark evaluation of AI-image detectors.

---

## Repository map

| Path                        | Paper section                               | Purpose                                                                                                                                                                                                       |
| --------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `renew_data/`               | §3 — Dataset Construction                   | End-to-end pipeline: Community Notes TSV → filtered notes → tweet/media retrieval → misinformation labels → dataset merge. Designed for incremental renewal as new Community Notes releases become available. |
| `misinformation/`           | §3.2 — Data Annotation                      | Hybrid weakly supervised labeling (keyword rules + Gemma 3 VLM + majority-vote fusion) that classifies each note–post pair as _miscaptioned_, _edited_, _AI-generated_, or _other_.                           |
| `plots_volume.py`           | §4 — Evolution of Multimodal Misinformation | Weekly volume plots of Community Notes by misinformation category, with generative-model release markers (DALL·E 3, Sora, Veo, etc.).                                                                         |
| `plots_virality_share.py`   | §5.1 — Virality Share                       | Computes and plots the Virality Share metric V(c,m) — how over- or under-represented each category is among viral posts (top percentile).                                                                     |
| `plots_engagement_index.py` | §5.2 — Engagement Dynamics                  | Monthly z-score Engagement Index (retweets + replies − 2 × favorites) that separates active discourse from passive approval.                                                                                  |
| `consensus_calculations.py` | §6 — Consensus Dynamics                     | Merges note status history, computes consensus probability, notes-to-consensus, helpful share, and first-note reaction time per misinformation type.                                                          |
| `ai_model_references.py`    | §7 — AI References in Community Notes       | Regex-based extraction of AI model/tool mentions and generic AI-generation phrases from note and post text, with context classification (URL vs. plain text vs. @mention).                                    |
| `benchmark_evaluation/`     | §8 — Evaluation of Detection Systems        | Inference pipelines for three Synthetic Image Detectors (SPAI, RINE, B-Free) and three VLMs (Gemma 3, Grok, GPT-5-mini), plus comparative metrics and temporal-degradation plots.                             |

---

## High-level flow

```
Community Notes TSV
        │
        ▼
  ┌─────────────┐
  │ renew_data/  │  filter → fetch tweets/media → integrate
  └──────┬──────┘
         │
         ▼
  ┌──────────────┐
  │misinformation│  keyword + VLM labeling → final labels
  └──────┬───────┘
         │
         ▼
   CONVEX dataset CSVs  (image set + video set)
         │
    ┌────┼──────────┬──────────────┬─────────────┐
    ▼    ▼          ▼              ▼              ▼
 Volume  Virality & Consensus   AI Model      Benchmark
 Plots   Engagement Dynamics    References    Evaluation
 (§4)    (§5)       (§6)        (§7)          (§8)
```

Each subdirectory contains its own markdown file with detailed instructions. We recommend running each pipeline step independently: download the Community Notes TSV checkpoint from [X's public release page](https://x.com/i/communitynotes/download-data), configure file paths in each script, and set up the required credentials and API keys before proceeding to the next stage. Verify the output of each step before moving on — the per-module markdown files describe what to expect.
