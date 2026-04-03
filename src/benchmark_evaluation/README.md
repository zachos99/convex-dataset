# Benchmark evaluation

This folder contain scripts for the Benchmark Evaluation using CONVEX dataset’s **AI-image** and **miscaptioned (treated as “real”)** images.

## What’s in here

| Kind                          | Systems                                                                  | Main scripts                                                                                                       |
| ----------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| **Specialized detectors** (3) | **SPAI**, **RINE**, **B-Free**                                           | `spai_pipeline.py` + `spai_utils.py`, `rine_pipeline.py` + `rine_utils.py`, `bfree_pipeline.py` + `bfree_utils.py` |
| **Vision LLMs** (3)           | **GPT-5-mini** (OpenAI), **Grok** (xAI), **Gemma-3** (Google Gemini API) | `openai_inference.py`, `grok_xai_inference.py`, `gemma_inference.py`                                               |
| **Compare everything**        | All six above                                                            | `compare_plot_metrics.py` (one table + optional over-time plot)                                                    |

## Task

Each model answers: **“Is this image AI or REAL?”**  
Ground truth for the benchmark: **AI folder = positive**, **miscaptioned folder = negative** (real / not-AI for this setup). Metrics (recall, FPR, F1, etc.) come from comparing predictions to that split.

## Typical flow (high level)

1. **Run the external tool or API** so you have raw outputs (CSVs, JSONL, etc.) for both image sets.
   - Pipelines assume you already ran **SPAI** and **B-Free** (through their official implementation) elsewhere and point to their CSVs.
   - **RINE** can be run from `rine_pipeline.py` (needs e.g. `IMGBB_API_KEY` in `.env`).
   - **LLM** scripts call APIs: `CHATGPT_API_KEY`, `XAI_API_KEY`, `GEMINI_API_KEY` as documented in each file.

2. **Enrich** outputs by joining to your main image/tweet CSV (`dataset_csv` paths in the scripts—**you must set these to your real paths**).

3. **Evaluate** with `evaluate_*` functions in each `*_utils.py` / inference file, or run **`compare_plot_metrics.py`** after enriched CSVs exist.
