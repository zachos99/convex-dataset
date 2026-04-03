# Misinformation labeling pipeline

Short scripts that label Community Note rows by misinformation type: `ai_generated`, `edited`, `miscaptioned`, or `other`.

## Run order

1. `**misinformation_keywords.py**` — Regex rules on `noteText` → column `misinfo_type_keys`.
2. `**misinformation_gemma.py**` (`RERUN = False`) — Multimodal model on post text, note, images → `misinfo_type_llm`, `confidence`, `rationale`.
3. `**find_disagreements.py**` — Rows where `misinfo_type_keys != misinfo_type_llm`, skipping rows whose keyword label is `other` → disagreement CSV.
4. `**misinformation_gemma.py**` (`RERUN = True`) — Second pass on the disagreement file; uses keyword + first LLM as hints → `misinfo_type_llm_rerun`, `confidence_rerun`, `rationale_rerun`.
5. `**misinformation_final_label.py**` — Fuses keys, first LLM, and optional rerun → `misinfo_type_final`, `misinfo_type_final_source_flag`.

**Merge step:** The rerun script only outputs rows you sent it. Join those rerun columns back into the **full** post–first-pass CSV (e.g. on a stable row id) before running `misinformation_final_label.py`, so every row has `misinfo_type_llm_rerun` where a second pass was run.

## Files


| Script                          | Role                                               |
| ------------------------------- | -------------------------------------------------- |
| `misinformation_keywords.py`    | Keyword / pattern classifier on `noteText`.        |
| `misinformation_gemma.py`       | Google GenAI calls; first pass and optional rerun. |
| `find_disagreements.py`         | Export keyword vs. LLM mismatches.                 |
| `misinformation_final_label.py` | Rule-based fusion of the three signals.            |


## Configuration

- Set paths in each script’s main block (or import and call the functions).
- `misinformation_gemma.py`: Configure `GOOGLE_AI_STUDIO_API_KEY` in the environment; set `BASE_MEDIA_DIR` to the folder that contains image paths listed in the `media` column.

