import re
from collections import Counter, defaultdict
import pandas as pd

# -----------------------
# Fixed columns
# -----------------------
MISINFO_COL = "misinfo_type_final"
POST_TEXT_COL = "full_text"
NOTE_TEXT_COL = "noteText"




def build_model_label_regexes():
    """
    Per-label regexes so we can attribute counts to specific models/tools.
    Keep these conservative; add more labels if needed.
    """
    label_patterns = {
        "grok": r"\bgrok\b",
        "chatgpt": r"\bchat\s*gpt\b",
        "openai": r"\bopenai\b",
        "gpt": r"\bgpt\b",
        "gemini": r"\bgemini\b",
        "bard": r"\bbard\b",
        "claude": r"\bclaude\b",
        "perplexity": r"\bperplexity\b",
        "llama": r"\bllama(?:[\s\-\._]*)(?:2|3)?\b|\bllama\b",
        "deepseek": r"\bdeepseek\b",
        "mistral": r"\bmistral\b",
        "copilot": r"\bcopilot\b",
        "qwen": r"\bqwen\b",
        "midjourney": r"\bmidjourney\b",
        "dall-e": r"\bdall[·\- ]?e\b|\bdalle\b",
        "stable diffusion": r"\bstable\s*diffusion\b|\bstablediffusion\b",
        "leonardo ai": r"\bleonardo(?:[\s\-\._]+)ai\b",
        "sora": r"\bsora\b",
        "veo": r"\bveo\b",
        "nano banana": r"\bnano(?:[\s\-\._]+)banana\b",
    }

    return {lab: re.compile(pat, flags=re.IGNORECASE) for lab, pat in label_patterns.items()}


def _token_bounds(text, i):
    """Return (start,end) of the whitespace-delimited token containing index i."""
    n = len(text)
    s = i
    e = i
    while s > 0 and not text[s - 1].isspace():
        s -= 1
    while e < n and not text[e].isspace():
        e += 1
    return s, e


def classify_mention_context(text, m):
    """
    Classify match context using the token containing the match.
    Priority: url > at_mention > plain
    """
    s, e = _token_bounds(text, m.start())
    tok = text[s:e]
    tok_l = tok.lower()

    # URL-like token
    if (
        "http://" in tok_l
        or "https://" in tok_l
        or "www." in tok_l
        or ".com" in tok_l
        or "x.com/" in tok_l
    ):
        return "url"

    # @mention (e.g., @grok)
    rel = m.start() - s
    if rel > 0 and tok[rel - 1] == "@":
        return "at_mention"

    return "plain"

def model_mention_context_stats(series, label_res, top_k = 15):
    """
    Count per-label occurrences by context bucket (url / at_mention / plain).
    Returns a DataFrame with totals and shares.
    """
    counts = defaultdict(Counter)

    for text in series.fillna("").astype(str).values:
        for label, rx in label_res.items():
            for m in rx.finditer(text):
                ctx = classify_mention_context(text, m)
                counts[label][ctx] += 1
                counts[label]["total"] += 1

    rows = []
    for label, c in counts.items():
        total = c["total"]
        if total == 0:
            continue
        rows.append(
            {
                "label": label,
                "total": total,
                "url": c["url"],
                "at_mention": c["at_mention"],
                "plain": c["plain"],
                "url_share": round(c["url"] / total, 4),
                "at_mention_share": round(c["at_mention"] / total, 4),
                "plain_share": round(c["plain"] / total, 4),
            }
        )

    out = pd.DataFrame(rows).sort_values("total", ascending=False).head(top_k)
    return out



def build_regexes(max_gap_tokens=3):
    # --- Model / tool mentions (case-insensitive, separators allowed where relevant) ---

    # OpenAI / GPT family
    chatgpt_re = re.compile(r"\bchat\s*gpt\b", flags=re.IGNORECASE)
    openai_re = re.compile(r"\bopenai\b", flags=re.IGNORECASE)


    # Boolean (NO capturing group)
    gpt_bool_re = re.compile(
        r"\bgpt(?:[\s\-\._]*)"
        r"(?:(?:\d+(?:\.\d+)?[a-z]?))?"
        r"\b",
        flags=re.IGNORECASE,
    )


    other_llm_patterns = [
        r"\bgemini\b",
        r"\bbard\b",
        r"\bclaude\b",
        r"\bperplexity\b",
        r"\bllama\b|\bllama(?:[\s\-\._]*)(?:2|3)\b|\bllama(?:2|3)\b",
        r"\bgrok\b",
        r"\bdeepseek\b",
        r"\bmistral\b",
        r"\bcopilot\b",
        r"\bqwen\b",
    ]
    other_llm_re = re.compile("|".join(f"(?:{p})" for p in other_llm_patterns), flags=re.IGNORECASE)

    image_patterns = [
        r"\bmidjourney\b",
        r"\bdall[·\- ]?e\b|\bdalle\b",
        r"\bstable\s*diffusion\b|\bstablediffusion\b",
    ]
    image_re = re.compile("|".join(f"(?:{p})" for p in image_patterns), flags=re.IGNORECASE)

    video_patterns = [
        r"\bsora\b",
        r"\bveo\b",
        r"\bnano(?:[\s\-\._]+)banana\b",
    ]
    video_re = re.compile("|".join(f"(?:{p})" for p in video_patterns), flags=re.IGNORECASE)

    # Union for a single boolean "model mention"
    model_mention_re = re.compile(
        "|".join(
            [
                chatgpt_re.pattern,
                openai_re.pattern,
                gpt_bool_re.pattern,
                other_llm_re.pattern,
                image_re.pattern,
                video_re.pattern,
            ]
        ),
        flags=re.IGNORECASE,
    )

    # --- Generic AI-generation mentions (strict-ish proximity) ---
    verb = r"(?:generated|created|made|produced|synthesized)" # removed |used|using
    ai_agent = r"(?:ai|artificial\s+intelligence)"
    
    # Bounded character window (much safer/faster than token-gap regex)
    # Rough mapping: ~12 chars/token + some slack
    max_chars = 12 * max_gap_tokens + 20  


    generic_ai_re = re.compile(
        rf"(?:(?P<verb>{verb})[\s\S]{{0,{max_chars}}}(?P<agent>{ai_agent}))|"
        rf"(?:(?P<agent2>{ai_agent})[\s\S]{{0,{max_chars}}}(?P<verb2>{verb}))",
        flags=re.IGNORECASE,
    )

    
    return model_mention_re, generic_ai_re


def collect_hits(series, regex):
    hits = Counter()
    for val in series.fillna("").astype(str).values:
        for m in regex.finditer(val):
            hits[m.group(0).strip().lower()] += 1   # <-- strip added
    return hits


def collect_generic_hits_unique_per_row(series, generic_re):
    hits = Counter()
    for val in series.fillna("").astype(str).values:
        seen = set()
        for m in generic_re.finditer(val):
            verb = m.group("verb") or m.group("verb2")
            agent = m.group("agent") or m.group("agent2")
            if not verb or not agent:
                continue
            verb = verb.lower()
            agent = "ai" if "artificial" not in agent.lower() else "artificial intelligence"
            seen.add(f"{agent} + {verb}")
        for key in seen:
            hits[key] += 1
    return hits


def show_top(counter, title, k = 20):
    print(f"\n=== Top matches: {title} (top {k}) ===")
    for s, cnt in counter.most_common(k):
        print(f"{cnt:>6}  {s}")


def count_ai_signals(
    df,
    max_gap_tokens = 3,
    note_text_only = False,
    top_k = 20,
):
    model_re, generic_re = build_regexes(max_gap_tokens=max_gap_tokens)

    df = df.copy()

    # Normalize only what we need
    df[NOTE_TEXT_COL] = df[NOTE_TEXT_COL].fillna("").astype(str)

    if not note_text_only:
        df[POST_TEXT_COL] = df[POST_TEXT_COL].fillna("").astype(str)
    else:
        # ensure column exists for code paths that might reference it
        if POST_TEXT_COL not in df.columns:
            df[POST_TEXT_COL] = ""

    # ---- NOTE booleans ----
    df["note_model_mention"] = df[NOTE_TEXT_COL].str.contains(model_re, na=False)

    df["note_generic_ai_mention"] = df[NOTE_TEXT_COL].str.contains(generic_re, na=False)

    df["note_any_ai_signal"] = df["note_model_mention"] | df["note_generic_ai_mention"]

    # ---- POST booleans (optional) ----
    if not note_text_only:
        df["post_model_mention"] = df[POST_TEXT_COL].str.contains(model_re, na=False)

        df["post_generic_ai_mention"] = df[POST_TEXT_COL].str.contains(generic_re, na=False)

        df["post_any_ai_signal"] = df["post_model_mention"] | df["post_generic_ai_mention"]
    else:
        df["post_model_mention"] = False
        df["post_generic_ai_mention"] = False
        df["post_any_ai_signal"] = False

    n = len(df)

    # ---- Overall tables ----
    overall = pd.DataFrame(
        {
            "post": [
                int(df["post_model_mention"].sum()),
                int(df["post_generic_ai_mention"].sum()),
                int(df["post_any_ai_signal"].sum()),
            ],
            "note": [
                int(df["note_model_mention"].sum()),
                int(df["note_generic_ai_mention"].sum()),
                int(df["note_any_ai_signal"].sum()),
            ],
        },
        index=["model_mention", "generic_ai_generation_mention", "any_ai_signal"],
    )

    print("\n=== Overall counts (rows=signal, cols=source) ===")
    print(overall.to_string())

    print("\n=== Overall rates (fraction of rows) ===")
    print((overall / n).round(4).to_string())

    # Extra: composition among AI-signaled rows (helpful sanity)
    note_ai_n = int(df["note_any_ai_signal"].sum())
    if note_ai_n > 0:
        note_model_share = df.loc[df["note_any_ai_signal"], "note_model_mention"].mean()
        note_generic_share = df.loc[df["note_any_ai_signal"], "note_generic_ai_mention"].mean()
        print("\n=== NOTE AI-signal composition (among rows with any AI signal) ===")
        print(f"model_mention share  : {note_model_share:.4f}")
        print(f"generic_mention share: {note_generic_share:.4f}")

    if not note_text_only:
        post_ai_n = int(df["post_any_ai_signal"].sum())
        if post_ai_n > 0:
            post_model_share = df.loc[df["post_any_ai_signal"], "post_model_mention"].mean()
            post_generic_share = df.loc[df["post_any_ai_signal"], "post_generic_ai_mention"].mean()
            print("\n=== POST AI-signal composition (among rows with any AI signal) ===")
            print(f"model_mention share  : {post_model_share:.4f}")
            print(f"generic_mention share: {post_generic_share:.4f}")

    # ---- Per misinformation type ----
    agg = (
        df.groupby(MISINFO_COL)
        .agg(
            n=("note_any_ai_signal", "size"),
            post_model=("post_model_mention", "sum"),
            post_generic=("post_generic_ai_mention", "sum"),
            post_any=("post_any_ai_signal", "sum"),
            note_model=("note_model_mention", "sum"),
            note_generic=("note_generic_ai_mention", "sum"),
            note_any=("note_any_ai_signal", "sum"),
        )
        .reset_index()
        .sort_values("n", ascending=False)
    )

    for col in ["post_model", "post_generic", "post_any", "note_model", "note_generic", "note_any"]:
        agg[col + "_rate"] = (agg[col] / agg["n"]).round(4)

    print("\n=== Per misinformation type (counts) ===")
    print(
        agg[
            [MISINFO_COL, "n", "post_model", "post_generic", "post_any", "note_model", "note_generic", "note_any"]
        ].to_string(index=False)
    )

    print("\n=== Per misinformation type (rates) ===")
    print(
        agg[
            [
                MISINFO_COL,
                "n",
                "post_model_rate",
                "post_generic_rate",
                "post_any_rate",
                "note_model_rate",
                "note_generic_rate",
                "note_any_rate",
            ]
        ].to_string(index=False)
    )

    # ---- Holistic top-match summaries on FULL DATA ----

    if not note_text_only:
        show_top(collect_hits(df[POST_TEXT_COL], model_re), "POST model/tool mentions", k=top_k)
        show_top(collect_generic_hits_unique_per_row(df[POST_TEXT_COL], generic_re), "POST generic AI-generation mentions", k=top_k)

    show_top(collect_hits(df[NOTE_TEXT_COL], model_re), "NOTE model/tool mentions", k=top_k)

    
    
    
    label_res = build_model_label_regexes()

    # Extra: how much each specific model label is covered among
    # notes that have any model/tool mention.
    note_model_n = int(df["note_model_mention"].sum())
    if note_model_n > 0:
        note_model_label_rows = []
        note_text_series = df[NOTE_TEXT_COL].fillna("").astype(str)
        note_mask = df["note_model_mention"]
        for label, rx in label_res.items():
            cnt = int(note_text_series.loc[note_mask].str.contains(rx, na=False).sum())
            note_model_label_rows.append(
                {"label": label, "count": cnt, "percent_of_note_model_mention": round(cnt / note_model_n, 4)}
            )
        note_model_label_df = pd.DataFrame(note_model_label_rows).sort_values("count", ascending=False).reset_index(drop=True)
        print("\n=== NOTE model/tool label coverage (of notes with note_model_mention) ===")
        print("=== notes with at least one match per label (a note may match several labels) ===")
        print(note_model_label_df.to_string(index=False))
    else:
        print("\n=== NOTE model/tool label coverage (of notes with note_model_mention) ===")
        print("No note_model_mention matches found.")

    if not note_text_only:
        print("\n=== POST model/tool mention context stats (top labels) ===")
        ctx_post = model_mention_context_stats(df[POST_TEXT_COL], label_res, top_k=15)
        print(ctx_post.to_string(index=False))

    print("\n=== NOTE model/tool mention context stats (top labels) ===")
    ctx_note = model_mention_context_stats(df[NOTE_TEXT_COL], label_res, top_k=15)
    print(ctx_note.to_string(index=False))
    
    


    show_top(collect_generic_hits_unique_per_row(df[NOTE_TEXT_COL], generic_re), "NOTE generic AI-generation mentions", k=top_k)


    return df, agg, overall


if __name__ == "__main__":
    MAX_GAP_TOKENS = 7  
    df = pd.read_csv("path/to/dataset.csv", low_memory=False)  
    df, agg, overall = count_ai_signals(df, max_gap_tokens=MAX_GAP_TOKENS, note_text_only=True)
