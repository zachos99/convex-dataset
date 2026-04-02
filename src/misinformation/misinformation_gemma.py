import pandas as pd
import json, os, time, re
import mimetypes
from pathlib import Path
from PIL import Image
from io import BytesIO
import dotenv  
from google import genai
from google.genai import types  

dotenv.load_dotenv()


"""
    Set the Google AI Studio API key
"""
GOOGLE_AI_STUDIO_API_KEY = os.getenv("GOOGLE_AI_STUDIO_API_KEY")


BASE_MEDIA_DIR = Path("../path/to/images")


def parse_media(media_raw):
    if pd.isna(media_raw) or not str(media_raw).strip():
        return []
    return [p.strip() for p in str(media_raw).split(",") if p.strip()]


def load_image_bytes(path, max_size=512):
    # jpg -> png fallback if jpg missing
    if not os.path.exists(path) and path.lower().endswith(".jpg"):
        alt = path[:-4] + ".png"
        if os.path.exists(alt):
            path = alt

    mime_type, _ = mimetypes.guess_type(path)
    if not mime_type:
        mime_type = "image/jpeg"

    with Image.open(path) as img:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        # Decide output format from mime guess
        out_format = "JPEG" if mime_type == "image/jpeg" else "PNG"

        # If saving JPEG, ensure RGB (avoid RGBA->JPEG crash)
        if out_format == "JPEG":
            if img.mode in ("RGBA", "LA") or ("A" in img.getbands()):
                img = img.convert("RGB")
            elif img.mode != "RGB":
                img = img.convert("RGB")

        buf = BytesIO()
        img.save(buf, format=out_format)
        return buf.getvalue(), mime_type


def parse_llm_json(response_text):
    if not response_text or not isinstance(response_text, str):
        return None

    txt = response_text.strip()

    # Strip leading ```json and trailing ``` with optional whitespace
    txt = re.sub(r"^\s*```(?:json)?\s*", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\s*```\s*$", "", txt).strip()

    # Extract the first JSON object by brace scanning (safer than first/last brace)
    start = txt.find("{")
    if start == -1:
        return None

    depth = 0
    end = None
    in_str = False
    esc = False

    for i in range(start, len(txt)):
        ch = txt[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break

    if end is None:
        return None

    json_str = txt[start:end+1]
    try:
        return json.loads(json_str)
    except Exception:
        return None




def mm_inference_google(
    model,
    system_prompt=None,
    user_prompt=None,
    image_paths=None,
    max_tokens=1024,
    temperature=0.2
):
    """
    Multimodal inference wrapper for Gemma and Gemini.
    
    - Combines system_prompt + user_prompt internally
    - Accepts 1 to 4 local image paths
    - Returns text output or None
    """

    # 1. Client initialization: This remains correct
    client = genai.Client(api_key=GOOGLE_AI_STUDIO_API_KEY)



    # Combine system + user prompt into a single string
    # Gemma does not support system prompts
    if system_prompt:
        full_prompt = f"{system_prompt.strip()}\n\n{user_prompt.strip()}"
    else:
        full_prompt = user_prompt.strip()

    # Construct user parts
    user_parts = [types.Part.from_text(text=full_prompt.strip())]

    if image_paths:
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        for path in image_paths:
            try:
                image_bytes, mime_type = load_image_bytes(path)
                user_parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))

            except Exception as e:
                print(f"❌ Loading or Part creation failed for an image '{path}': {e}")
                continue

    # Prepare contents
    contents = [types.Content(role="user", parts=user_parts)]

    config_args = {
        "temperature": temperature,
        "max_output_tokens": max_tokens
    }

    
    try:
        response = client.models.generate_content(
            model=model,         
            contents=contents, 
            config=types.GenerateContentConfig(**config_args)      
        )
        return response.text
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None


  


def gemma_prompt(full_text, note_text, image_paths):


    full_text = full_text or ""
    note_text = note_text or ""
    num_media = len(image_paths) if image_paths else 0
    
    system_prompt = ("""

        You are a misinformation analyst reviewing a post on X. You will see the post text, one or more media files and a Community Note written by other users. Your job is to classify the TYPE of media-related misinformation. 
        Use exactly ONE label from this set for the media:
            - ai_generated: The image or video is fully or partly created by an AI system (e.g. generated by AI, diffusion model, deepfake where the person or scene is synthetic; tools like Midjourney, DALL·E, Sora).
            - edited: The image or video is based on real footage but has been digitally manipulated (e.g. photoshopped, doctored, digitally altered, composite, splice, superimposed, face swap, objects added or removed, misleading crop or heavy visual edit).
            - miscaptioned: The image or video itself is essentially authentic and not significantly edited, but the post gives a false or misleading context (e.g. out of context, wrong context, miscaptioned, misattributed, old photo or video reused as if it were new, wrong time, wrong place, wrong event, wrong person, wrong causal link).
            - other: The Community Note discusses misinformation that is not mainly about the media or there is not enough evidence to decide which of the three media types applies.
        Very important instructions:
        - Treat the Community Note as the primary evidence: it usually explains why the post is misleading.
        - Use the post text and the media to cross-check and support or reject what the note says.
        - If multiple labels seem possible, choose the ONE that best describes the MAIN way the media misleads.
        - If there is not enough information to be sure, choose 'other'
        Output format (JSON):
        {{
          "misinfo_label": "ai_generated" | "edited" | "miscaptioned" | "other",
          "confidence": <number between 0 and 1>,
          "rationale": "short explanation, 1–2 sentences"
        }}
        Do not add any extra text before or after the JSON
    """)



    media_desc = f"There are {num_media} attached media file(s)."

    
    

    user_prompt = f"""
        COMMUNITY_NOTE:
        <<<
        {note_text}
        >>>
        
        POST_TEXT:
        <<<
        {full_text}
        >>>        

        MEDIA:
        <<<
        {media_desc}
        >>>
        """

    user_content = {
        "user_prompt": user_prompt,
        "image_paths": image_paths,
    }

    return system_prompt, user_content


def gemma_prompt_second_pass(full_text, note_text, image_paths, keyword_label, llm_label):


    full_text = full_text or "" 
    note_text = note_text or ""
    num_media = len(image_paths) if image_paths else 0
    media_desc = f"There are {num_media} attached media file(s)."

    
    system_prompt_second_pass = ("""

        You are a misinformation analyst reviewing a post on X. You will see the post text, one or more media files and a Community Note written by other users. Your job is to classify the TYPE of media-related misinformation. 
        Use exactly ONE label from this set for the media:
            - ai_generated: The image or video is fully or partly created by an AI system (e.g. generated by AI, diffusion model, deepfake where the person or scene is synthetic; tools like Midjourney, DALL·E, Sora).
            - edited: The image or video is based on real footage but has been digitally manipulated (e.g. photoshopped, doctored, digitally altered, composite, splice, superimposed, face swap, objects added or removed, misleading crop or heavy visual edit).
            - miscaptioned: The image or video itself is essentially authentic and not significantly edited, but the post gives a false or misleading context (e.g. out of context, wrong context, miscaptioned, misattributed, old photo or video reused as if it were new, wrong time, wrong place, wrong event, wrong person, wrong causal link).
            - other: The Community Note discusses misinformation that is not mainly about the media or there is not enough evidence to decide which of the three media types applies.
        Very important instructions:
        - Treat the Community Note as the primary evidence: it usually explains why the post is misleading.
        - Use the post text and the media to cross-check and support or reject what the note says.
        - If multiple labels seem possible, choose the ONE that best describes the MAIN way the media misleads.
        - If there is not enough information to be sure, choose 'other'
        Output format (JSON):
        {{
          "misinfo_label": "ai_generated" | "edited" | "miscaptioned" | "other",
          "confidence": <number between 0 and 1>,
          "rationale": "short explanation, 1–2 sentences"
        }}
        Do not add any extra text before or after the JSON
    """)



    user_prompt = f"""
        COMMUNITY_NOTE:
        <<<
        {note_text}
        >>>

        POST_TEXT:
        <<<
        {full_text}
        >>>

        MEDIA:
        <<<
        {media_desc}
        >>>

        PREVIOUS LABEL SUGGESTIONS (may be wrong and may disagree):
        - keyword_based_label: "{keyword_label}"
        - first_llm_label: "{llm_label}"

        TASK:
            1. Evaluate whether each suggested label fits the evidence.
            2. If one suggested label clearly matches the evidence better, choose it.
    """

    user_content = {
        "user_prompt": user_prompt,
        "image_paths": image_paths,
    }

    return system_prompt_second_pass, user_content



def extract_misinfo_batch(
    csv_path,
    model,
    temp,
    max_tokens,
    start=None,
    end=None,
    save_every=200,
    rerun=False,
    batch_tag=None,  # override batch suffix in output filename without re-slicing input
):

    print(f"Reading from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Slice the input only when start/end are provided (standalone / __main__ use).
    # When called from run_modality the input is already the batch file, so start/end
    # are None and no slicing happens — batch_tag is used for naming only.
    if start is not None and end is not None:
        df_iter = df.iloc[start:end].copy()
    else:
        df_iter = df.copy()

    base, ext = os.path.splitext(csv_path)
    # Prefer explicit batch_tag; fall back to deriving from start/end; empty if neither.
    if batch_tag is not None:
        _batch_tag = batch_tag
    elif start is not None and end is not None:
        _batch_tag = f"{start}_{end}"
    else:
        _batch_tag = ""

    if rerun:
        # Rerun pass: df_iter is the FULL (or batched) first-pass result.
        # Internally filter to disagreement rows so we only call Gemma on those,
        # then merge rerun columns back into the full slice at the end.
        rerun_cols = ['misinfo_type_llm_rerun', 'confidence_rerun', 'rationale_rerun', 'llm_rerun_response']
        for col in rerun_cols:
            if col not in df_iter.columns:
                df_iter[col] = None

        # Disagreement mask: keys != llm, excluding rows where keys is 'other' or empty.
        def _norm(s):
            v = str(s).strip().lower() if pd.notna(s) else ""
            return "" if v in ("nan", "none") else v

        keys_vals = df_iter['misinfo_type_keys'].map(_norm)
        llm_vals  = df_iter['misinfo_type_llm'].map(_norm)
        disagree_mask = (
            (keys_vals != "") & (keys_vals != "other") & (keys_vals != llm_vals)  # genuine label disagreement
        ) | (
            (keys_vals == "other") & (llm_vals == "")                             # keys=other but Gemma returned nothing → no final label
        )

        # df_full_slice keeps the complete slice for the final merged save.
        # df_iter is now narrowed to only the disagreement rows (same integer index).
        df_full_slice = df_iter
        df_iter = df_iter.loc[disagree_mask].copy()
        print(f"Rerun: {len(df_iter)} disagreement rows out of {len(df_full_slice)} (batch).")
    else:
        # first pass: create the main columns
        df_full_slice = df_iter
        df_iter['misinfo_type_llm'] = None
        df_iter['confidence'] = None
        df_iter['rationale'] = None
        df_iter['llm_response'] = None



    # Decide output filename — strip all known intermediate suffixes (with optional
    # _START_END batch tag) so every stage derives from the same clean stem.
    #   keys input:   ..._with_misinfo_keys[_S_E].csv
    #   rerun input:  ..._with_misinfo_gemma[_S_E].csv
    #   (legacy):     ..._keys_llm_disagreements[_S_E].csv
    base_for_out = base
    base_for_out = re.sub(r"_with_misinfo_keys(_\d+_\d+)?$", "", base_for_out)
    base_for_out = re.sub(r"_keys_llm_disagreements(_\d+_\d+)?$", "", base_for_out)
    base_for_out = re.sub(r"_with_misinfo_gemma(_\d+_\d+)?(_rerun)?$", "", base_for_out)

    out_suffix = "_with_misinfo_gemma" + (f"_{_batch_tag}" if _batch_tag else "")
    if rerun:
        out_suffix = out_suffix + "_rerun"

    out_path = base_for_out + out_suffix + ".csv"

    processed = 0
    chunk_id = 0


    for idx, row in df_iter.iterrows():
        full_text = row.get("full_text", "")
        note_text = row.get("noteText", "")
        media_str = row.get("media", "")

        # parse comma-separated media paths
        if isinstance(media_str, str) and media_str.strip():
            raw_paths = parse_media(row.get("media", ""))
            paths_all = [str(BASE_MEDIA_DIR / p) for p in raw_paths]

            # Gemma multimodal input supports images, not videos.
            # Skip non-image media (e.g., .mp4) so we don't crash in load_image_bytes().
            image_exts = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
            paths = []
            for p in paths_all:
                ext = Path(p).suffix.lower()
                if ext in image_exts:
                    paths.append(p)
        else:
            paths = []

        if rerun:
            system_prompt, user_content = gemma_prompt_second_pass(
                full_text=full_text,
                note_text=note_text,
                image_paths=paths,
                keyword_label=row.get("misinfo_type_keys", ""),
                llm_label=row.get("misinfo_type_llm", ""),
            )
        else:   
            system_prompt, user_content = gemma_prompt(
                full_text=full_text,
                note_text=note_text,
                image_paths=paths,
            )

        print(f"[{idx}] Gemma inference on {len(paths)} media...")

        try:
            response_text = mm_inference_google(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_content["user_prompt"],
                image_paths=user_content["image_paths"],
                max_tokens=max_tokens,
                temperature=temp,
            )
        except Exception as e:
            print(f"❌ Error at index {idx}: {e}")
            response_text = f"ERROR: {e}"

        # Store raw
        if rerun:
            df_iter.at[idx, 'llm_rerun_response'] = response_text
        else:
            df_iter.at[idx, 'llm_response'] = response_text
        
        misinfo_label = None
        conf = None
        rationale = None

        # Parse JSON
        data = parse_llm_json(response_text)
        if data is not None:
            misinfo_label = data.get("misinfo_label")
            conf = data.get("confidence")
            rationale = data.get("rationale")
            print(f"Label: {misinfo_label}")
        else:
            print(f"JSON parse failed or no JSON at index {idx}")

        if rerun:
            df_iter.at[idx, 'misinfo_type_llm_rerun'] = misinfo_label
            df_iter.at[idx, 'confidence_rerun'] = conf
            df_iter.at[idx, 'rationale_rerun'] = rationale
        else:
            df_iter.at[idx, 'misinfo_type_llm'] = misinfo_label
            df_iter.at[idx, 'confidence'] = conf
            df_iter.at[idx, 'rationale'] = rationale

        processed += 1

        # RATE LIMIT PROTECTION
        time.sleep(2.5)   # 2.5 is safe for 30 req/min

        # CHECKPOINT (save chunk files for long runs)
        if processed % save_every == 0:
            chunk_file = f"{base}_{_batch_tag}_chunk{chunk_id}.csv"
            print(f"Saving checkpoint: {chunk_file}")
            df_iter.to_csv(chunk_file, index=False)
            chunk_id += 1

    if rerun:
        # Merge the rerun columns from the processed disagreement rows back into the
        # full slice (df_full_slice).  The integer index is preserved by the iloc/copy
        # above, so loc-based assignment is exact even when start/end batching is used.
        rerun_cols = ['misinfo_type_llm_rerun', 'confidence_rerun', 'rationale_rerun', 'llm_rerun_response']
        df_full_slice.loc[df_iter.index, rerun_cols] = df_iter[rerun_cols].values
        df_full_slice.to_csv(out_path, index=False)
    else:
        df_iter.to_csv(out_path, index=False)
    print(f"Saved Gemma output → {out_path}")

    return out_path








if __name__ == "__main__":

    START = None
    END =  None 


    MODEL = 'gemma-3-27b-it' 
    TEMPERATURE = 0.1
    MAX_TOKENS = 512


    RERUN = False

    if RERUN:
        # output of find_disagreements.py
        csv_path = "path/to/disagreement/file"
    else:
        csv_path = f"path/to/normalized/file"


    extract_misinfo_batch(
        csv_path=csv_path,
        model=MODEL,
        temp=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        start=START,
        end=END,
        save_every=200,
        rerun=RERUN
    )




























































