import os
import csv
import asyncio
from urllib.parse import urlparse
import pandas as pd
import httpx, httpcore


# NOTE: This script includes a monkey patch for `twikit` to keep guest/login flows working.

"""
  ! MONKEY PATCH ALERT ! 24/3/2026
"""

# MONKEY PATCH: Remove this block when twikit is updated to fix ON_DEMAND_FILE_REGEX
import re
_tx_mod = __import__('twikit.x_client_transaction.transaction', fromlist=['ClientTransaction'])
_tx_mod.ON_DEMAND_FILE_REGEX = re.compile(
    r""",(\d+):["']ondemand\.s["']""", flags=(re.VERBOSE | re.MULTILINE))
_tx_mod.ON_DEMAND_HASH_PATTERN = r',{}:"([0-9a-f]+)"'

async def _patched_get_indices(self, home_page_response, session, headers):
    key_byte_indices = []
    response = self.validate_response(home_page_response) or self.home_page_response
    on_demand_file_index = _tx_mod.ON_DEMAND_FILE_REGEX.search(str(response)).group(1)
    regex = re.compile(_tx_mod.ON_DEMAND_HASH_PATTERN.format(on_demand_file_index))
    filename = regex.search(str(response)).group(1)
    on_demand_file_url = f"https://abs.twimg.com/responsive-web/client-web/ondemand.s.{filename}a.js"
    on_demand_file_response = await session.request(method="GET", url=on_demand_file_url, headers=headers)
    key_byte_indices_match = _tx_mod.INDICES_REGEX.finditer(str(on_demand_file_response.text))
    for item in key_byte_indices_match:
        key_byte_indices.append(item.group(2))
    if not key_byte_indices:
        raise Exception("Couldn't get KEY_BYTE indices")
    key_byte_indices = list(map(int, key_byte_indices))
    return key_byte_indices[0], key_byte_indices[1:]

_tx_mod.ClientTransaction.get_indices = _patched_get_indices
# END MONKEY PATCH

""" """


from twikit import Client
from twikit.guest import GuestClient

# Common transient errors httpx/httpcore
RETRYABLE_ACTIVATE_ERRORS = (
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
    httpx.ConnectError,
    httpcore.ConnectTimeout,
)

# Constants
BATCH_SIZE = 145
DELAY_BETWEEN_TWEETS = 3 


# Helper to avoid breaking when user cant be reactivated
async def activate_guest_simple(client, retries=10, sleep_seconds=60):
    """
    Re-activate GuestClient with simple retries.
    - On failure: sleep `sleep_seconds` and retry.
    Returns True on success, False if all attempts fail.
    """
    for attempt in range(1, retries + 1):
        try:
            await client.activate()
            print("Guest client activated")
            return True
        except RETRYABLE_ACTIVATE_ERRORS as e:
            print(f"activate() timeout/network issue ({type(e).__name__}) — attempt {attempt}/{retries}. Sleeping {sleep_seconds}s…")
            await asyncio.sleep(sleep_seconds)
        except Exception as e:
            print(f"activate() failed (non-retryable): {type(e).__name__}: {e}")
            return False
    print("activate() failed after all retries")
    return False

def _normalize_created_at(value):
    """String for CSV / merge_notes_with_tweet_data."""
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)


# --------------------------------------------------------
# Helpers for video downloading (unused while video branch in get_tweets_by_ids is commented out)
# --------------------------------------------------------


def _stream_key(s):
    """Sort key: (bitrate, height, width) with safe fallbacks."""
    br = getattr(s, "bitrate", None)
    h = getattr(s, "height", None)
    w = getattr(s, "width", None)
    # Treat missing as very small so they sort to the front
    return (
        br if isinstance(br, int) else -1,
        h if isinstance(h, int) else -1,
        w if isinstance(w, int) else -1,
    )


def _stream_content_type(s):
    return (getattr(s, "content_type", "") or "").lower()


def pick_stream_second_lowest_mp4_or_any(streams):
    """Pick second-lowest-bitrate MP4 if >=2 streams; else best MP4; else second-lowest / any stream."""
    if not streams:
        return None, None

    mp4s = [s for s in streams if "mp4" in _stream_content_type(s)]
    ordered = sorted(mp4s, key=_stream_key)
    if len(ordered) >= 2:
        return ordered[1], "mp4"
    if len(ordered) == 1:
        return ordered[0], "mp4"

    # no mp4: take second-lowest of ANY
    any_ordered = sorted(streams, key=_stream_key)
    if not any_ordered:
        return None, None
    chosen = any_ordered[1] if len(any_ordered) >= 2 else any_ordered[0]

    ct = _stream_content_type(chosen)
    if "mpegurl" in ct or "m3u8" in ct:
        ext = "m3u8"
    elif "gif" in ct:
        ext = "mp4"  # Twitter "animated_gif" is usually MP4-encoded
    elif "mp2t" in ct or "ts" in ct:
        ext = "ts"
    else:
        ext = "mp4"  # sane default
    return chosen, ext


def _append_media_rel_path(media_files, abs_filename, base_for_media):
    rel = os.path.relpath(
        os.path.normpath(abs_filename),
        os.path.normpath(base_for_media),
    )
    media_files.append(rel.replace("\\", "/"))


async def _download_with_retry_if_empty(path, download_callable):
    """Run download; if file is missing or 0 bytes, retry once."""
    await download_callable()
    try:
        if (not os.path.isfile(path)) or os.path.getsize(path) == 0:
            await download_callable()
    except OSError:
        pass


def _infer_photo_extension(media_obj):
    """
    Infer image extension from media metadata/URL.
    Falls back to jpg if unknown.
    """
    # Try explicit extension-like attrs first
    for attr in ("ext", "extension"):
        value = getattr(media_obj, attr, None)
        if isinstance(value, str) and value.strip():
            ext = value.strip().lower().lstrip(".")
            if ext in {"jpg", "jpeg", "png", "webp", "gif"}:
                return ext

    # Try content type if present
    content_type = (getattr(media_obj, "content_type", "") or "").lower()
    if "image/jpeg" in content_type:
        return "jpg"
    if "image/png" in content_type:
        return "png"
    if "image/webp" in content_type:
        return "webp"
    if "image/gif" in content_type:
        return "gif"

    # Try known URL attrs
    for attr in ("media_url_https", "media_url", "url"):
        media_url = getattr(media_obj, attr, None)
        if not media_url or not isinstance(media_url, str):
            continue

        path = urlparse(media_url).path.lower()
        if "." in path:
            candidate = path.rsplit(".", 1)[-1]
            if candidate in {"jpg", "jpeg", "png", "webp", "gif"}:
                return candidate

    return "jpg"

async def get_tweets_by_ids(
    input_path,
    output_path,
    image_dir,
    mode,
    start_id=None,
    end_id=None,
    *,
    cookies_path=None,
):
    """
    Fetch tweets by id (notes CSV must have column tweetId).

    Writes columns tweet_id, media, full_text, created_at_datetime (plus extra columns).

    media: comma-separated paths relative to image_dir
    """
    # Load tweet IDs from CSV
    df = pd.read_csv(input_path)
    tweet_ids = df["tweetId"].dropna().astype(str).tolist()

    # Create image dir
    image_dir = os.path.abspath(image_dir)
    os.makedirs(image_dir, exist_ok=True)

    base_for_media = image_dir

    # Fresh main CSV each run (script used to append onto stale files)
    if os.path.exists(output_path):
        os.remove(output_path)

    
    # Run for a subset
    # Normalize None → full range
    if start_id is None and end_id is None:
        start_id = 0
        end_id = len(tweet_ids)

    # Now slice 
    tweet_ids = tweet_ids[start_id:end_id]


    """
        Connection to x.com with Twikit
        
        A) Using credentials
         - The first time, log in using the login method and save cookies   
         - After the second time, load the saved cookies (to avoid bans)

        B) Connect as a Guest- Easier, no account bans

    """

    if mode == 'login':
        client = Client()
        cookie_file = cookies_path or "cookies.json"

        if os.path.isfile(cookie_file):
            """Cookie file exists: load it (normal runs)."""
            client.load_cookies(cookie_file)
        else:
            """No cookie file yet: log in once, then save cookies for later runs."""
            email = os.getenv("X_EMAIL")
            username = os.getenv("X_USERNAME")
            password = os.getenv("X_PASSWORD")
            missing = [k for k, v in [("X_EMAIL", email), ("X_USERNAME", username), ("X_PASSWORD", password)] if not v]
            if missing:
                raise RuntimeError(
                    "Missing login credentials in environment: "
                    + ", ".join(missing)
                    + ". Put them in your .env (not committed) or export them before running."
                )
            await client.login(
                auth_info_1=email,
                auth_info_2=username,
                password=password,
            )
            client.save_cookies(cookie_file)

    elif mode == 'guest': 
        client = GuestClient()
        await client.activate() 
    else:
        print("Invalid mode. Must be 'login' or 'guest'")
        return

    results = []
    wrote_main_header = False
    
    for i in range(0, len(tweet_ids), BATCH_SIZE):
       batch = tweet_ids[i:i + BATCH_SIZE]
       print(f"\nProcessing batch {i//BATCH_SIZE + 1} ({len(batch)} tweets)...")

       results = [] 

       for idx, tweet_id in enumerate(batch):
            try:
                tweet = await client.get_tweet_by_id(tweet_id)
                
                if tweet is None:
                    raise ValueError(f"\n {i + idx + start_id + 1}) Tweet {tweet_id} is None (possibly deleted). Skipping.")
    
                media_files = []
                if tweet.media:
                    for media_idx, m in enumerate(tweet.media):
                        if m.type == "photo":
                            image_ext = _infer_photo_extension(m)
                            filename = os.path.join(
                                image_dir, f"photo_{tweet_id}_{media_idx}.{image_ext}"
                            )
                            await _download_with_retry_if_empty(
                                filename, lambda: m.download(filename)
                            )
                            _append_media_rel_path(media_files, filename, base_for_media)

                        # --- Video / animated_gif download (disabled: uncomment block to enable) ---
                        elif m.type in ("video", "animated_gif"):
                            streams = getattr(m, "streams", None) or []
                            if not streams:
                                print(
                                    f"⚠️ No streams for tweet {tweet_id} media_idx={media_idx} ({m.type})"
                                )
                            else:
                                stream, ext = pick_stream_second_lowest_mp4_or_any(streams)
                                if not stream or not ext:
                                    print(
                                        f"⚠️ No suitable stream for tweet {tweet_id} media_idx={media_idx}"
                                    )
                                else:
                                    filename = os.path.join(
                                        image_dir,
                                        f"video_{tweet_id}_{media_idx}.{ext}",
                                    )
                                    await _download_with_retry_if_empty(
                                        filename, lambda: stream.download(filename)
                                    )
                                    _append_media_rel_path(
                                        media_files, filename, base_for_media
                                    )

                result = {
                    "iteration_id": i + idx + start_id + 1, 
                    "tweet_id": tweet_id,
                    "created_at_datetime": _normalize_created_at(
                        tweet.created_at_datetime if mode == 'login' else tweet.created_at
                    ),
                    "user": tweet.user.name if tweet.user else None,
                    "user_followers": tweet.user.followers_count if tweet.user else None,
                    "text": tweet.text,
                    "lang": tweet.lang,
                    "in_reply_to_id": tweet.in_reply_to,
                    "in_reply_to_url": ("https://x.com/i/web/status/" + tweet.in_reply_to) if tweet.in_reply_to else None,
                    "quoted_post_id": tweet.quote.id if tweet.quote else None,
                    "quoted_post_text": tweet.quote.full_text if tweet.quote else None,
                    "retweeted_tweet": tweet.retweeted_tweet.full_text if tweet.retweeted_tweet else None,
                    "media": ", ".join(media_files) if media_files else None,
                    "reply_count": tweet.reply_count,
                    "favorite_count": tweet.favorite_count,
                    "view_count": tweet.view_count,
                    "view_count_state": tweet.view_count_state,
                    "retweet_count": tweet.retweet_count,
                    "bookmark_count": tweet.bookmark_count,
                    "place": tweet.place.full_name if hasattr(tweet, "place") and tweet.place else None,
                    "replies": len(tweet.replies) if hasattr(tweet, "replies") and tweet.replies else None,
                    "hashtags": ", ".join(tweet.hashtags) if tweet.hashtags else None,
                    "thumbnail_title": tweet.thumbnail_title,
                    "thumbnail_url": tweet.thumbnail_url,
                    "urls": ", ".join([u['expanded_url'] for u in tweet.urls]) if tweet.urls else None,
                    "full_text": tweet.full_text,
                    'tweetUrl':f"https://x.com/i/web/status/{tweet_id}",
                }

                results.append(result)

                print(f"{i + idx + start_id + 1}) Tweet {tweet_id} processed.")
                await asyncio.sleep(DELAY_BETWEEN_TWEETS)


            except Exception as e:
                print(f"{i + idx + start_id + 1}) Error processing tweet {tweet_id}: {e}")

                results.append({
                    "iteration_id": i + idx + start_id + 1,
                    "tweet_id": tweet_id,
                    "created_at_datetime": None,
                    "user": None,
                    "user_followers": None,
                    "text": None,
                    "lang": None,
                    "in_reply_to_id": None,
                    "in_reply_to_url": None,
                    "quoted_post_id": None,
                    "quoted_post_text": None,
                    "retweeted_tweet": None,
                    "media": None,
                    "reply_count": None,
                    "favorite_count": None,
                    "view_count": None,
                    "view_count_state": None,
                    "retweet_count": None,
                    "bookmark_count": None,
                    "place": None,
                    "replies": None,
                    "hashtags": None,
                    "thumbnail_title": None,
                    "thumbnail_url": None,
                    "urls": None,
                    "full_text": None,
                    'tweetUrl':f"https://x.com/i/web/status/{tweet_id}",
                })

                await asyncio.sleep(DELAY_BETWEEN_TWEETS)


            if mode == 'guest':
                if (i + idx + start_id + 1) % 40 == 0:
                    print("Re-activating guest client to refresh token...")
                    ok = await activate_guest_simple(client, retries=10, sleep_seconds=60)
                    if not ok:
                        print("Could not re-activate after retries — continuing (you may see failures until next attempt).")


       # Save the current batch to a csv
       if not results:
           print("No rows in this batch; skipping write.")
           continue

       first_iter_id = i + start_id + 1
       last_iter_id = i + len(batch) + start_id
       batch_output_path = output_path.replace(".csv", f"_{first_iter_id}-{last_iter_id}.csv")

       print(f"Saving batch to {batch_output_path}...")

       keys = results[0].keys()
       with open(batch_output_path, mode='w', newline='', encoding='utf-8') as f:
           writer = csv.DictWriter(f, fieldnames=keys)
           writer.writeheader()
           writer.writerows(results)

       # Also write/append into the main output_path that downstream expects
       if results:
           with open(output_path, mode='a', newline='', encoding='utf-8') as f:
               writer = csv.DictWriter(f, fieldnames=keys)
               if not wrote_main_header:
                   writer.writeheader()
                   wrote_main_header = True
               writer.writerows(results)


       # Wait between batches if more tweets remain
       if i + BATCH_SIZE < len(tweet_ids):
           print(f"\n Waiting 5 minutes to respect rate limits...")
           for t in range(5, 0, -1):
               print(f"  ... {t} minutes left")
               await asyncio.sleep(61)


def run_tweet_extraction_sync(
    input_path_image,
    input_path_video,
    image_dir,
    mode,
    start_id=None,
    end_id=None,
    *,
    cookies_path=None,
):
    """
    Sync wrapper: fetch tweets for image and video note subsets.

    Media is saved under ``image_dir/image_set`` and ``image_dir/video_set``.
    Tweet CSVs are always written next to the notes CSVs by replacing ``.csv`` with ``_tweet_data.csv``.
    """
    if start_id is None and end_id is None:
        out_img = input_path_image.replace(".csv", "_tweet_data.csv")
        out_vid = input_path_video.replace(".csv", "_tweet_data.csv")
    else:
        out_img = input_path_image.replace(".csv", f"_{start_id}-{end_id}_tweet_data.csv")
        out_vid = input_path_video.replace(".csv", f"_{start_id}-{end_id}_tweet_data.csv")

    media_root = os.path.abspath(image_dir)
    dir_image_set = os.path.join(media_root, "image_set")
    dir_video_set = os.path.join(media_root, "video_set")

    asyncio.run(
        get_tweets_by_ids(
            input_path_image,
            out_img,
            dir_image_set,
            mode,
            start_id,
            end_id,
            cookies_path=cookies_path,
        )
    )
    asyncio.run(
        get_tweets_by_ids(
            input_path_video,
            out_vid,
            dir_video_set,
            mode,
            start_id,
            end_id,
            cookies_path=cookies_path,
        )
    )

    print(f"Tweet data for image set is saved in {out_img}")
    print(f"Tweet data for video set is saved in {out_vid}")

    return {"image": out_img, "video": out_vid}



if __name__ == "__main__":
    # Example run
    
    """
    Tweet Extraction Pipeline
        - This pipeline is used to extract tweets and download media
        - You have to install and set up Twikit library first
        - Use either guest mode or login mode
            - For login mode, you need to use credentials, then save a cookies file --> setup on tweet_extraction.py
        - This run will produce per modality:
            - One combined CSV (per modality)
            - One shard CSV per batch
                --> so a long run produces multiple shards plus the single combined file.
    """

    START_ID = 0
    END_ID = 100
    MODE = "guest"
    INPUT_IMAGE = "notes-data-image-set.csv"
    INPUT_VIDEO = "notes-data-video-set.csv"
    image_dir = "tweet_media"

    paths = run_tweet_extraction_sync(
        input_path_image=INPUT_IMAGE,
        input_path_video=INPUT_VIDEO,
        image_dir=image_dir,
        mode=MODE,
        start_id=START_ID,
        end_id=END_ID,
    )








