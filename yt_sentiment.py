
#!/usr/bin/env python3
"""
YouTube comment sentiment analysis: fetch -> clean -> classify -> store -> summarize.

Setup:
    pip install -r requirements.txt
    export YOUTUBE_API_KEY="your_key"        # Windows (PowerShell): $env:YOUTUBE_API_KEY="your_key"

Examples:
    python yt_sentiment.py --video https://youtu.be/VIDEO_ID
    python yt_sentiment.py --channel @SomeChannel --max-videos 5 --max-comments 300
    python yt_sentiment.py --video ID1 ID2 --channel @A @B
"""
import argparse
import csv
import html
import os
import re
import sqlite3
import sys
from collections import Counter

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

DEFAULT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
# For multilingual comments use: cardiffnlp/twitter-xlm-roberta-base-sentiment


class QuotaExceeded(Exception):
    pass


# ----------------------------------------------------------------- fetching
def parse_video_id(ref):
    ref = ref.strip()
    patterns = [
        r"[?&]v=([\w-]{11})",
        r"youtu\.be/([\w-]{11})",
        r"youtube\.com/(?:shorts|embed|live)/([\w-]{11})",
        r"^([\w-]{11})$",
    ]
    for p in patterns:
        m = re.search(p, ref)
        if m:
            return m.group(1)
    raise ValueError(f"Could not parse a video ID from: {ref}")


def resolve_uploads_playlist(yt, ref):
    """Accepts a channel URL, @handle, or channel ID. Returns (uploads_playlist_id, channel_title)."""
    ref = ref.strip()
    params = {"part": "contentDetails,snippet"}
    m = re.search(r"youtube\.com/channel/(UC[\w-]{22})", ref) or re.fullmatch(r"(UC[\w-]{22})", ref)
    if m:
        params["id"] = m.group(1)
    else:
        m = re.search(r"youtube\.com/(@[\w.\-]+)", ref) or re.fullmatch(r"(@[\w.\-]+)", ref)
        if not m:
            raise ValueError(f"Use a channel URL, @handle, or channel ID: {ref}")
        params["forHandle"] = m.group(1)
    items = yt.channels().list(**params).execute().get("items", [])
    if not items:
        raise ValueError(f"Channel not found: {ref}")
    ch = items[0]
    return ch["contentDetails"]["relatedPlaylists"]["uploads"], ch["snippet"]["title"]


def list_channel_videos(yt, playlist_id, max_videos):
    videos, token = [], None
    while len(videos) < max_videos:
        res = yt.playlistItems().list(
            part="snippet", playlistId=playlist_id,
            maxResults=min(50, max_videos - len(videos)), pageToken=token,
        ).execute()
        for it in res["items"]:
            sn = it["snippet"]
            videos.append((sn["resourceId"]["videoId"], sn["title"]))
        token = res.get("nextPageToken")
        if not token:
            break
    return videos


def get_video_titles(yt, ids):
    titles = {}
    for i in range(0, len(ids), 50):
        res = yt.videos().list(part="snippet", id=",".join(ids[i:i + 50])).execute()
        for it in res.get("items", []):
            titles[it["id"]] = it["snippet"]["title"]
    return titles


def fetch_comments(yt, video_id, limit):
    out, token = [], None
    while len(out) < limit:
        try:
            res = yt.commentThreads().list(
                part="snippet", videoId=video_id, maxResults=min(100, limit - len(out)),
                pageToken=token, textFormat="plainText", order="relevance",
            ).execute()
        except HttpError as e:
            msg = str(e)
            if "quotaExceeded" in msg:
                raise QuotaExceeded() from e
            if "commentsDisabled" in msg or e.resp.status in (403, 404):
                print(f"  ! Comments unavailable for {video_id}, skipping.")
                break
            raise
        for it in res["items"]:
            sn = it["snippet"]["topLevelComment"]["snippet"]
            out.append({
                "comment_id": it["snippet"]["topLevelComment"]["id"],
                "video_id": video_id,
                "author": sn.get("authorDisplayName", ""),
                "published_at": sn.get("publishedAt", ""),
                "like_count": sn.get("likeCount", 0),
                "text": sn.get("textDisplay", ""),
            })
        token = res.get("nextPageToken")
        if not token:
            break
    return out


# ----------------------------------------------------------------- cleaning
URL_RE = re.compile(r"https?://\S+|www\.\S+")
TAG_RE = re.compile(r"<[^>]+>")


def clean_text(text):
    text = html.unescape(text)
    text = TAG_RE.sub(" ", text)
    text = URL_RE.sub("http", text)
    text = re.sub(r"@\w+", "@user", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_and_dedupe(rows):
    seen, out = set(), []
    for r in rows:
        r["clean_text"] = clean_text(r["text"])
        key = r["clean_text"].lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


# ----------------------------------------------------------- classification
def classify(rows, model_name, batch_size=32):
    from transformers import pipeline  # imported late so --help is fast

    print(f"Loading model {model_name} (first run downloads ~500 MB)...")
    pipe = pipeline("text-classification", model=model_name)
    texts = [r["clean_text"] for r in rows]
    for i in range(0, len(texts), batch_size):
        preds = pipe(texts[i:i + batch_size], truncation=True, max_length=512)
        for r, p in zip(rows[i:i + batch_size], preds):
            r["sentiment"] = p["label"].lower()
            r["confidence"] = round(float(p["score"]), 4)
        print(f"  classified {min(i + batch_size, len(texts))}/{len(texts)}", end="\r")
    print()
    return rows


# ------------------------------------------------------------------ storage
FIELDS = ["comment_id", "video_id", "video_title", "author", "published_at",
          "like_count", "text", "clean_text", "sentiment", "confidence"]


def save_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Saved CSV: {path}")


def save_sqlite(rows, path):
    con = sqlite3.connect(path)
    con.execute(
        """CREATE TABLE IF NOT EXISTS comments (
            comment_id TEXT PRIMARY KEY, video_id TEXT, video_title TEXT, author TEXT,
            published_at TEXT, like_count INTEGER, text TEXT, clean_text TEXT,
            sentiment TEXT, confidence REAL)"""
    )
    con.executemany(
        f"INSERT OR REPLACE INTO comments ({','.join(FIELDS)}) VALUES ({','.join('?' * len(FIELDS))})",
        [tuple(r.get(k) for k in FIELDS) for r in rows],
    )
    con.commit()
    con.close()
    print(f"Saved SQLite: {path}")


# ----------------------------------------------------------------- summary
def summarize(rows):
    total = len(rows)
    counts = Counter(r["sentiment"] for r in rows)
    print("\n=== Sentiment summary ===")
    print(f"Total comments analyzed: {total}")
    for label in ("positive", "neutral", "negative"):
        n = counts.get(label, 0)
        print(f"  {label:<9} {n:>6}  ({100 * n / total:.1f}%)")
    for label in ("positive", "negative"):
        top = sorted((r for r in rows if r["sentiment"] == label),
                     key=lambda r: r["confidence"], reverse=True)[:3]
        if top:
            print(f"\nTop {label} comments:")
            for r in top:
                print(f"  [{r['confidence']:.2f}] {r['clean_text'][:140]}")


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="YouTube comment sentiment analysis")
    ap.add_argument("--video", nargs="*", default=[], help="video URLs or IDs")
    ap.add_argument("--channel", nargs="*", default=[], help="channel URLs, @handles, or IDs")
    ap.add_argument("--max-videos", type=int, default=5, help="latest videos per channel")
    ap.add_argument("--max-comments", type=int, default=200, help="comments per video")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--csv", default="results.csv")
    ap.add_argument("--db", default="results.db")
    args = ap.parse_args()

    if not args.video and not args.channel:
        ap.error("Provide at least one --video or --channel")
    key = os.environ.get("YOUTUBE_API_KEY")
    if not key:
        sys.exit("Set the YOUTUBE_API_KEY environment variable first.")

    yt = build("youtube", "v3", developerKey=key)

    # 1. Collect the list of videos to analyze
    targets = {}  # video_id -> title
    try:
        ids = [parse_video_id(v) for v in args.video]
        if ids:
            targets.update(get_video_titles(yt, ids))
            for i in ids:
                targets.setdefault(i, "")
        for ch in args.channel:
            playlist, name = resolve_uploads_playlist(yt, ch)
            vids = list_channel_videos(yt, playlist, args.max_videos)
            print(f"Channel '{name}': {len(vids)} videos")
            targets.update(dict(vids))
    except (ValueError, HttpError) as e:
        sys.exit(f"Error: {e}")

    # 2. Fetch comments
    rows = []
    try:
        for n, (vid, title) in enumerate(targets.items(), 1):
            print(f"[{n}/{len(targets)}] Fetching comments: {title or vid}")
            got = fetch_comments(yt, vid, args.max_comments)
            for r in got:
                r["video_title"] = title
            rows.extend(got)
    except QuotaExceeded:
        print("\n! Daily YouTube API quota exceeded. Continuing with what was collected.")

    # 3. Clean, classify, store, summarize
    rows = clean_and_dedupe(rows)
    if not rows:
        sys.exit("No comments to analyze.")
    rows = classify(rows, args.model)
    save_csv(rows, args.csv)
    save_sqlite(rows, args.db)
    summarize(rows)


if __name__ == "__main__":
    main()