"""
YouTube Comment Sentiment: web app.
Visitors paste video links (or a channel), click Analyze, and see charts.

Run locally:  python -m streamlit run app.py
Needs yt_sentiment.py in the same folder, and YOUTUBE_API_KEY set
(environment variable locally, or a "secret" on the hosting platform).
"""
import os

import altair as alt
import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from yt_sentiment import (
    QuotaExceeded, clean_and_dedupe, fetch_comments, get_video_titles,
    list_channel_videos, parse_video_id, resolve_uploads_playlist,
)

# ------------------------------------------------------------------ settings
MODELS = {
    "English (faster)": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "Multilingual": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
}
MAX_VIDEOS = 5          # videos per analysis
MAX_COMMENTS = 300      # comments per video
MAX_RUNS = 10           # analyses per visitor session (protects your API quota)
ORDER = ["positive", "neutral", "negative"]
COLORS = {"positive": "#2e9e5b", "neutral": "#8a8f98", "negative": "#d64545"}
SCALE = alt.Scale(domain=ORDER, range=[COLORS[k] for k in ORDER])

st.set_page_config(page_title="YouTube Sentiment", page_icon="", layout="wide")


# ------------------------------------------------------------------- helpers
def get_key():
    key = os.environ.get("YOUTUBE_API_KEY")
    if not key:
        try:
            key = st.secrets["YOUTUBE_API_KEY"]
        except Exception:
            key = None
    return key


@st.cache_data(ttl=3600, show_spinner=False)
def collect(mode, refs, max_videos, max_comments):
    """Fetch + clean comments. Cached for 1 hour so repeat requests cost no API quota."""
    yt = build("youtube", "v3", developerKey=get_key())
    targets = {}
    if mode == "Video links":
        ids = [parse_video_id(r) for r in refs]
        targets.update(get_video_titles(yt, ids))
        for i in ids:
            targets.setdefault(i, "")
    else:
        playlist, _ = resolve_uploads_playlist(yt, refs[0])
        targets.update(dict(list_channel_videos(yt, playlist, max_videos)))

    rows, quota_hit = [], False
    try:
        for vid, title in targets.items():
            got = fetch_comments(yt, vid, max_comments)
            for r in got:
                r["video_title"] = title
            rows.extend(got)
    except QuotaExceeded:
        quota_hit = True
    return clean_and_dedupe(rows), quota_hit


@st.cache_resource(show_spinner="Loading sentiment model (first time only, may take a few minutes)...")
def load_model(name):
    from transformers import pipeline
    return pipeline("text-classification", model=name)


def classify(rows, model_name):
    pipe = load_model(model_name)
    bar = st.progress(0.0, text="Classifying comments...")
    for i in range(0, len(rows), 32):
        batch = rows[i:i + 32]
        preds = pipe([r["clean_text"] for r in batch], truncation=True, max_length=512)
        for r, p in zip(batch, preds):
            r["sentiment"] = p["label"].lower()
            r["confidence"] = round(float(p["score"]), 4)
        bar.progress(min(1.0, (i + 32) / len(rows)), text="Classifying comments...")
    bar.empty()
    return rows


def make_df(rows):
    df = pd.DataFrame(rows)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    label = df["video_title"].fillna("").replace("", pd.NA).fillna(df["video_id"])
    df["video_label"] = label.str.slice(0, 45)
    return df


# ------------------------------------------------------------------- results
def show_results(df):
    st.sidebar.header("Filter results")
    videos = st.sidebar.multiselect("Videos", sorted(df["video_label"].unique()))
    sent = st.sidebar.multiselect("Sentiment", ORDER, default=ORDER)
    min_conf = st.sidebar.slider("Min confidence", 0.0, 1.0, 0.0, 0.05)
    search = st.sidebar.text_input("Search text")

    f = df[df["sentiment"].isin(sent) & (df["confidence"] >= min_conf)]
    if videos:
        f = f[f["video_label"].isin(videos)]
    if search:
        f = f[f["clean_text"].str.contains(search, case=False, na=False)]
    if f.empty:
        st.info("No comments match the current filters.")
        return

    counts, total = f["sentiment"].value_counts(), len(f)
    cols = st.columns(4)
    cols[0].metric("Comments", f"{total:,}")
    for col, label in zip(cols[1:], ORDER):
        n = int(counts.get(label, 0))
        col.metric(label.capitalize(), f"{100 * n / total:.1f}%", f"{n:,} comments", delta_color="off")

    left, right = st.columns(2)
    with left:
        st.subheader("Overall split")
        pie_df = counts.rename_axis("sentiment").reset_index(name="count")
        st.altair_chart(
            alt.Chart(pie_df).mark_arc(innerRadius=60).encode(
                theta="count:Q",
                color=alt.Color("sentiment:N", scale=SCALE, legend=alt.Legend(title=None)),
                tooltip=["sentiment", "count"]),
            use_container_width=True)
    with right:
        st.subheader("Sentiment by video")
        bv = f.groupby(["video_label", "sentiment"]).size().reset_index(name="count")
        st.altair_chart(
            alt.Chart(bv).mark_bar().encode(
                y=alt.Y("video_label:N", title=None, sort="-x"),
                x=alt.X("count:Q", stack="normalize", title="Share", axis=alt.Axis(format="%")),
                color=alt.Color("sentiment:N", scale=SCALE, legend=None),
                tooltip=["video_label", "sentiment", "count"]),
            use_container_width=True)

    timed = f.dropna(subset=["published_at"]).copy()
    if not timed.empty:
        st.subheader("Sentiment over time")
        span = (timed["published_at"].max() - timed["published_at"].min()).days
        freq = "D" if span <= 90 else "W"
        timed["period"] = timed["published_at"].dt.tz_localize(None).dt.to_period(freq).dt.start_time
        bt = timed.groupby(["period", "sentiment"]).size().reset_index(name="count")
        st.altair_chart(
            alt.Chart(bt).mark_bar().encode(
                x=alt.X("period:T", title="Date"), y=alt.Y("count:Q", title="Comments"),
                color=alt.Color("sentiment:N", scale=SCALE, legend=alt.Legend(title=None)),
                tooltip=["period:T", "sentiment", "count"]),
            use_container_width=True)

    st.subheader("Top comments")
    t1, t2 = st.tabs(["Most positive", "Most negative"])
    for tab, label in ((t1, "positive"), (t2, "negative")):
        with tab:
            top = f[f["sentiment"] == label].sort_values(
                ["confidence", "like_count"], ascending=False).head(10)
            st.dataframe(top[["clean_text", "confidence", "like_count", "video_label"]],
                         hide_index=True, use_container_width=True)

    with st.expander("All comments"):
        st.dataframe(
            f[["published_at", "video_label", "author", "clean_text", "sentiment", "confidence", "like_count"]],
            hide_index=True, use_container_width=True)
        st.download_button("Download CSV", f.to_csv(index=False).encode("utf-8-sig"),
                           "sentiment_results.csv", "text/csv")


# ------------------------------------------------------------------------ UI
st.title("📊 YouTube Comment Sentiment")
st.caption("Paste YouTube links, click Analyze, and see how viewers feel. Public comments only.")

if not get_key():
    st.error("No YouTube API key found. Set the YOUTUBE_API_KEY environment variable or secret.")
    st.stop()

mode = st.radio("Analyze", ["Video links", "Channel"], horizontal=True)
with st.form("analyze"):
    if mode == "Video links":
        text = st.text_area("YouTube video links (one per line)", height=110,
                            placeholder="https://www.youtube.com/watch?v=...")
    else:
        text = st.text_input("Channel (@handle, URL, or ID)", placeholder="@SomeChannel")
    c1, c2, c3 = st.columns(3)
    max_comments = c1.slider("Comments per video", 20, MAX_COMMENTS, 100, 20)
    max_videos = c2.slider("Latest videos (channel mode)", 1, MAX_VIDEOS, 3)
    model_choice = c3.selectbox("Language", list(MODELS))
    go = st.form_submit_button("Analyze", type="primary")

if go:
    refs = tuple(l.strip() for l in text.splitlines() if l.strip())
    st.session_state.setdefault("runs", 0)
    if not refs:
        st.warning("Please enter at least one link.")
    elif mode == "Video links" and len(refs) > MAX_VIDEOS:
        st.warning(f"Please enter at most {MAX_VIDEOS} links.")
    elif st.session_state["runs"] >= MAX_RUNS:
        st.warning("You've reached the limit of analyses for this session. Please come back later.")
    else:
        try:
            with st.spinner("Fetching comments from YouTube..."):
                rows, quota_hit = collect(mode, refs, max_videos, max_comments)
            if quota_hit:
                st.warning("The daily YouTube quota ran out, so results may be partial. Try again tomorrow.")
            if not rows:
                st.info("No comments found. Comments may be disabled on these videos.")
            else:
                rows = classify(rows, MODELS[model_choice])
                st.session_state["df"] = make_df(rows)
                st.session_state["runs"] += 1
        except ValueError as e:
            st.error(str(e))
        except HttpError:
            st.error("YouTube rejected the request. Check the link, or try again later.")

if "df" in st.session_state:
    show_results(st.session_state["df"])