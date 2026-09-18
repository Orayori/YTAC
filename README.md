# YTAC — YouTube TikTok Automatic Clipping

YTAC (YouTube TikTok Automatic Clipping) is a local, AI-powered system that automatically turns long-form YouTube videos into short-form vertical videos for TikTok, YouTube Shorts, and similar platforms.

YTAC analyzes a video's transcript and content, identifies engaging and self-contained moments, selects multiple unique clips, automatically reframes the video for vertical viewing, adds animated word-by-word captions, and renders the finished clips using FFmpeg.

## Current Version

**v3.0**

---

## Features

### AI-Powered Clip Selection

YTAC uses local AI to analyze the content and identify potential high-retention moments.

The selection system considers signals such as:

- Strong hooks
- Curiosity
- Emotional moments
- Story development
- Revelations and payoffs
- Practical value
- Quotable statements
- Strong endings
- Self-contained context

It also attempts to avoid:

- Greetings
- Introductions with little value
- Housekeeping
- Sponsor segments
- Long pauses
- Repeated information
- Weak openings
- Clips that depend heavily on missing context

---

### Multiple Unique Clips

YTAC can generate multiple clips from a single source video.

The system attempts to:

- Find as many worthwhile clips as possible
- Avoid unnecessary filler
- Avoid overlapping clips
- Prefer stronger clips when candidates overlap
- Keep clips within the target short-form duration

Default target:

**45–60 seconds per clip**

---

### Local AI

YTAC is designed to run locally using Ollama.

The default AI model is:

```text
qwen2.5:3b
