# YouTube clip worker

Only process videos you own or have permission to repurpose.

## Install

1. Install `yt-dlp` and put it on `PATH`.
2. Default local mode: install `python -m pip install -r requirements-local.txt`, then pull `ollama pull qwen2.5:3b`. Copy `.env.example` to `.env` only for optional OpenAI mode.
3. Run `node worker.mjs`.
4. Import `n8n-workflow.json` into self-hosted n8n. Set `workerUrl` to reachable worker URL, such as `http://host.docker.internal:8787` when n8n runs in Docker on this Windows machine.

POST body:

```json
{ "url": "https://www.youtube.com/watch?v=...", "maxClips": 8 }
```

Worker returns a job id immediately. Poll `GET /jobs/{id}` until status is `done` or `failed`. Final response has clip paths, ranks, timestamps, captions, and suggested post copy.

Default uses local Whisper plus Ollama. `OPENAI_API_KEY` stays only in worker if optional cloud mode is enabled with `AI_BACKEND=openai`.
