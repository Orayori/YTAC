import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { existsSync, readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { randomUUID } from 'node:crypto';

loadEnv();

const port = Number(process.env.PORT || 8787);
const outputRoot = resolve(process.env.OUTPUT_DIR || 'output');
const aiBackend = (process.env.AI_BACKEND || 'local').toLowerCase();
const model = process.env.OLLAMA_MODEL || 'qwen2.5:3b';

const jobs = new Map();

const MIN_CLIP_SECONDS = 45;
const MAX_CLIP_SECONDS = 60;
const MAX_REQUESTED_CLIPS = 30;
const MAX_CANDIDATES = Number(process.env.MAX_CANDIDATES || 80);
const MIN_OVERLAP_RATIO = Number(process.env.MIN_OVERLAP || 0.35);

function loadEnv() {
  if (!existsSync('.env')) return;

  for (const line of readFileSync('.env', 'utf8').split(/\r?\n/)) {
    const match = line.match(/^([A-Z0-9_]+)=(.*)$/);

    if (match && !process.env[match[1]]) {
      process.env[match[1]] = match[2]
        .trim()
        .replace(/^(['"])(.*)\1$/, '$2');
    }
  }
}

function run(command, args, cwd) {
  return new Promise((resolveRun, reject) => {
    const child = spawn(command, args, {
      cwd,
      windowsHide: true
    });

    let output = '';

    child.stdout.on('data', data => {
      output += data;
    });

    child.stderr.on('data', data => {
      output += data;
    });

    child.on('error', reject);

    child.on('close', code => {
      if (code) {
        reject(
          new Error(
            `${command} exited ${code}: ${output.slice(-2500)}`
          )
        );
      } else {
        resolveRun(output);
      }
    });
  });
}

function reply(res, code, value) {
  res.writeHead(code, {
    'content-type': 'application/json; charset=utf-8',
    'access-control-allow-origin': '*'
  });

  res.end(JSON.stringify(value));
}

async function body(req) {
  let text = '';

  for await (const chunk of req) {
    text += chunk;
  }

  return text ? JSON.parse(text) : {};
}

async function getDuration(file, cwd) {
  const result = await run(
    'ffprobe',
    [
      '-v',
      'error',
      '-show_entries',
      'format=duration',
      '-of',
      'default=noprint_wrappers=1:nokey=1',
      file
    ],
    cwd
  );

  const duration = Number.parseFloat(result.trim());

  if (!Number.isFinite(duration) || duration <= 0) {
    throw new Error('Could not determine video duration.');
  }

  return duration;
}

function cleanWords(words) {
  return (Array.isArray(words) ? words : [])
    .map(word => ({
      word: String(word.word ?? '').trim(),
      start: Number(word.start),
      end: Number(word.end)
    }))
    .filter(
      word =>
        word.word &&
        Number.isFinite(word.start) &&
        Number.isFinite(word.end) &&
        word.end > word.start
    )
    .sort((a, b) => a.start - b.start);
}

function overlapSeconds(a, b) {
  return Math.max(
    0,
    Math.min(a.end, b.end) - Math.max(a.start, b.start)
  );
}

function overlapRatio(a, b) {
  return (
    overlapSeconds(a, b) /
    Math.min(a.end - a.start, b.end - b.start)
  );
}

function normalizeClip(raw, sourceDuration) {
  let start = Number(raw.start);
  let end = Number(raw.end);

  if (!Number.isFinite(start) || !Number.isFinite(end)) {
    return null;
  }

  start = Math.max(0, Math.min(start, sourceDuration));
  end = Math.max(0, Math.min(end, sourceDuration));

  if (end < start) {
    [start, end] = [end, start];
  }

  if (end - start < MIN_CLIP_SECONDS) {
    const center = (start + end) / 2;

    start = Math.max(
      0,
      center - MIN_CLIP_SECONDS / 2
    );

    end = Math.min(
      sourceDuration,
      start + MIN_CLIP_SECONDS
    );

    start = Math.max(
      0,
      end - MIN_CLIP_SECONDS
    );
  }

  if (end - start > MAX_CLIP_SECONDS) {
    end = start + MAX_CLIP_SECONDS;
  }

  if (end - start < MIN_CLIP_SECONDS - 0.01) {
    return null;
  }

  return {
    start: Number(start.toFixed(2)),
    end: Number(end.toFixed(2)),
    score: Number(raw.score) || 0,
    hook: String(raw.hook || '').slice(0, 240),
    caption: String(
      raw.caption || raw.hook || ''
    ).slice(0, 280),
    reason: String(raw.reason || '').slice(0, 500)
  };
}

function generateCandidates(words, sourceDuration) {
  const candidates = [];

  const maximumStart = Math.max(
    0,
    sourceDuration - MIN_CLIP_SECONDS
  );

  for (
    let start = 0;
    start <= maximumStart &&
    candidates.length < MAX_CANDIDATES;
    start += 15
  ) {
    const end = Math.min(
      sourceDuration,
      start + MAX_CLIP_SECONDS
    );

    const inside = words.filter(
      word =>
        word.end > start &&
        word.start < end
    );

    if (inside.length < 20) {
      continue;
    }

    const speechSeconds = inside.reduce(
      (sum, word) =>
        sum +
        Math.max(
          0,
          Math.min(word.end, end) -
          Math.max(word.start, start)
        ),
      0
    );

    const speechDensity =
      speechSeconds / (end - start);

    const wordsPerSecond =
      inside.length / (end - start);

    const baseline =
      speechDensity * 40 +
      Math.min(wordsPerSecond, 4) * 8;

    candidates.push({
      id: `C${candidates.length + 1}`,
      start,
      end,
      excerpt: inside
        .slice(0, 180)
        .map(word => word.word)
        .join(' '),
      baseline: Number(baseline.toFixed(2))
    });
  }

  return candidates;
}

async function askAI(prompt) {
  if (aiBackend === 'openai') {
    const response = await fetch(
      'https://api.openai.com/v1/chat/completions',
      {
        method: 'POST',
        headers: {
          authorization:
            `Bearer ${process.env.OPENAI_API_KEY}`,
          'content-type': 'application/json'
        },
        body: JSON.stringify({
          model: 'gpt-4.1-mini',
          response_format: {
            type: 'json_object'
          },
          temperature: 0.15,
          messages: [
            {
              role: 'user',
              content: prompt
            }
          ]
        })
      }
    );

    if (!response.ok) {
      throw new Error(
        `OpenAI: ${await response.text()}`
      );
    }

    const data = await response.json();

    return JSON.parse(
      data.choices[0].message.content
    );
  }

  const response = await fetch(
    'http://127.0.0.1:11434/api/chat',
    {
      method: 'POST',
      headers: {
        'content-type': 'application/json'
      },
      body: JSON.stringify({
        model,
        stream: false,
        format: 'json',
        options: {
          temperature: 0.15
        },
        messages: [
          {
            role: 'user',
            content: prompt
          }
        ]
      })
    }
  );

  if (!response.ok) {
    throw new Error(
      `Ollama: ${await response.text()}`
    );
  }

  const data = await response.json();

  return JSON.parse(
    data.message.content
  );
}

async function chooseClips(
  words,
  sourceDuration,
  requested
) {
  const candidates = generateCandidates(
    words,
    sourceDuration
  );

  if (!candidates.length) {
    throw new Error(
      'Not enough spoken content for 45-60 second clips.'
    );
  }

  const prompt = `
You are selecting short-form clips from a YouTube video.

Choose the strongest UNIQUE and SELF-CONTAINED moments.

Requested maximum clips: ${requested}

Every candidate is already 45-60 seconds.

A strong clip should:

- immediately give the viewer a reason to keep watching
- contain a useful, surprising, funny, emotional, controversial, interesting, or story-driven moment
- make sense without requiring the previous section
- contain a clear payoff or conclusion
- avoid long pauses
- avoid greetings and housekeeping
- avoid sponsor reads and advertisements
- avoid repetitive information
- avoid clips that depend on missing context
- avoid selecting overlapping candidates
- avoid selecting weak clips simply to reach the requested number

Select ONLY from the candidate IDs below.

Return JSON only:

{
  "clips": [
    {
      "id": "C1",
      "score": 95,
      "hook": "Short hook",
      "caption": "Suggested social caption",
      "reason": "Why this clip is worth using"
    }
  ]
}

Score each selected clip from 0-100.

Do not select a clip just because there are empty slots.

CANDIDATES:

${candidates
  .map(
    candidate =>
      `${candidate.id} ` +
      `[${candidate.start.toFixed(1)}-${candidate.end.toFixed(1)}] ` +
      candidate.excerpt
  )
  .join('\n')}
`;

  let data;

  try {
    data = await askAI(prompt);
  } catch (error) {
    console.warn(
      `AI selection failed; using fallback: ${error.message}`
    );

    data = {
      clips: candidates.map(candidate => ({
        id: candidate.id,
        score: candidate.baseline,
        hook: candidate.excerpt.slice(0, 120),
        caption: candidate.excerpt.slice(0, 160),
        reason:
          'Fallback selection based on speech density.'
      }))
    };
  }

  const candidateMap = new Map(
    candidates.map(candidate => [
      candidate.id,
      candidate
    ])
  );

  const scored = (
    Array.isArray(data?.clips)
      ? data.clips
      : []
  )
    .map(item => {
      const candidate =
        candidateMap.get(String(item.id));

      if (!candidate) {
        return null;
      }

      return normalizeClip(
        {
          ...item,
          start: candidate.start,
          end: candidate.end,
          score: item.score
        },
        sourceDuration
      );
    })
    .filter(Boolean)
    .sort(
      (a, b) =>
        b.score - a.score
    );

  const selected = [];

  for (const clip of scored) {
    if (selected.length >= requested) {
      break;
    }

    const conflicts = selected.some(
      existing =>
        overlapRatio(existing, clip) >=
        MIN_OVERLAP_RATIO
    );

    if (conflicts) {
      continue;
    }

    selected.push(clip);
  }

  if (!selected.length) {
    for (
      const candidate of candidates.sort(
        (a, b) =>
          b.baseline - a.baseline
      )
    ) {
      const clip = normalizeClip(
        candidate,
        sourceDuration
      );

      if (!clip) {
        continue;
      }

      const conflicts = selected.some(
        existing =>
          overlapRatio(existing, clip) >=
          MIN_OVERLAP_RATIO
      );

      if (conflicts) {
        continue;
      }

      selected.push(clip);

      if (
        selected.length >= requested
      ) {
        break;
      }
    }
  }

  return selected
    .sort((a, b) => a.start - b.start)
    .map((clip, index) => ({
      ...clip,
      rank: index + 1
    }));
}

function assTime(seconds) {
  const safe = Math.max(
    0,
    Number(seconds) || 0
  );

  const hours =
    Math.floor(safe / 3600);

  const minutes =
    Math.floor((safe % 3600) / 60);

  const secondsPart =
    Math.floor((safe % 60) * 100) / 100;

  return (
    `${hours}:` +
    `${String(minutes).padStart(2, '0')}:` +
    `${secondsPart.toFixed(2).padStart(5, '0')}`
  );
}

function escapeASS(value) {
  return String(value ?? '')
    .replace(/[\\{}]/g, '\\$&')
    .replace(/\n/g, '\\N');
}

function createASS(words, start, end) {
  const header = `[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: TikTok,Arial,62,&H00FFFFFF,&H0000FFFF,&H00101010,&H96000000,-1,0,0,0,100,100,0,0,1,3,2,2,70,70,250,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
`;

  const selectedWords = words
    .filter(
      word =>
        word.end > start &&
        word.start < end
    )
    .map(word => ({
      ...word,
      start: Math.max(word.start, start),
      end: Math.min(word.end, end)
    }));

  const chunks = [];

  // Group captions into short, readable phrases.
  // Target: approximately 2-5 words per caption.
  let current = [];

  for (const word of selectedWords) {
    current.push(word);

    const text = current
      .map(item => item.word)
      .join(' ');

    const duration =
      current[current.length - 1].end -
      current[0].start;

    const shouldBreak =
      current.length >= 5 ||
      duration >= 2.2 ||
      /[.!?]$/.test(text);

    if (shouldBreak) {
      chunks.push(current);
      current = [];
    }
  }

  if (current.length) {
    chunks.push(current);
  }

  let events = '';

  for (const chunk of chunks) {
    if (!chunk.length) {
      continue;
    }

    const chunkStart = Math.max(
      0,
      chunk[0].start - start
    );

    const chunkEnd = Math.max(
      chunkStart + 0.15,
      chunk[chunk.length - 1].end - start
    );

    const text = chunk
      .map(item => escapeASS(item.word))
      .join(' ');

    events +=
      `Dialogue: 0,` +
      `${assTime(chunkStart)},` +
      `${assTime(chunkEnd)},` +
      `TikTok,,0,0,0,,` +
      `{\\an2}` +
      `${text}\n`;
  }

  return header + events;
}

async function transcribe(
  audio,
  captions
) {
  if (existsSync(captions)) {
    return JSON.parse(
      await run(
        'python',
        [
          'youtube_captions.py',
          captions
        ],
        process.cwd()
      )
    );
  }

  if (aiBackend === 'local') {
    return JSON.parse(
      await run(
        'python',
        [
          'local_ai.py',
          'transcribe',
          audio
        ],
        process.cwd()
      )
    );
  }

  const form = new FormData();

  form.append(
    'model',
    'gpt-4o-mini-transcribe'
  );

  form.append(
    'response_format',
    'verbose_json'
  );

  form.append(
    'timestamp_granularities[]',
    'word'
  );

  form.append(
    'file',
    new Blob([
      await readFile(audio)
    ]),
    'audio.mp3'
  );

  const response = await fetch(
    'https://api.openai.com/v1/audio/transcriptions',
    {
      method: 'POST',
      headers: {
        authorization:
          `Bearer ${process.env.OPENAI_API_KEY}`
      },
      body: form
    }
  );

  if (!response.ok) {
    throw new Error(
      `Transcription: ${await response.text()}`
    );
  }

  return response.json();
}

async function renderClip(
  video,
  directory,
  words,
  clip,
  index
) {
  const assFile = join(
    directory,
    `clip-${index + 1}.ass`
  );

  const outputFile = join(
    directory,
    `clip-${index + 1}.mp4`
  );

  await writeFile(
    assFile,
    createASS(
      words,
      clip.start,
      clip.end
    )
  );

  const escapedASS = assFile
    .replace(/\\/g, '/')
    .replace(/:/g, '\\:')
    .replace(/'/g, "\\'");

  const filter =
    `scale=1080:1920:` +
    `force_original_aspect_ratio=increase,` +
    `crop=1080:1920,` +
    `setsar=1,` +
    `ass='${escapedASS}'`;

  await run(
    'ffmpeg',
    [
      '-y',
      '-ss',
      String(clip.start),
      '-t',
      String(
        clip.end - clip.start
      ),
      '-i',
      video,
      '-vf',
      filter,
      '-af',
      'loudnorm=I=-14:TP=-1.5:LRA=11',
      '-c:v',
      'libx264',
      '-preset',
      process.env.FFMPEG_PRESET ||
        'veryfast',
      '-crf',
      process.env.FFMPEG_CRF ||
        '20',
      '-c:a',
      'aac',
      '-b:a',
      '128k',
      '-movflags',
      '+faststart',
      outputFile
    ],
    directory
  );

  if (!existsSync(outputFile)) {
    throw new Error(
      `FFmpeg did not create ${outputFile}`
    );
  }

  clip.file = outputFile;

  clip.duration = Number(
    (
      await getDuration(
        outputFile,
        directory
      )
    ).toFixed(2)
  );

  return clip;
}

async function processJob(job) {
  const directory = join(
    outputRoot,
    job.id
  );

  await mkdir(directory, {
    recursive: true
  });

  const video = join(
    directory,
    'source.mp4'
  );

  const audio = join(
    directory,
    'audio.mp3'
  );

  const captions = join(
    directory,
    'source.en.json3'
  );

  job.progress = 5;
  job.message =
    'Downloading source video';

  await run(
    'python',
    [
      '-m',
      'yt_dlp',
      '--no-playlist',
      '--write-subs',
      '--write-auto-subs',
      '--sub-langs',
      'en',
      '--sub-format',
      'json3',
      '-f',
      'bv*+ba/b',
      '--merge-output-format',
      'mp4',
      '-o',
      video,
      job.url
    ],
    directory
  );

  job.progress = 20;
  job.message =
    'Checking source video';

  const sourceDuration =
    await getDuration(
      video,
      directory
    );

  if (
    sourceDuration <
    MIN_CLIP_SECONDS
  ) {
    throw new Error(
      `Source is only ` +
      `${sourceDuration.toFixed(1)}s; ` +
      `at least ${MIN_CLIP_SECONDS}s ` +
      `is required.`
    );
  }

  job.progress = 25;
  job.message =
    'Extracting audio';

  await run(
    'ffmpeg',
    [
      '-y',
      '-i',
      video,
      '-vn',
      '-ac',
      '1',
      '-ar',
      '16000',
      audio
    ],
    directory
  );

  job.progress = 45;
  job.message =
    'Transcribing video';

  const transcript =
    await transcribe(
      audio,
      captions
    );

  const words =
    cleanWords(
      transcript.words
    );

  if (words.length < 20) {
    throw new Error(
      'Transcript contains too little usable speech.'
    );
  }

  job.progress = 60;
  job.message =
    'Finding unique high-retention moments';

  const clips =
    await chooseClips(
      words,
      sourceDuration,
      job.maxClips
    );

  if (!clips.length) {
    throw new Error(
      'No worthwhile clips were found.'
    );
  }

  const rendered = [];
  const failed = [];

  for (
    let index = 0;
    index < clips.length;
    index++
  ) {
    try {
      job.progress =
        Math.min(
          95,
          70 +
            Math.round(
              (index /
                clips.length) *
              25
            )
        );

      job.message =
        `Rendering clip ` +
        `${index + 1} of ` +
        `${clips.length}`;

      rendered.push(
        await renderClip(
          video,
          directory,
          words,
          clips[index],
          index
        )
      );
    } catch (error) {
      failed.push({
        clip: index + 1,
        error: error.message
      });
    }
  }

  if (!rendered.length) {
    throw new Error(
      `All clip renders failed: ` +
      failed
        .map(x => x.error)
        .join(' | ')
    );
  }

  job.progress = 100;
  job.message = 'Complete';
  job.status = 'done';

  job.result = {
    clips: rendered,
    failedClips: failed,
    transcript:
      transcript.text ||
      words
        .map(word => word.word)
        .join(' '),
    sourceDuration,
    outputDir: directory
  };
}

function validYouTubeURL(value) {
  try {
    const url = new URL(value);

    return (
      url.protocol === 'https:' &&
      [
        'youtube.com',
        'www.youtube.com',
        'm.youtube.com',
        'youtu.be'
      ].includes(url.hostname)
    );
  } catch {
    return false;
  }
}

createServer(
  async (req, res) => {
    try {
      if (req.method === 'OPTIONS') {
        res.writeHead(204, {
          'access-control-allow-origin': '*',
          'access-control-allow-methods':
            'GET,POST,OPTIONS',
          'access-control-allow-headers':
            'content-type'
        });

        return res.end();
      }

      if (
        req.method === 'GET' &&
        req.url === '/health'
      ) {
        return reply(res, 200, {
          ok: true,
          status: 'ready',
          aiBackend,
          model:
            aiBackend === 'local'
              ? model
              : 'gpt-4.1-mini'
        });
      }

      if (
        req.method === 'POST' &&
        req.url === '/jobs'
      ) {
        const {
          url,
          maxClips = 8
        } = await body(req);

        if (!validYouTubeURL(url)) {
          return reply(res, 400, {
            error:
              'url must be a valid YouTube HTTPS link'
          });
        }

        if (
          aiBackend === 'openai' &&
          !process.env.OPENAI_API_KEY
        ) {
          return reply(res, 500, {
            error:
              'OPENAI_API_KEY is missing while AI_BACKEND=openai'
          });
        }

        const requested =
          Math.min(
            Math.max(
              Number.parseInt(
                maxClips,
                10
              ) || 8,
              1
            ),
            MAX_REQUESTED_CLIPS
          );

        const job = {
          id: randomUUID(),
          url,
          maxClips: requested,
          status: 'running',
          progress: 0,
          message: 'Queued'
        };

        jobs.set(
          job.id,
          job
        );

        processJob(job).catch(
          error => {
            job.status = 'failed';
            job.progress = 100;
            job.message = 'Failed';
            job.error =
              error?.message ||
              String(error);
          }
        );

        return reply(res, 202, {
          id: job.id,
          status: job.status,
          poll:
            `/jobs/${job.id}`
        });
      }

      const id =
        req.url?.match(
          /^\/jobs\/([^/]+)$/
        )?.[1];

      if (
        req.method === 'GET' &&
        id
      ) {
        return jobs.has(id)
          ? reply(
              res,
              200,
              jobs.get(id)
            )
          : reply(
              res,
              404,
              {
                error:
                  'job not found'
              }
            );
      }

      return reply(
        res,
        404,
        {
          error: 'not found'
        }
      );
    } catch (error) {
      return reply(
        res,
        500,
        {
          error:
            error?.message ||
            String(error)
        }
      );
    }
  }
).listen(
  port,
  () =>
    console.log(
      `YTAC worker listening on ` +
      `http://127.0.0.1:${port}`
    )
);
