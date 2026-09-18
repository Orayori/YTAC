import { createServer } from 'node:http';

import { spawn } from 'node:child_process';

import { mkdir, readFile, writeFile } from 'node:fs/promises';

import { existsSync, readFileSync } from 'node:fs';

import { join, resolve } from 'node:path';

import { randomUUID } from 'node:crypto';

import {
  buildContentProfile,
  generateCandidates as generateIntelligenceCandidates,
  buildIntelligencePrompt
} from './intelligence_engine.mjs';



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



  const score = Math.max(
    0,
    Math.min(100, Number(raw.score) || 0)
  );

  const scores =
    raw.scores &&
    typeof raw.scores === 'object'
      ? {
          hook: Math.max(
            0,
            Math.min(
              100,
              Number(raw.scores.hook) || 0
            )
          ),
          curiosity: Math.max(
            0,
            Math.min(
              100,
              Number(raw.scores.curiosity) || 0
            )
          ),
          emotion: Math.max(
            0,
            Math.min(
              100,
              Number(raw.scores.emotion) || 0
            )
          ),
          story: Math.max(
            0,
            Math.min(
              100,
              Number(raw.scores.story) || 0
            )
          ),
          selfContained: Math.max(
            0,
            Math.min(
              100,
              Number(raw.scores.selfContained) || 0
            )
          ),
          practicalValue: Math.max(
            0,
            Math.min(
              100,
              Number(raw.scores.practicalValue) || 0
            )
          ),
          quotable: Math.max(
            0,
            Math.min(
              100,
              Number(raw.scores.quotable) || 0
            )
          ),
          ending: Math.max(
            0,
            Math.min(
              100,
              Number(raw.scores.ending) || 0
            )
          )
        }
      : null;

  return {
    start: Number(start.toFixed(2)),
    end: Number(end.toFixed(2)),
    score,
    title: String(
      raw.title || ''
    ).slice(0, 140),
    hook: String(
      raw.hook || ''
    ).slice(0, 240),
    caption: String(
      raw.caption || raw.hook || ''
    ).slice(0, 280),
    reason: String(
      raw.reason || ''
    ).slice(0, 500),
    signals: Array.isArray(raw.signals)
      ? raw.signals
          .map(value => String(value).trim())
          .filter(Boolean)
          .slice(0, 8)
      : [],
    scores,
    contentType: String(
      raw.contentType || ''
    ).slice(0, 60),
    density: String(
      raw.density || ''
    ).slice(0, 20)
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
  const profile =
    buildContentProfile(
      words,
      sourceDuration
    );

  const candidates =
    generateIntelligenceCandidates(
      words,
      sourceDuration
    );

  if (!candidates.length) {
    throw new Error(
      'Not enough spoken content for 45-60 second clips.'
    );
  }

  const prompt =
    buildIntelligencePrompt(
      profile,
      candidates,
      requested
    );

  let data;

  try {
    data = await askAI(prompt);
  } catch (error) {
    console.warn(
      `AI intelligence analysis failed; using fallback: ${error.message}`
    );

    data = {
      contentType: 'other',
      density:
        profile.speechDensity >= 0.55
          ? 'high'
          : profile.speechDensity >= 0.30
            ? 'medium'
            : 'low',
      clips: candidates.map(
        candidate => ({
          id: candidate.id,
          score: candidate.baseline,
          title:
            candidate.excerpt.slice(
              0,
              100
            ),
          hook:
            candidate.excerpt.slice(
              0,
              120
            ),
          caption:
            candidate.excerpt.slice(
              0,
              160
            ),
          reason:
            'Fallback selection based on speech density and temporal coverage.',
          signals: [
            'speech_density'
          ],
          scores: {
            hook:
              candidate.baseline,
            curiosity: 0,
            emotion: 0,
            story: 0,
            selfContained: 0,
            practicalValue: 0,
            quotable: 0,
            ending: 0
          }
        })
      )
    };
  }

  const candidateMap =
    new Map(
      candidates.map(
        candidate => [
          candidate.id,
          candidate
        ]
      )
    );

  const contentType =
    String(
      data?.contentType ||
        'other'
    )
      .trim()
      .slice(0, 60);

  const density =
    String(
      data?.density ||
        'medium'
    )
      .trim()
      .slice(0, 20);

  const scored = (
    Array.isArray(data?.clips)
      ? data.clips
      : []
  )
    .map(item => {
      const candidate =
        candidateMap.get(
          String(item.id)
        );

      if (!candidate) {
        return null;
      }

      const score =
        Math.max(
          0,
          Math.min(
            100,
            Number(item.score) ||
              candidate.baseline
          )
        );

      return normalizeClip(
        {
          ...item,
          start:
            candidate.start,
          end:
            candidate.end,
          score,
          contentType,
          density
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
    if (
      selected.length >=
      requested
    ) {
      break;
    }

    if (
      selected.some(
        existing =>
          overlapRatio(
            existing,
            clip
          ) >=
          MIN_OVERLAP_RATIO
      )
    ) {
      continue;
    }

    selected.push(clip);
  }

  /*
   * Only use deterministic fallback candidates if the AI
   * returned too few usable clips. This preserves quality
   * while preventing a failed model call from breaking
   * the complete pipeline.
   */
  if (
    selected.length <
    requested
  ) {
    for (
      const candidate of [
        ...candidates
      ].sort(
        (a, b) =>
          b.baseline -
          a.baseline
      )
    ) {
      if (
        selected.length >=
        requested
      ) {
        break;
      }

      const clip =
        normalizeClip(
          {
            ...candidate,
            score:
              candidate.baseline,
            title:
              candidate.excerpt.slice(
                0,
                100
              ),
            hook:
              candidate.excerpt.slice(
                0,
                120
              ),
            caption:
              candidate.excerpt.slice(
                0,
                160
              ),
            reason:
              'Fallback candidate retained to preserve temporal coverage.',
            signals: [
              'speech_density'
            ],
            contentType,
            density
          },
          sourceDuration
        );

      if (!clip) {
        continue;
      }

      if (
        selected.some(
          existing =>
            overlapRatio(
              existing,
              clip
            ) >=
            MIN_OVERLAP_RATIO
        )
      ) {
        continue;
      }

      selected.push(clip);
    }
  }

  return selected
    .sort(
      (a, b) =>
        a.start - b.start
    )
    .map(
      (clip, index) => ({
        ...clip,
        rank: index + 1
      })
    );
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



/*

 * Creates TikTok-style active-word captions.

 *

 * - 2-5 words per caption

 * - Current spoken word = yellow

 * - Other words = white

 * - Bold

 * - Large

 * - Black outline

 * - Shadow

 * - Bottom-center positioning

 */

function createASS(words, start, end) {
  const header = `[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes
WrapStyle: 2
Collisions: Normal

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: TikTok,Arial Rounded MT Bold,86,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,6,3,2,120,120,330,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
`;

  const selectedWords = words
    .filter(
      word =>
        word.end > start &&
        word.start < end &&
        String(word.word || '').trim()
    )
    .map(word => ({
      ...word,
      start: Math.max(start, Number(word.start)),
      end: Math.min(end, Number(word.end))
    }))
    .filter(
      word =>
        Number.isFinite(word.start) &&
        Number.isFinite(word.end) &&
        word.end > word.start
    );

  let events = '';

  for (let i = 0; i < selectedWords.length; i++) {
    const word = selectedWords[i];

    let wordStart =
      word.start - start;

    let wordEnd =
      word.end - start;

    const MIN_WORD_DURATION = 0.08;
    const MAX_WORD_DURATION = 1.8;

    wordStart = Math.max(
      0,
      wordStart
    );

    const duration = Math.min(
      MAX_WORD_DURATION,
      Math.max(
        MIN_WORD_DURATION,
        wordEnd - wordStart
      )
    );

    wordEnd =
      wordStart + duration;

    const next =
      selectedWords[i + 1];

    if (next) {
      const nextStart =
        Math.max(
          0,
          next.start - start
        );

      /*
       * Never allow the current word
       * to overlap the next word.
       */
      if (wordEnd > nextStart) {
        wordEnd =
          Math.max(
            wordStart + MIN_WORD_DURATION,
            nextStart
          );
      }

      /*
       * If there is a tiny natural gap,
       * allow the current word to remain
       * visible until the next word starts.
       */
      const gap =
        nextStart - wordEnd;

      if (
        gap > 0 &&
        gap <= 0.10
      ) {
        wordEnd = nextStart;
      }
    }

    wordEnd =
      Math.min(
        end - start,
        wordEnd
      );

    if (wordEnd <= wordStart) {
      continue;
    }

    const text = escapeASS(
      String(word.word || '')
        .trim()
        .toUpperCase()
    );

    if (!text) {
      continue;
    }

    const caption =
      `{\\an2\\c&H00FFFF&\\fs86}` +
      `{\\t(0,100,\\fs96)}` +
      text;

    events +=
      `Dialogue: 0,` +
      `${assTime(wordStart)},` +
      `${assTime(wordEnd)},` +
      `TikTok,,0,0,0,,` +
      caption +
      `\n`;
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



  /*

   * Smart vertical framing

   *

   * smart_crop.py samples the clip, detects the

   * speaker/face position, and calculates the best

   * horizontal crop for a 1080x1920 video.

   *

   * If detection fails for any reason, the worker

   * automatically falls back to the center crop.

   */



  let framing = {

    scaleWidth: 1080,

    scaleHeight: 1920,

    cropX: 0,

    cropY: 0,

    focusX: 0.5,

    facesDetected: false

  };



  try {

    const smartCropScript = join(

      process.cwd(),

      'smart_crop.py'

    );



    const smartCropOutput = await run(

      'python',

      [

        smartCropScript,

        video,

        String(clip.start),

        String(clip.end)

      ],

      directory

    );



    const detected =

      JSON.parse(

        smartCropOutput.trim()

      );



    if (

      Number.isFinite(

        Number(detected.scaleWidth)

      ) &&

      Number.isFinite(

        Number(detected.scaleHeight)

      ) &&

      Number.isFinite(

        Number(detected.cropX)

      ) &&

      Number.isFinite(

        Number(detected.cropY)

      )

    ) {

      framing = {

        ...framing,

        ...detected

      };



      console.log(

        `Smart framing: ` +

        `focusX=${detected.focusX}, ` +

        `facesDetected=${detected.facesDetected}`

      );

    }

  } catch (error) {

    console.warn(

      `Smart framing failed; ` +

      `using center crop: ` +

      error.message

    );

  }



  const filter =

    `scale=${Math.max(

      1080,

      Math.round(

        framing.scaleWidth

      )

    )}:${Math.max(

      1920,

      Math.round(

        framing.scaleHeight

      )

    )},` +

    `crop=1080:1920:` +

    `${Math.max(

      0,

      Math.round(

        framing.cropX

      )

    )}:` +

    `${Math.max(

      0,

      Math.round(

        framing.cropY

      )

    )},` +

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

    'Analyzing content type, virality signals, and unique high-retention moments';



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
    intelligence: {
      version: '3.0',
      candidateLimit:
        Number(
          process.env.MAX_CANDIDATES ||
            80
        ),
      longVideoThreshold: 1800,
      analysisChunkSeconds: 1200,
      analysisChunkOverlap: 60
    },
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

