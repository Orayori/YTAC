import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { existsSync, readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { randomUUID } from 'node:crypto';

loadEnv();
const port = Number(process.env.PORT || 8787);
const outputRoot = resolve(process.env.OUTPUT_DIR || 'output');
const jobs = new Map();

function loadEnv() {
  if (!existsSync('.env')) return;
  for (const line of requireText('.env').split(/\r?\n/)) {
    const match = line.match(/^([A-Z0-9_]+)=(.*)$/);
    if (match && !process.env[match[1]]) process.env[match[1]] = match[2];
  }
}
function requireText(path) { return readFileSync(path, 'utf8'); }
function run(command, args, cwd) {
  return new Promise((resolveRun, reject) => {
    const child = spawn(command, args, { cwd, windowsHide: true }); let text = '';
    child.stdout.on('data', d => text += d); child.stderr.on('data', d => text += d);
    child.on('error', reject); child.on('close', code => code ? reject(new Error(`${command} exited ${code}: ${text.slice(-1200)}`)) : resolveRun(text));
  });
}
function reply(res, code, value) { res.writeHead(code, { 'content-type': 'application/json' }); res.end(JSON.stringify(value)); }
async function body(req) { let s = ''; for await (const c of req) s += c; return JSON.parse(s || '{}'); }
function assTime(seconds) { const h = Math.floor(seconds / 3600), m = Math.floor(seconds % 3600 / 60), s = seconds % 60; return `${h}:${String(m).padStart(2,'0')}:${s.toFixed(2).padStart(5,'0')}`; }
function esc(s) { return String(s).replace(/[\\{}]/g, '\\$&').replace(/\n/g, '\\N'); }
function ass(words, start, end) {
  const header = `[Script Info]\nScriptType: v4.00+\n[V4+ Styles]\nFormat: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\nStyle: TikTok,Arial,22,&H00FFFFFF,&H0000FFFF,&H00101010,&H96000000,-1,0,0,0,100,100,0,0,1,2.5,1,2,35,35,130,1\n[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n`;
  return header + words.filter(w => w.start >= start && w.end <= end).map((w, i, a) => {
    const t0 = Math.max(0, w.start - start), t1 = Math.max(t0 + .12, w.end - start);
    const before = a.slice(Math.max(0, i - 3), i).map(x => esc(x.word)).join(' ');
    const after = a.slice(i + 1, i + 4).map(x => esc(x.word)).join(' ');
    return `Dialogue: 0,${assTime(t0)},${assTime(t1)},TikTok,,0,0,0,,${before} {\\c&H00FFFF&}${esc(w.word)}{\\c&HFFFFFF&} ${after}`;
  }).join('\n');
}
async function openai(path, payload) {
  const response = await fetch(`https://api.openai.com/v1/${path}`, { method: 'POST', headers: { authorization: `Bearer ${process.env.OPENAI_API_KEY}`, 'content-type': 'application/json' }, body: JSON.stringify(payload) });
  if (!response.ok) throw new Error(`OpenAI: ${await response.text()}`); return response.json();
}
async function transcribe(file, captions) {
  if (existsSync(captions)) return JSON.parse(await run('python', ['youtube_captions.py', captions], process.cwd()));
  if ((process.env.AI_BACKEND || 'local') === 'local') {
    return JSON.parse(await run('python', ['local_ai.py', 'transcribe', file], process.cwd()));
  }
  const form = new FormData(); form.append('model', 'gpt-4o-mini-transcribe'); form.append('response_format', 'verbose_json'); form.append('timestamp_granularities[]', 'word'); form.append('file', new Blob([await readFile(file)]), 'audio.mp3');
  const r = await fetch('https://api.openai.com/v1/audio/transcriptions', { method: 'POST', headers: { authorization: `Bearer ${process.env.OPENAI_API_KEY}` }, body: form });
  if (!r.ok) throw new Error(`Transcription: ${await r.text()}`); return r.json();
}
async function chooseClips(transcript, maxClips) {
  const prompt = `Select up to ${maxClips} non-overlapping 45-60 second clips. Return strict JSON array only: [{start,end,rank,hook,caption}]. Rules: self-contained continuing story without repeats; hook in first 2-3 seconds; high energy/payoff; exclude ads, sponsors, pauses, filler. rank 1 is highest predicted watch-through. Transcript with seconds:\n${transcript}`;
  if ((process.env.AI_BACKEND || 'local') === 'local') {
    const r = await fetch('http://127.0.0.1:11434/api/chat', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ model: process.env.OLLAMA_MODEL || 'llama3.2', stream: false, format: 'json', messages: [{ role: 'user', content: prompt + '\nWrap array as {"clips": [...]}' }] }) });
    if (!r.ok) throw new Error(`Ollama: ${await r.text()}`); return JSON.parse((await r.json()).message.content).clips;
  }
  const r = await openai('chat/completions', { model: 'gpt-4.1-mini', response_format: { type: 'json_object' }, messages: [{ role: 'user', content: prompt + '\nWrap array as {"clips": [...]}' }] });
  return JSON.parse(r.choices[0].message.content).clips;
}
async function processJob(job) {
  const dir = join(outputRoot, job.id); await mkdir(dir, { recursive: true });
  const video = join(dir, 'source.mp4'), audio = join(dir, 'audio.mp3');
  const captions = join(dir, 'source.en.json3');
  await run('python', ['-m', 'yt_dlp', '--no-playlist', '--write-subs', '--write-auto-subs', '--sub-langs', 'en', '--sub-format', 'json3', '-f', 'bv*+ba/b', '--merge-output-format', 'mp4', '-o', video, job.url], dir);
  await run('ffmpeg', ['-y', '-i', video, '-vn', '-ac', '1', '-ar', '16000', audio], dir);
  const transcript = await transcribe(audio, captions); const clips = await chooseClips((transcript.words || []).map(w => `[${w.start.toFixed(1)}-${w.end.toFixed(1)}] ${w.word}`).join(' '), job.maxClips);
  for (const [i, clip] of clips.entries()) {
    const assFile = join(dir, `clip-${i + 1}.ass`), out = join(dir, `clip-${i + 1}.mp4`);
    await writeFile(assFile, ass(transcript.words || [], clip.start, clip.end));
    await run('ffmpeg', ['-y', '-ss', String(clip.start), '-to', String(clip.end), '-i', video, '-vf', `crop=ih*9/16:ih:(iw-ih*9/16)/2:0,scale=1080:1920,ass=${assFile.replace(/\\/g, '/').replace(':', '\\:')}`, '-af', 'loudnorm', '-c:v', 'libx264', '-c:a', 'aac', '-movflags', '+faststart', out], dir);
    clip.file = out; clip.duration = Number((clip.end - clip.start).toFixed(2));
  }
  job.result = { clips, transcript: transcript.text, outputDir: dir }; job.status = 'done';
}
createServer(async (req, res) => {
  try {
    if (req.method === 'POST' && req.url === '/jobs') {
      const { url, maxClips = 8 } = await body(req);
      if (!/^https:\/\/(www\.)?youtube\.com\/|^https:\/\/youtu\.be\//.test(url || '')) return reply(res, 400, { error: 'url must be a YouTube HTTPS link' });
      if ((process.env.AI_BACKEND || 'local') !== 'local' && !process.env.OPENAI_API_KEY) return reply(res, 500, { error: 'OPENAI_API_KEY is missing' });
      const job = { id: randomUUID(), url, maxClips: Math.min(Math.max(Number(maxClips) || 8, 1), 20), status: 'running' }; jobs.set(job.id, job); processJob(job).catch(e => { job.status = 'failed'; job.error = e.message; }); return reply(res, 202, { id: job.id, status: job.status, poll: `/jobs/${job.id}` });
    }
    const id = req.url?.match(/^\/jobs\/([^/]+)$/)?.[1]; if (req.method === 'GET' && id) return jobs.has(id) ? reply(res, 200, jobs.get(id)) : reply(res, 404, { error: 'job not found' });
    reply(res, 404, { error: 'not found' });
  } catch (e) { reply(res, 500, { error: e.message }); }
}).listen(port, () => console.log(`Clip worker on http://127.0.0.1:${port}`));
