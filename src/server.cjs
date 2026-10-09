'use strict';

/**
 * server.cjs — ThinkDrop Voice Bridge (thin rewrite)
 *
 * Replaces the old 5.9k-line voice pipeline (archived in src/deprecated/)
 * with a small bridge that owns exactly three things:
 *
 *   1. A hidden real-Chrome worker page (companion-driver.cjs + Playwright)
 *      that runs webkitSpeechRecognition — the only STT path.
 *   2. A WebSocket (/voice.ws) carrying worker events → ThinkDrop main,
 *      and TTS commands/audio → the worker.
 *   3. TTS synthesis via the provider chain (voice-provider.cjs):
 *      openai (gpt-4o-mini-tts "marin") → cartesia → inworld → groq → macos.
 *
 * All conversation intelligence (classification, translation, persona,
 * StateGraph, handoffs) lives in comms-graph — transcripts are forwarded
 * to main.js /voice.event and routed exactly like typed prompts.
 *
 * Endpoints:
 *   GET  /health
 *   GET  /voice/worker            → the worker HTML page (loaded by Chrome)
 *   WS   /voice.ws                → worker page connection
 *   POST /voice.session.start     → launch hidden Chrome, begin listening
 *   POST /voice.session.stop      → suspend listening (browser stays resident)
 *   GET  /voice.session.status
 *   POST /voice.say   {text,lang} → synthesize + stream audio to worker
 *   POST /voice.stop              → abort playback (barge-in)
 *   POST /voice.provider          → {action:'list'} | {action:'set',provider}
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const { WebSocketServer } = require('ws');
const axios = require('axios');
const logger = require('./logger.cjs');
const { CompanionDriver } = require('./companion-driver.cjs');
const voiceProvider = require('./voice-provider.cjs');
const fillers = require('./fillers.cjs');
const fillerPolicy = require('./filler-policy.cjs');
const wakeWord = require('./wake-word.cjs');
const realtime = require('./realtime/openai.cjs');

require('dotenv').config({ path: path.join(__dirname, '..', '.env') });
// Root .env holds shared service keys (MCP_USER_MEMORY_API_KEY etc.) —
// loaded after this service's .env so service-local values always win.
require('dotenv').config({ path: path.join(__dirname, '..', '..', '..', '.env') });

const PORT = parseInt(process.env.PORT || '3006', 10);
const MAIN_PORT = parseInt(process.env.THINKDROP_MAIN_PORT || '3010', 10);
const MAIN_EVENT_URL = `http://localhost:${MAIN_PORT}/voice.event`;
const WORKER_HTML = path.join(__dirname, '..', 'public', 'voice-worker.html');

// ── Session / worker state ────────────────────────────────────────────────────
let sessionWanted = false;
let sessionMode = 'pipeline'; // 'pipeline' (Chrome SR + TTS) | 'realtime' (S2S)
let _lastSelectionCtx = null; // { armed, captured, sourceApp, excerpt } pushed by main
let _lastScreenCtx = null;    // { appName, windowTitle, url, text, capturedAt } pushed by monitor
let workerSocket = null;   // active WS connection from the worker page
let workerReady = false;
let workerActivated = false; // 'session:active' already sent to THIS worker page

// ── Voice history (context for realtime sessions + ambient gating) ───────────
const voiceHistory = []; // [{role:'user'|'assistant', text}] — last N voice turns
const VOICE_HISTORY_MAX = 12;
function noteVoice(role, text) {
  const t = (text || '').trim();
  if (!t) return;
  voiceHistory.push({ role, text: t.slice(0, 300) });
  if (voiceHistory.length > VOICE_HISTORY_MAX) voiceHistory.shift();
}

// ── Echo suppression ──────────────────────────────────────────────────────────
// We know the exact text we're about to speak, so a transcript that matches it
// while it's playing (or just after) is our own TTS echoing back into the mic.
let lastSpoken = null;        // { text, norm, until }
let speechInFlight = false;
let lastFinal = { norm: '', at: 0 }; // consecutive-duplicate dedupe
const ECHO_GRACE_MS = parseInt(process.env.VOICE_ECHO_GRACE_MS || '3000', 10);
const ECHO_SIM_THRESHOLD = parseFloat(process.env.VOICE_ECHO_SIM || '0.6');

function normalizeText(t) {
  return (t || '').toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

// Word-overlap coefficient — echo clips are usually a subsequence of what we
// spoke, so containment scores better than Jaccard here.
function echoSimilarity(a, b) {
  if (!a || !b) return 0;
  if (a === b || a.includes(b) || b.includes(a)) return 1;
  const wa = new Set(a.split(' ').filter(w => w.length > 1));
  const wb = new Set(b.split(' ').filter(w => w.length > 1));
  if (!wa.size || !wb.size) return 0;
  let hit = 0;
  for (const w of wa) if (wb.has(w)) hit++;
  return hit / Math.min(wa.size, wb.size);
}

function noteSpeaking(text) {
  // Estimated duration (~60ms/char) bounds the filter even if speak-done
  // never arrives; speak-done tightens it to ECHO_GRACE_MS.
  const est = Math.min(Math.max(text.length * 60, 2000), 30000);
  lastSpoken = { text, norm: normalizeText(text), until: Date.now() + est + ECHO_GRACE_MS };
  speechInFlight = true;
  clearTimeout(sleepTimer);   // idle clock pauses while the assistant is talking
  clearTimeout(ambientTimer); ambientTimer = null; // not silence anymore
}

function noteSpeechDone() {
  speechInFlight = false;
  if (lastSpoken) lastSpoken.until = Date.now() + ECHO_GRACE_MS;
  armSleepTimer();    // idle clock starts counting only once we've gone quiet
  scheduleAmbient();  // and the room may get a soft hum after a while
}

/** Returns a drop reason string, or false if the transcript is genuine. */
function echoOrDupeReason(msg) {
  const now = Date.now();
  const norm = normalizeText(msg.text);
  if (!norm) return 'empty';
  if (norm === lastFinal.norm && now - lastFinal.at < 3000) return 'duplicate';
  if (lastSpoken && (speechInFlight || now < lastSpoken.until)) {
    const sim = echoSimilarity(norm, lastSpoken.norm);
    if (sim >= ECHO_SIM_THRESHOLD) return `echo(${sim.toFixed(2)})`;
  }
  return false;
}

// ── Turn aggregation ──────────────────────────────────────────────────────────
// Humans wait ~200-700ms for turn end before answering. Each `final` starts a
// hold window; another final or fresh user speech inside the window keeps the
// turn open. On flush the buffered finals merge into ONE dispatch — so
// "do X, then Y, also Z" becomes a single task instead of three.
const TURN_HOLD_MS = parseInt(process.env.VOICE_TURN_HOLD_MS || '900', 10);
let turnBuffer = [];
let turnLang = 'en';
let turnTimer = null;

function flushTurn() {
  turnTimer = null;
  if (!turnBuffer.length) return;
  const text = turnBuffer.join(' ').replace(/\s+/g, ' ').trim();
  const lang = turnLang;
  turnBuffer = [];
  noteVoice('user', text);
  relayToMain({ type: 'final', text, lang });
}

function holdTurn(holdMs) {
  clearTimeout(turnTimer);
  turnTimer = setTimeout(flushTurn, holdMs);
}

// ── Filler playback ───────────────────────────────────────────────────────────
// Sends a pre-generated clip to the worker. The clip IS speech output, so it
// goes through the same echo gate (noteSpeaking) as real TTS.
function playFiller(category, lang) {
  const e = fillers.pick(category, lang);
  if (!e) return false;
  noteSpeaking(e.text);
  const src = `http://127.0.0.1:${PORT}/voice.filler.file?p=${encodeURIComponent(e.rel)}`;
  sendToWorker({ type: 'filler-play', src });
  logger.info('[VoiceBridge] filler', { cat: e.category, lang: e.lang, text: e.text });
  return true;
}

// ── Sleep mode ────────────────────────────────────────────────────────────────
// While sleeping the worker keeps listening but the server drops every
// transcript that isn't a wake phrase — zero LLM/TTS spend. GhostLayer shows
// a pulsing "sleeping" pill so the state is visible.
const SLEEP_AFTER_MS = parseInt(process.env.VOICE_SLEEP_AFTER_MS || '90000', 10);
const SLEEP_ENABLED = process.env.VOICE_SLEEP !== '0';
let sleeping = false;
let sleepTimer = null;

function armSleepTimer() {
  if (!SLEEP_ENABLED || !sessionWanted || sleeping) return;
  clearTimeout(sleepTimer);
  sleepTimer = setTimeout(() => enterSleep('idle'), SLEEP_AFTER_MS);
}

function enterSleep(reason) {
  if (sleeping || !sessionWanted) return;
  // Never sleep mid-response or mid-turn — the idle clock pauses while audio
  // is in flight, but belt-and-suspenders: re-arm instead of sleeping.
  if (speechInFlight || turnTimer) { armSleepTimer(); return; }
  sleeping = true;
  clearTimeout(ambientTimer); ambientTimer = null;
  flushTurn(); // don't strand a held mid-turn utterance
  fillerPolicy.reset();
  playFiller('going_to_sleep', turnLang); // audible cue it went quiet
  logger.info('[VoiceBridge] sleeping', { reason });
  relayToMain({ type: 'sleep-state', sleeping: true, reason });
  relayToMain({ type: 'state', state: 'sleeping' });
  sendToWorker({ type: 'sleep', active: true });
}

function wakeUp(source, lang) {
  if (!sleeping) return;
  sleeping = false;
  logger.info('[VoiceBridge] awake', { source });
  relayToMain({ type: 'sleep-state', sleeping: false, source });
  relayToMain({ type: 'state', state: 'listening' });
  sendToWorker({ type: 'sleep', active: false });
  playFiller('wake', lang);
  armSleepTimer();
}

// Exact-match spoken switch into Talk Mode — tight patterns so
// "let's talk about X" mid-sentence can't false-trigger.
const TALK_CMD = /^(hey thinkdrop,? )?(talk|realtime|conversation|real talk) mode\.?$|^let'?s (just )?talk\.?$|^talk to me\.?$/i;

// Spoken cancel — kills the running voice task instead of becoming a new one.
const CANCEL_CMD = /^(cancel|abort|stop)( that| it| the task| everything)?\.?$|^never ?mind\.?$|^forget (it|that)\.?$|^shut (up|it)\.?$/i;

// Short hard-interrupt words that justify barge-in even under 10 chars.
const BARGE_SHORT_RE = /\b(stop|wait|hold on|cancel|hey)\b/i;

// Mid-utterance backchannels — fire during sustained speech (≥3s of interims),
// throttled by the policy's own 5s minimum. firstInterimAt resets on burst gaps
// and on finals so a "new monologue" gets its own 3s wind-up.
let firstInterimAt = 0;
function maybeMidSpeechBackchannel(lang) {
  if (!firstInterimAt || Date.now() - firstInterimAt < 3000) return;
  if (fillerPolicy.decideBackchannel()) playFiller('backchannel', lang || turnLang);
}

function voiceCancel(lang) {
  clearTimeout(turnTimer); turnTimer = null; turnBuffer = [];
  noteSpeechDone();
  sendToWorker({ type: 'stop' });
  relayToMain({ type: 'voice-cancel' });
  playFiller('ack', lang); // "ok" — confirms without another LLM round-trip
  logger.info('[VoiceBridge] spoken cancel');
}

// ── Ambient silence (humming, sighs, soft singing) ────────────────────────────
// During true idle — session up, nobody speaking, no turn pending, not
// sleeping, pipeline mode (realtime owns its own silence) — an occasional
// soft vocalization keeps the presence alive. Any activity cancels it.
const AMBIENT_ENABLED = process.env.VOICE_AMBIENT !== '0';
const AMBIENT_MIN_MS = parseInt(process.env.VOICE_AMBIENT_MIN_MS || '25000', 10);
const AMBIENT_MAX_MS = parseInt(process.env.VOICE_AMBIENT_MAX_MS || '60000', 10);
const AMBIENT_CATS = ['humming', 'sigh', 'singing', 'breath'];
let ambientTimer = null;
let lastInterimAt = 0;

function _silenceOk() {
  return sessionWanted && sessionMode === 'pipeline' && !sleeping &&
    !speechInFlight && !turnTimer && !turnBuffer.length &&
    Date.now() - lastInterimAt > 4000;
}

function scheduleAmbient() {
  clearTimeout(ambientTimer); ambientTimer = null;
  if (!AMBIENT_ENABLED || !_silenceOk()) return;
  const wait = AMBIENT_MIN_MS + Math.random() * Math.max(0, AMBIENT_MAX_MS - AMBIENT_MIN_MS);
  ambientTimer = setTimeout(() => {
    ambientTimer = null;
    if (_silenceOk()) playFiller(AMBIENT_CATS[Math.floor(Math.random() * AMBIENT_CATS.length)], turnLang);
    scheduleAmbient();
  }, wait);
}

// ── Proactive thoughts (heartbeat / synthesis-agent → /voice.speak) ───────────
// The personality service already POSTs here; this endpoint is the gated
// mouthpiece. Floods collapse to the newest thought; nothing speaks while
// the user is mid-turn or we're mid-sentence — it retries once at silence.
const THOUGHTS_ENABLED = process.env.VOICE_THOUGHTS !== '0';
const THOUGHT_MIN_GAP_MS = parseInt(process.env.VOICE_THOUGHT_MIN_GAP_MS || '45000', 10);
let lastThoughtAt = 0;
let pendingThoughtTimer = null;

async function speakThought({ text, lang }) {
  const t = (text || '').trim();
  if (!t) return { ok: false, error: 'text required' };
  if (!THOUGHTS_ENABLED) return { ok: true, spoken: false, reason: 'thoughts-disabled' };
  if (!sessionWanted) return { ok: true, spoken: false, reason: 'session-inactive' };
  if (sleeping) return { ok: true, spoken: false, reason: 'sleeping' };

  if (sessionMode === 'realtime') {
    // The model voices it in its own words at the next natural point.
    sendToWorker({ type: 'rt-inject', text: `Share this with the user naturally, in your own words, keeping it brief: ${t}` });
    lastThoughtAt = Date.now();
    noteVoice('assistant', t);
    return { ok: true, spoken: true, via: 'realtime' };
  }

  if (Date.now() - lastThoughtAt < THOUGHT_MIN_GAP_MS) {
    relayToMain({ type: 'thought-dropped', reason: 'rate-limit', text: t.slice(0, 120) });
    return { ok: true, spoken: false, reason: 'rate-limit' };
  }
  if (speechInFlight || turnTimer || turnBuffer.length) {
    clearTimeout(pendingThoughtTimer);
    pendingThoughtTimer = setTimeout(() => {
      pendingThoughtTimer = null;
      if (sessionWanted && !sleeping && !speechInFlight && !turnTimer && !turnBuffer.length) {
        lastThoughtAt = Date.now();
        noteVoice('assistant', t);
        say({ text: t, lang }).catch(() => {});
      } else {
        relayToMain({ type: 'thought-dropped', reason: 'busy', text: t.slice(0, 120) });
      }
    }, 4000);
    return { ok: true, spoken: 'deferred' };
  }
  lastThoughtAt = Date.now();
  noteVoice('assistant', t);
  const r = await say({ text: t, lang });
  return { ok: true, ...r, via: 'pipeline' };
}

// ── Realtime context — what the model knows when the call opens ───────────────
const COMMS_PORT = parseInt(process.env.COMMS_GRAPH_PORT || '3015', 10);
const MEMORY_PORT = parseInt(process.env.MEMORY_SERVICE_PORT || '3001', 10);

/** Live frontmost-app snapshot — app name, window title, browser URL. */
async function fetchActiveAppContext() {
  try {
    const headers = {};
    if (process.env.MCP_USER_MEMORY_API_KEY) headers.Authorization = `Bearer ${process.env.MCP_USER_MEMORY_API_KEY}`;
    const r = await axios.post(`http://localhost:${MEMORY_PORT}/memory.getActiveAppContext`,
      { version: 'mcp.v1', service: 'user-memory', requestId: `voice-${Date.now()}`,
        action: 'memory.getActiveAppContext', payload: {} },
      { timeout: 800, httpAgent: relayAgent, headers });
    const app = r.data?.data?.app;
    if (!app || !app.appName) return null;
    return {
      appName: app.appName,
      windowTitle: app.windowTitle || '',
      url: app.url || null,
      filePath: app.filePath || null,
    };
  } catch (_) { return null; }
}

function describeActiveApp(app) {
  const parts = [`frontmost app is ${app.appName}`];
  if (app.windowTitle) parts.push(`window "${app.windowTitle}"`);
  if (app.url) parts.push(`page ${app.url}`);
  if (app.filePath) parts.push(`open file ${app.filePath}`);
  return parts.join(', ');
}

/** Latest background-monitor OCR capture — real screen text, not pixels. */
async function fetchRecentScreenText(maxAgeSeconds = 120) {
  try {
    const headers = {};
    if (process.env.MCP_USER_MEMORY_API_KEY) headers.Authorization = `Bearer ${process.env.MCP_USER_MEMORY_API_KEY}`;
    const r = await axios.post(`http://localhost:${MEMORY_PORT}/memory.getRecentOcr`,
      { version: 'mcp.v1', service: 'user-memory', requestId: `voice-${Date.now()}`,
        action: 'memory.getRecentOcr', payload: { maxAgeSeconds } },
      { timeout: 800, httpAgent: relayAgent, headers });
    const c = r.data?.data?.capture;
    if (!r.data?.data?.available || !c || !c.text) return null;
    return {
      text: String(c.text).replace(/\s+/g, ' ').trim(),
      appName: c.appName,
      windowTitle: c.windowTitle,
      capturedAt: c.capturedAt,
      ageMs: c.ageMs,
    };
  } catch (_) { return null; }
}

async function buildRealtimeContext() {
  let block = '';
  const [app, screen] = await Promise.all([fetchActiveAppContext(), fetchRecentScreenText(300)]);
  if (app) {
    lastRtContextKey = `${app.appName}|${app.windowTitle}|${app.url || ''}`;
    block += `Right now the user's ${describeActiveApp(app)} (snapshot from call start — silent refreshes arrive during the call).\n\n`;
  }
  // Prefer a monitor-pushed capture when it's clearly newer than the DB row —
  // pushes arrive before LiteParser+insert finish, so after an app/tab switch
  // the push is often the freshest screen text available at call start.
  const pushAgeMs = _lastScreenCtx
    ? Date.now() - new Date(_lastScreenCtx.capturedAt).getTime()
    : Infinity;
  const usePush = !!_lastScreenCtx && pushAgeMs < 60000 &&
    (!screen || pushAgeMs < (screen.ageMs || 0) - 2000);
  const scr = usePush
    ? { text: _lastScreenCtx.text, appName: _lastScreenCtx.appName, ageMs: pushAgeMs,
        key: `push|${_lastScreenCtx.capturedAt}|${_lastScreenCtx.appName}` }
    : screen && { ...screen, key: `${screen.capturedAt}|${screen.appName}` };
  if (scr) {
    lastRtScreenKey = scr.key;
    block += `Screen content OCR captured ${Math.round((scr.ageMs || 0) / 1000)}s ago on ${scr.appName}: "${scr.text.slice(0, 800)}"\n\n`;
  }
  if (_lastSelectionCtx?.armed || _lastSelectionCtx?.captured) {
    const s = _lastSelectionCtx;
    const what = s.excerpt
      ? `text selected in ${s.sourceApp || 'another app'}: "${s.excerpt}"`
      : `a text selection in ${s.sourceApp || 'another app'}`;
    block += `The user has ${what} — "this/that" questions likely refer to it; run_thinkdrop_task receives the full text automatically.\n\n`;
  }
  if (voiceHistory.length) {
    block += 'Recent conversation with this user:\n' +
      voiceHistory.map(m => `${m.role === 'user' ? 'User' : 'You'}: ${m.text}`).join('\n') + '\n\n';
  }
  try {
    const r = await axios.get(`http://localhost:${COMMS_PORT}/tasks`, { timeout: 800, httpAgent: relayAgent });
    const tasks = (r.data?.tasks || []).slice(-8);
    if (tasks.length) {
      block += 'Recent ThinkDrop tasks (visible to the user):\n' +
        tasks.map(t => `- [${t.status || '?'}] ${(t.prompt || t.sub_prompt || '').slice(0, 80)}`).join('\n');
    }
  } catch (_) {}
  return block.trim() || null;
}

// Silent live-context refresh — pushes a context-only conversation item (the
// model does NOT speak it) whenever the frontmost app/title/URL changes.
// Keeps "did I switch tabs" / "what app am I in" honest without pixel vision.
let rtContextTimer = null;
let lastRtContextKey = null;
let lastRtScreenKey = null;
const RT_CONTEXT_MS = parseInt(process.env.VOICE_RT_CONTEXT_MS || '15000', 10);

function startRtContextRefresh() {
  stopRtContextRefresh();
  rtContextTimer = setInterval(async () => {
    if (sessionMode !== 'realtime' || !rtConnected) return;
    // Frontmost-app change — identity metadata only.
    const app = await fetchActiveAppContext();
    if (app) {
      const key = `${app.appName}|${app.windowTitle}|${app.url || ''}`;
      if (key !== lastRtContextKey) {
        lastRtContextKey = key;
        sendToWorker({ type: 'rt-context', text: `Context update — the user's ${describeActiveApp(app)}.` });
      }
    }
    // Screen-content change — new OCR capture (app switch or scroll) lands
    // as a short digest so "what's this page say" stays honest mid-call.
    const screen = await fetchRecentScreenText();
    if (screen) {
      const key = `${screen.capturedAt}|${screen.appName}`;
      if (key !== lastRtScreenKey) {
        lastRtScreenKey = key;
        sendToWorker({ type: 'rt-context', text: `Screen content update (OCR of ${screen.appName}): "${screen.text.slice(0, 400)}"` });
      }
    }
  }, RT_CONTEXT_MS);
}

function stopRtContextRefresh() {
  if (rtContextTimer) { clearInterval(rtContextTimer); rtContextTimer = null; }
  lastRtContextKey = null;
  lastRtScreenKey = null;
}

function enterRealtime(lang) {
  if (!realtime.isAvailable()) { playFiller('no', lang); return; }
  sessionMode = 'realtime';
  clearTimeout(sleepTimer);            // realtime owns pacing
  clearTimeout(ambientTimer); ambientTimer = null;
  clearTimeout(turnTimer); turnTimer = null; turnBuffer = [];
  playFiller('wake', lang);            // audible "switching" cue
  sendToWorker({ type: 'realtime', active: true });
  relayToMain({ type: 'mode', mode: 'realtime' });
  logger.info('[VoiceBridge] talk mode on (spoken)');
}

/** Route a realtime function call to ThinkDrop main and feed the answer back. */
async function handleToolCall(msg) {
  logger.info('[VoiceBridge] tool call', { name: msg.name });
  const args = JSON.parse(msg.arguments || '{}');
  const r = await axios.post(`http://localhost:${MAIN_PORT}/voice.tool`,
    { name: msg.name, args }, { timeout: 30000, httpAgent: relayAgent });
  const output = r.data?.output || 'Done.';
  sendToWorker({ type: 'rt-tool-result', call_id: msg.call_id, output });
}

// Deterministic backstop for lane-control speech the model answers itself
// ("stop plan mode" narrated without a tool call). main decides whether the
// transcript is a control phrase (same _matchPlanCheckAction as every other
// surface); on a hit we cancel whatever response is generating and inject the
// real outcome as a system note the model announces in its own words.
async function controlCheck(text) {
  const r = await axios.post(`http://localhost:${MAIN_PORT}/voice.control-check`,
    { text }, { timeout: 8000, httpAgent: relayAgent });
  if (!r.data?.handled) return;
  logger.info('[VoiceBridge] control intercept', { text: String(text).slice(0, 80), output: r.data.output });
  sendToWorker({ type: 'rt-cancel' });
  sendToWorker({ type: 'rt-inject', text: `[system] ${r.data.output || 'Done.'}` });
}

/** Warm the worker's decode cache with everything cached for `lang` (+ en). */
function preloadWorkerFillers(lang) {
  const m = fillers.status().manifest;
  const wantLangs = [...new Set([(lang || 'en').split('-')[0], 'en'])];
  const urls = [];
  for (const l of wantLangs) {
    for (const entries of Object.values(m.langs[l] || {})) {
      for (const e of entries.filter(Boolean)) {
        urls.push(`http://127.0.0.1:${PORT}/voice.filler.file?p=${encodeURIComponent(e.file)}`);
      }
    }
  }
  if (urls.length) sendToWorker({ type: 'filler-preload', urls });
}

const driver = new CompanionDriver({
  workerUrl: () => {
    const p = new URLSearchParams();
    // Barge-in is on by default — the worker forwards mid-speech interims
    // tagged duringSpeech; the echo filter decides drop vs. stop+process.
    // VOICE_BARGE_IN=0 restores the old listen-or-talk behavior.
    if (process.env.VOICE_BARGE_IN !== '0') p.set('duplex', '1');
    // VOICE_MIC_DEVICE: label substring pins the S2S mic (e.g. 'MacBook').
    if (process.env.VOICE_MIC_DEVICE) p.set('mic', process.env.VOICE_MIC_DEVICE);
    // VOICE_TTS_SINK: 'auto-headphones' or a deviceId — routes TTS output.
    if (process.env.VOICE_TTS_SINK) p.set('sink', process.env.VOICE_TTS_SINK);
    const qs = p.toString();
    return `http://127.0.0.1:${PORT}/voice/worker${qs ? '?' + qs : ''}`;
  },
  onDisconnect: (reason) => {
    workerReady = false;
    workerSocket = null;
    workerActivated = false;
    clearTimeout(sleepTimer); // no worker, nothing to sleep
    relayToMain({ type: 'state', state: 'disconnected', reason });
  },
});

// ── WS helpers ────────────────────────────────────────────────────────────────
function sendToWorker(msg) {
  if (workerSocket && workerSocket.readyState === 1) {
    try { workerSocket.send(JSON.stringify(msg)); } catch (_) {}
  }
}

// ── Bounded relay to main ─────────────────────────────────────────────────────
// relayToMain is fire-and-forget, but UNBOUNDED posts OOM'd the service: level
// events alone run ~10/sec and thousands of pending requests piled up whenever
// main stalled (~500MB heap in 85s). Keep-alive agent + in-flight cap + a drop
// policy for display-only events keeps the heap flat under congestion.
const relayAgent = new http.Agent({ keepAlive: true, maxSockets: 4 });
const RELAY_CRITICAL = new Set([
  'final', 'final-part', 'state', 'error', 'interrupted', 'speak-done',
  'sleep-state', 'sleep-dropped', 'echo-dropped', 'filler-progress',
  'rt-transcript', 'rt-state', 'rt-tool-call',
  'mode', 'voice-cancel', 'audio-route', 'thought-dropped',
]);
const RELAY_MAX_INFLIGHT = 8;
let relayInFlight = 0;

function relayToMain(payload) {
  // level/interim/speech-event are display-only — drop under congestion.
  if (relayInFlight >= RELAY_MAX_INFLIGHT && !RELAY_CRITICAL.has(payload.type)) return;
  relayInFlight++;
  axios.post(MAIN_EVENT_URL, payload, { timeout: 3000, httpAgent: relayAgent })
    .catch((err) => {
      // level events fire ~10/sec — warn-spam during a main outage drowns
      // the log; keep them at debug.
      const logFn = payload.type === 'level' ? logger.debug : logger.warn;
      logFn('[VoiceBridge] relay to main failed', { error: err.message, type: payload.type });
    })
    .finally(() => { relayInFlight--; });
}

function sendAudioToWorker(buf) {
  if (workerSocket && workerSocket.readyState === 1) {
    try { workerSocket.send(buf); } catch (_) {}
  }
}

/** Forward a worker event to the Electron main process (fire-and-forget). */
// ── WebSocket server (worker page connects here) ──────────────────────────────
const wss = new WebSocketServer({ noServer: true });

wss.on('connection', (ws) => {
  logger.info('[VoiceBridge] worker connected');
  workerSocket = ws;
  ws.on('message', (data, isBinary) => {
    if (isBinary) return; // worker never sends binary upstream
    let msg;
    try { msg = JSON.parse(data.toString()); } catch (_) { return; }
    handleWorkerMessage(msg);
  });
  ws.on('close', () => {
    logger.info('[VoiceBridge] worker disconnected');
    if (workerSocket === ws) { workerSocket = null; workerReady = false; workerActivated = false; }
    relayToMain({ type: 'state', state: 'disconnected', reason: 'ws-close' });
  });
  ws.on('error', () => {});
});

function handleWorkerMessage(msg) {
  // While realtime owns the mic, pipeline SR artifacts must never reach the
  // dispatcher — a leaked 'final' would send the model's own voice to
  // comms-graph as a task. 'level' still flows (voice bars during S2S).
  if (sessionMode === 'realtime' &&
      (msg.type === 'final' || msg.type === 'interim' || msg.type === 'speech-event')) {
    return;
  }
  switch (msg.type) {
    case 'ready':
      workerReady = true;
      logger.info('[VoiceBridge] worker ready', { lang: msg.lang });
      if (sessionWanted) activateWorker();
      relayToMain({ type: 'state', state: 'ready' });
      preloadWorkerFillers(msg.lang);
      break;
    case 'interim': {
      // Duplex mode: interim arrived while our TTS is playing. Similarity to
      // the spoken text decides echo (drop) vs genuine barge-in (stop+relay).
      if (msg.duringSpeech && speechInFlight && lastSpoken) {
        const norm = normalizeText(msg.text);
        const sim = echoSimilarity(norm, lastSpoken.norm);
        if (sim >= ECHO_SIM_THRESHOLD) break; // our own voice — swallow
        // Short interims are too ambiguous to kill playback — unless they're
        // a hard interrupt word ("stop", "wait", "cancel").
        if (norm.length < 10 && !BARGE_SHORT_RE.test(norm)) break;
        logger.info('[VoiceBridge] barge-in speech over TTS', { text: msg.text, sim: sim.toFixed(2) });
        noteSpeechDone();
        sendToWorker({ type: 'stop' });
        relayToMain(msg);
        break;
      }
      {
        const now = Date.now();
        if (now - lastInterimAt > 2500) firstInterimAt = now; // fresh burst of speech
        lastInterimAt = now;
      }
      clearTimeout(ambientTimer); ambientTimer = null; // user is talking — not silence
      // Mid-utterance backchannel: during a long monologue SR produces only
      // interims (no finals until a pause), so without this the "mm-hmm"s
      // the pause-slot logic provides never fire.
      if (!sleeping && sessionMode === 'pipeline' && !speechInFlight) {
        maybeMidSpeechBackchannel(msg.lang);
      }
      // Fresh user speech while holding a turn → keep the turn open.
      if (turnTimer) holdTurn(TURN_HOLD_MS);
      if (!sleeping) armSleepTimer();
      relayToMain(msg);
      break;
    }
    case 'speech-event': {
      // User pause mid-turn (buffered finals still held) is the classic
      // backchannel slot — a soft "mm-hmm" shows we're still tracking.
      if (msg.event === 'speech-end' && turnTimer && turnBuffer.length && !speechInFlight && !sleeping) {
        if (fillerPolicy.decideBackchannel()) playFiller('backchannel', turnLang);
      }
      if (msg.event === 'speech-start') {
        if (turnTimer) holdTurn(TURN_HOLD_MS);
        if (!sleeping) armSleepTimer();
      }
      relayToMain(msg);
      break;
    }
    case 'state':
    case 'level':
      relayToMain(msg);
      break;
    case 'final': {
      // Sleeping: only a wake phrase gets through — everything else is dropped
      // silently (sleep-dropped for main-side observability, no dispatch).
      if (sleeping) {
        // "talk mode" while asleep = wake + switch in one breath.
        if (TALK_CMD.test(msg.text.trim())) { wakeUp('talk-cmd', msg.lang); enterRealtime(msg.lang); break; }
        const w = wakeWord.detect(msg.text);
        if (w.detected && w.type === 'wake') {
          const remainder = wakeWord.stripWakePhrase(msg.text, w.matchedPhrase).trim();
          wakeUp('wake-phrase', msg.lang);
          if (remainder.length > 2 && remainder !== msg.text) {
            // "hey thinkdrop, what's the weather" — wake AND carry the prompt
            relayToMain({ type: 'final-part', text: remainder, lang: msg.lang });
            turnBuffer.push(remainder);
            turnLang = msg.lang || turnLang;
            holdTurn(TURN_HOLD_MS);
          }
        } else {
          relayToMain({ type: 'sleep-dropped', text: msg.text });
        }
        break;
      }
      const reason = echoOrDupeReason(msg);
      if (reason) {
        logger.info('[VoiceBridge] dropping transcript', { reason, text: msg.text });
        relayToMain({ type: 'echo-dropped', text: msg.text, reason });
        break;
      }
      armSleepTimer(); // genuine user activity
      firstInterimAt = 0; // turn boundary — next monologue re-arms the 3s wind-up
      lastFinal = { norm: normalizeText(msg.text), at: Date.now() };
      // Spoken cancel kills the running task — don't dispatch it as a prompt.
      if (CANCEL_CMD.test(msg.text.trim())) { voiceCancel(msg.lang); break; }
      // Live transcript for the overlay — display only, no dispatch.
      relayToMain({ type: 'final-part', text: msg.text, lang: msg.lang });
      // Immediate human-feel ack from the cached voice library.
      const decision = fillerPolicy.decideFinal(msg.text);
      if (decision.filler) playFiller(decision.filler, msg.lang);
      // Explicit sleep command — don't dispatch it as a task.
      if (decision.cls === 'sleep') { enterSleep('sleep-command'); break; }
      // Spoken switch into Talk Mode — mid-session upgrade.
      if (sessionMode !== 'realtime' && TALK_CMD.test(msg.text.trim())) { enterRealtime(msg.lang); break; }
      // Aggregate: merge rapid mid-chain finals into ONE dispatch.
      turnBuffer.push(msg.text);
      turnLang = msg.lang || turnLang;
      holdTurn(decision.holdTurn ? TURN_HOLD_MS * 1.6 : TURN_HOLD_MS);
      break;
    }
    case 'speak-done':
    case 'interrupted':
      noteSpeechDone();
      relayToMain(msg);
      break;
    case 'rt-state':
      if (msg.state === 'connected') {
        clearTimeout(rtWatchdog); rtWatchdog = null;
        rtConnected = true;
        sessionMode = 'realtime';
        clearTimeout(sleepTimer); // realtime owns pacing
        startRtContextRefresh();  // keep the frontmost-app snapshot live
        relayToMain({ type: 'mode', mode: 'realtime' });
      } else if (msg.state === 'failed' && sessionWanted && sessionMode === 'realtime') {
        fallbackToPipeline('rt-failed');
      } else if (msg.state === 'disconnected' && !rtConnected && sessionWanted && sessionMode === 'realtime') {
        // ICE 'disconnected' is often transient mid-connect — only fall back
        // if we never got 'connected' in the first place.
        fallbackToPipeline('rt-never-connected');
      }
      relayToMain(msg);
      break;
    case 'rt-transcript': {
      noteVoice(msg.role === 'user' ? 'user' : 'assistant', msg.text);
      relayToMain(msg);
      // Control backstop — when the model answers a lane-control phrase
      // conversationally (no tool call), the transcript still performs the
      // real transition and the model is told the outcome.
      if (msg.role === 'user' && sessionMode === 'realtime' && msg.text && msg.text.trim()) {
        controlCheck(msg.text).catch(() => {});
      }
      break;
    }
    case 'rt-tool-call': {
      // Model called run_thinkdrop_task → route through main, feed result back.
      handleToolCall(msg).catch(err => {
        logger.warn('[VoiceBridge] tool call failed', { error: err.message });
        sendToWorker({ type: 'rt-tool-result', call_id: msg.call_id, output: `Error: ${err.message}` });
      });
      break;
    }
    case 'error':
      // S2S didn't connect — keep the session alive on the pipeline path.
      if (msg.error === 'realtime-start-failed') fallbackToPipeline(msg.detail || 'start-failed');
      // A failed/decode-erroring TTS must not leave the echo filter wedged.
      if (speechInFlight) noteSpeechDone();
      relayToMain(msg);
      break;
    case 'pong':
      break;
    default:
      relayToMain(msg);
  }
}

// ── Session control ───────────────────────────────────────────────────────────
// mode 'auto' (the default) prefers S2S and falls back to the pipeline when
// realtime can't connect — one button, best available path.
let rtWatchdog = null;
let rtConnected = false; // true once the WebRTC path actually connected
const RT_CONNECT_TIMEOUT_MS = parseInt(process.env.VOICE_RT_CONNECT_MS || '12000', 10);

function fallbackToPipeline(reason) {
  if (sessionMode !== 'realtime' || !sessionWanted) return;
  clearTimeout(rtWatchdog); rtWatchdog = null;
  stopRtContextRefresh();
  logger.warn('[VoiceBridge] realtime unavailable — falling back to pipeline', { reason });
  sessionMode = 'pipeline';
  sendToWorker({ type: 'realtime', active: false }); // worker restores SR
  relayToMain({ type: 'mode', mode: 'pipeline' });
  armSleepTimer();
  fillers.ensureLibrary(p => relayToMain({ type: 'filler-progress', ...p }));
}

async function startSession(mode = 'auto') {
  sessionWanted = true;
  rtConnected = false;
  sessionMode = (mode === 'realtime' || (mode === 'auto' && realtime.isAvailable()))
    ? 'realtime' : 'pipeline';
  relayToMain({ type: 'state', state: 'starting' }); // driver launch takes seconds
  relayToMain({ type: 'mode', mode: sessionMode });
  sleeping = false;
  fillerPolicy.reset();
  firstInterimAt = 0; lastInterimAt = 0;
  if (sessionMode === 'pipeline') armSleepTimer(); // Talk Mode owns its pacing
  wakeWord.detect('warmup'); // build the Fuse index now, not mid-sleep-gate
  const res = await driver.start();
  // If the worker is already connected (browser resident), activate now;
  // otherwise activation happens on the 'ready' handshake.
  if (workerReady) activateWorker();
  // Filler library runs in the background regardless of mode — auto sessions
  // can fall back to pipeline, and realtime exits re-enter it.
  fillers.ensureLibrary(p => relayToMain({ type: 'filler-progress', ...p }));
  if (workerReady && sessionMode === 'pipeline') preloadWorkerFillers();
  return res;
}

function activateWorker() {
  if (workerActivated) return; // ready-handshake + startSession can race — activate once
  workerActivated = true;
  sendToWorker({ type: 'session', active: true });
  if (sessionMode === 'realtime') {
    sendToWorker({ type: 'realtime', active: true });
    // If the WebRTC path never connects, don't leave the user hanging.
    clearTimeout(rtWatchdog);
    rtWatchdog = setTimeout(() => fallbackToPipeline('connect-timeout'), RT_CONNECT_TIMEOUT_MS);
  }
}

async function stopSession() {
  sessionWanted = false;
  workerActivated = false;
  if (sleeping) { sleeping = false; relayToMain({ type: 'sleep-state', sleeping: false, source: 'session-stop' }); }
  clearTimeout(sleepTimer);
  clearTimeout(rtWatchdog); rtWatchdog = null;
  stopRtContextRefresh();
  clearTimeout(ambientTimer); ambientTimer = null;
  clearTimeout(pendingThoughtTimer); pendingThoughtTimer = null;
  flushTurn(); // don't strand a held mid-turn utterance
  sendToWorker({ type: 'session', active: false });
  if (sessionMode === 'realtime') sendToWorker({ type: 'realtime', active: false });
  // Toggle-off means really off — quit the hidden Chrome entirely.
  // VOICE_KEEP_RESIDENT=1 keeps the old fast-re-entry behavior for debugging.
  if (process.env.VOICE_KEEP_RESIDENT !== '1') {
    try { await driver.close(); } catch (_) {}
  }
}

// ── TTS ───────────────────────────────────────────────────────────────────────
async function say({ text, lang }) {
  if (!text || !text.trim()) throw new Error('text required');
  // In Talk Mode the realtime model owns the speaker — a stray TTS here would
  // double-talk over the live call.
  if (sessionMode === 'realtime') return { ok: true, spoken: false, reason: 'realtime-mode' };
  try {
    const result = await voiceProvider.synthesize({ text, language: lang });
    const format = result.format || 'mp3';
    noteSpeaking(text);
    noteVoice('assistant', text);
    sendToWorker({ type: 'audio-begin', format });
    sendAudioToWorker(result.audioBuffer);
    sendToWorker({ type: 'audio-end' });
    return { ok: true, provider: voiceProvider.getProviderName(), format, bytes: result.audioBuffer.length };
  } catch (err) {
    // All quality providers failed → text-only (spoken:false). The robotic
    // speechSynthesis path only remains as an explicit debug escape hatch.
    if (process.env.VOICE_ALLOW_SYSTEM_VOICE === '1') {
      logger.warn('[VoiceBridge] synth failed — speechSynthesis fallback (debug)', { error: err.message });
      noteSpeaking(text);
      sendToWorker({ type: 'speak', text, lang });
      return { ok: true, provider: 'speechSynthesis', fallback: true };
    }
    logger.warn('[VoiceBridge] synth failed — response stays text-only', { error: err.message });
    return { ok: true, spoken: false, reason: 'tts-unavailable' };
  }
}

// ── HTTP server ───────────────────────────────────────────────────────────────
const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,POST,OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }

  const json = (code, obj) => {
    res.writeHead(code, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(obj));
  };
  const readBody = () => new Promise((resolve) => {
    const chunks = [];
    req.on('data', c => chunks.push(c));
    req.on('end', () => {
      try { resolve(JSON.parse(Buffer.concat(chunks).toString() || '{}')); }
      catch (_) { resolve({}); }
    });
  });

  try {
    // Worker page
    if (req.method === 'GET' && url.pathname === '/voice/worker') {
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      res.end(fs.readFileSync(WORKER_HTML));
      return;
    }

    if (req.method === 'GET' && url.pathname === '/health') {
      return json(200, {
        ok: true,
        service: 'voice-bridge',
        session: sessionWanted,
        workerReady,
        driver: driver.status(),
        provider: voiceProvider.getProviderName(),
      });
    }

    if (req.method === 'GET' && url.pathname === '/voice.session.status') {
      return json(200, {
        active: sessionWanted,
        sleeping,
        workerReady,
        driver: driver.status(),
        provider: voiceProvider.getProviderName(),
      });
    }

    if (req.method === 'POST' && url.pathname === '/voice.session.start') {
      const body = await readBody();
      const result = await startSession(body.mode);
      return json(200, { ok: true, mode: sessionMode, ...result });
    }

    if (req.method === 'POST' && url.pathname === '/voice.session.stop') {
      await stopSession();
      return json(200, { ok: true });
    }

    if (req.method === 'POST' && url.pathname === '/voice.say') {
      const body = await readBody();
      const result = await say(body);
      return json(200, result);
    }

    if (req.method === 'POST' && url.pathname === '/voice.stop') {
      noteSpeechDone();
      sendToWorker({ type: 'stop' });
      return json(200, { ok: true });
    }

    if (req.method === 'POST' && url.pathname === '/voice.sleep') {
      enterSleep('manual');
      return json(200, { ok: true, sleeping });
    }

    if (req.method === 'POST' && url.pathname === '/voice.wake') {
      wakeUp('manual');
      return json(200, { ok: true, sleeping });
    }

    // ── Realtime S2S — SDP broker; the API key never reaches the worker ───────
    if (req.method === 'POST' && url.pathname === '/voice.realtime.start') {
      if (!realtime.isAvailable()) return json(503, { error: 'realtime unavailable (OPENAI_API_KEY)' });
      const body = await readBody();
      if (!body.sdp) return json(400, { error: 'sdp required' });
      try {
        const ctx = await buildRealtimeContext();
        const { sdp, callId } = await realtime.createCall(body.sdp, ctx);
        return json(200, { sdp, callId });
      } catch (err) {
        logger.error('[VoiceBridge] realtime start failed', { error: err.message });
        return json(502, { error: err.message });
      }
    }

    if (req.method === 'POST' && url.pathname === '/voice.realtime.stop') {
      sendToWorker({ type: 'realtime', active: false });
      return json(200, { ok: true });
    }

    // ── Selection context — main pushes arm/capture/disarm so the live call ──
    // knows a highlighted-text selection exists ("what is this" refers to it).
    if (req.method === 'POST' && url.pathname === '/voice.context') {
      const body = await readBody();
      _lastSelectionCtx = {
        armed: !!body.armed,
        captured: !!body.captured,
        sourceApp: body.sourceApp || null,
        excerpt: body.excerpt ? String(body.excerpt).slice(0, 120) : null,
      };
      logger.info('[VoiceBridge] selection context', {
        armed: _lastSelectionCtx.armed, captured: _lastSelectionCtx.captured,
        app: _lastSelectionCtx.sourceApp,
      });
      if (sessionMode === 'realtime' && rtConnected) {
        const s = _lastSelectionCtx;
        const text = (s.armed || s.captured)
          ? `Context update — the user has ${s.excerpt ? `text selected in ${s.sourceApp || 'another app'}: "${s.excerpt}"` : `a text selection in ${s.sourceApp || 'another app'}`}. "This/that" questions likely refer to it.`
          : 'Context update — the user no longer has a text selection.';
        sendToWorker({ type: 'rt-context', text });
      }
      return json(200, { ok: true });
    }

    // ── Screen OCR push — the user-memory monitor sends fresh captures the ──
    // moment tesseract finishes (before LiteParser/DB-store), so a live call
    // sees screen changes within seconds instead of at the next rt poll.
    if (req.method === 'POST' && url.pathname === '/voice.screen') {
      const body = await readBody();
      _lastScreenCtx = {
        appName: body.appName || null,
        windowTitle: body.windowTitle || '',
        url: body.url || null,
        text: String(body.text || '').slice(0, 800),
        capturedAt: body.capturedAt || new Date().toISOString(),
      };
      logger.info('[VoiceBridge] screen context push', {
        app: _lastScreenCtx.appName, len: _lastScreenCtx.text.length,
      });
      // Keep the poller's key in sync so it doesn't re-push the same capture.
      lastRtScreenKey = `push|${_lastScreenCtx.capturedAt}|${_lastScreenCtx.appName}`;
      if (sessionMode === 'realtime' && rtConnected && _lastScreenCtx.text) {
        sendToWorker({ type: 'rt-context', text: `Screen content update (OCR of ${_lastScreenCtx.appName}): "${_lastScreenCtx.text.slice(0, 400)}"` });
      }
      return json(200, { ok: true });
    }

    // ── Async task result → the live call announces it ─────────────────────
    // main.js POSTs here when a voice-sourced task completes while the
    // session is in realtime mode (pipeline mode uses /voice.say instead).
    if (req.method === 'POST' && url.pathname === '/voice.task-result') {
      const body = await readBody();
      const status = body.status || 'done';
      const prompt = String(body.prompt || '').slice(0, 120);
      const answer = String(body.answer || body.error || '').slice(0, 800);
      if (sessionMode === 'realtime') {
        const text = status === 'done'
          ? `[Task completed] "${prompt}" finished. Result: ${answer || '(no output)'}. Tell the user the outcome briefly and naturally.`
          : `[Task ${status}] "${prompt}" ended with status ${status}${answer ? `: ${answer}` : ''}. Let the user know briefly.`;
        sendToWorker({ type: 'rt-inject', text });
        return json(200, { ok: true, delivered: 'realtime' });
      }
      return json(200, { ok: true, delivered: 'none', reason: 'not-realtime' });
    }

    // ── Proactive thoughts — heartbeat/synthesis-agent speak through here ──
    if (req.method === 'POST' && url.pathname === '/voice.speak') {
      const body = await readBody();
      const result = await speakThought({ text: body.text, lang: body.lang || body.language });
      return json(200, result);
    }

    // ── Filler library ────────────────────────────────────────────────────
    if (req.method === 'GET' && url.pathname === '/voice.filler.file') {
      const rel = url.searchParams.get('p') || '';
      const abs = fillers.resolveRel(rel);
      if (!abs || !fs.existsSync(abs)) return json(404, { error: 'filler not found' });
      const ext = path.extname(abs).slice(1).toLowerCase();
      const mime = { mp3: 'audio/mpeg', wav: 'audio/wav', opus: 'audio/ogg', ogg: 'audio/ogg' }[ext] || 'application/octet-stream';
      res.writeHead(200, { 'Content-Type': mime, 'Cache-Control': 'public, max-age=86400' });
      fs.createReadStream(abs).pipe(res);
      return;
    }

    if (req.method === 'GET' && url.pathname === '/voice.fillers.status') {
      return json(200, fillers.status());
    }

    if (req.method === 'POST' && url.pathname === '/voice.fillers.regen') {
      fillers.ensureLibrary(p => relayToMain({ type: 'filler-progress', ...p }));
      return json(200, { ok: true, ...fillers.status() });
    }

    if (req.method === 'POST' && url.pathname === '/voice.provider') {
      const body = await readBody();
      if (body.action === 'set' && body.provider) {
        const r = voiceProvider.setProvider(body.provider);
        // New voice → new manifest namespace → background regen kicks off.
        fillers.ensureLibrary(p => relayToMain({ type: 'filler-progress', ...p }));
        return json(200, r);
      }
      return json(200, { providers: voiceProvider.listProviders() });
    }

    json(404, { error: 'not found', path: url.pathname });
  } catch (err) {
    logger.error('[VoiceBridge] request failed', { error: err.message, path: url.pathname });
    json(500, { error: err.message });
  }
});

// WS upgrade — only /voice.ws
server.on('upgrade', (req, socket, head) => {
  if (req.url && req.url.startsWith('/voice.ws')) {
    wss.handleUpgrade(req, socket, head, (ws) => wss.emit('connection', ws, req));
  } else {
    socket.destroy();
  }
});

server.listen(PORT, () => {
  logger.info(`[VoiceBridge] listening on :${PORT} (worker: http://127.0.0.1:${PORT}/voice/worker)`);
});

// Graceful shutdown — close the hidden Chrome with the service.
async function shutdown() {
  try { await driver.close(); } catch (_) {}
  process.exit(0);
}
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
