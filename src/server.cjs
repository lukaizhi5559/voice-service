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

require('dotenv').config({ path: path.join(__dirname, '..', '.env') });

const PORT = parseInt(process.env.PORT || '3006', 10);
const MAIN_PORT = parseInt(process.env.THINKDROP_MAIN_PORT || '3010', 10);
const MAIN_EVENT_URL = `http://localhost:${MAIN_PORT}/voice.event`;
const WORKER_HTML = path.join(__dirname, '..', 'public', 'voice-worker.html');

// ── Session / worker state ────────────────────────────────────────────────────
let sessionWanted = false;
let workerSocket = null;   // active WS connection from the worker page
let workerReady = false;

const driver = new CompanionDriver({
  workerUrl: () => `http://127.0.0.1:${PORT}/voice/worker`,
  onDisconnect: (reason) => {
    workerReady = false;
    workerSocket = null;
    relayToMain({ type: 'state', state: 'disconnected', reason });
  },
});

// ── WS helpers ────────────────────────────────────────────────────────────────
function sendToWorker(msg) {
  if (workerSocket && workerSocket.readyState === 1) {
    try { workerSocket.send(JSON.stringify(msg)); } catch (_) {}
  }
}

function sendAudioToWorker(buf) {
  if (workerSocket && workerSocket.readyState === 1) {
    try { workerSocket.send(buf); } catch (_) {}
  }
}

/** Forward a worker event to the Electron main process (fire-and-forget). */
function relayToMain(payload) {
  axios.post(MAIN_EVENT_URL, payload, { timeout: 3000 }).catch((err) => {
    logger.warn('[VoiceBridge] relay to main failed', { error: err.message, type: payload.type });
  });
}

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
    if (workerSocket === ws) { workerSocket = null; workerReady = false; }
    relayToMain({ type: 'state', state: 'disconnected', reason: 'ws-close' });
  });
  ws.on('error', () => {});
});

function handleWorkerMessage(msg) {
  switch (msg.type) {
    case 'ready':
      workerReady = true;
      logger.info('[VoiceBridge] worker ready', { lang: msg.lang });
      if (sessionWanted) sendToWorker({ type: 'session', active: true });
      relayToMain({ type: 'state', state: 'ready' });
      break;
    case 'interim':
    case 'final':
    case 'state':
    case 'level':
    case 'error':
    case 'interrupted':
    case 'speak-done':
      relayToMain(msg);
      break;
    case 'pong':
      break;
    default:
      relayToMain(msg);
  }
}

// ── Session control ───────────────────────────────────────────────────────────
async function startSession() {
  sessionWanted = true;
  const res = await driver.start();
  // If the worker is already connected (browser resident), activate now;
  // otherwise activation happens on the 'ready' handshake.
  if (workerReady) sendToWorker({ type: 'session', active: true });
  return res;
}

async function stopSession() {
  sessionWanted = false;
  sendToWorker({ type: 'session', active: false });
}

// ── TTS ───────────────────────────────────────────────────────────────────────
async function say({ text, lang }) {
  if (!text || !text.trim()) throw new Error('text required');
  try {
    const result = await voiceProvider.synthesize({ text, language: lang });
    const format = result.format || 'mp3';
    sendToWorker({ type: 'audio-begin', format });
    sendAudioToWorker(result.audioBuffer);
    sendToWorker({ type: 'audio-end' });
    return { ok: true, provider: voiceProvider.getProviderName(), format, bytes: result.audioBuffer.length };
  } catch (err) {
    // All synth providers failed → tell the worker to use speechSynthesis.
    logger.warn('[VoiceBridge] synth failed — speechSynthesis fallback', { error: err.message });
    sendToWorker({ type: 'speak', text, lang });
    return { ok: true, provider: 'speechSynthesis', fallback: true };
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
        workerReady,
        driver: driver.status(),
        provider: voiceProvider.getProviderName(),
      });
    }

    if (req.method === 'POST' && url.pathname === '/voice.session.start') {
      const result = await startSession();
      return json(200, { ok: true, ...result });
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
      sendToWorker({ type: 'stop' });
      return json(200, { ok: true });
    }

    if (req.method === 'POST' && url.pathname === '/voice.provider') {
      const body = await readBody();
      if (body.action === 'set' && body.provider) {
        return json(200, voiceProvider.setProvider(body.provider));
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
