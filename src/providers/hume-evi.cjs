'use strict';

/**
 * Hume EVI voice provider
 *
 * Connects to Hume's Empathic Voice Interface (EVI) via WebSocket.
 * EVI handles STT + prosody-based end-of-turn + emotional TTS internally.
 *
 * Integration pattern:
 *   - VoiceButton streams raw PCM/WebM audio chunks to this provider
 *   - Provider forwards audio to EVI over WebSocket as base64 audio_input messages
 *   - EVI fires tool_call messages when it detects an action intent
 *     → tool_call triggers ThinkDrop's StateGraph (same as current fast-lane escalation)
 *   - EVI fires assistant_message + audio_output messages for chitchat/questions
 *     → audio_output is forwarded back to the renderer as voice:response
 *
 * CLM (Custom Language Model) is NOT used here — we use EVI's built-in LLM
 * supplemented with a tool_call ("trigger_stategraph") for action intents.
 * This is simpler and more reliable than hosting our own CLM server.
 *
 * Required env vars:
 *   HUME_API_KEY    — from app.hume.ai/developers
 *   HUME_API_SECRET — from app.hume.ai/developers
 *   HUME_CONFIG_ID  — optional, EVI configuration ID (uses default config if omitted)
 */

const WebSocket = require('ws');
const https = require('https');
const http = require('http');
const logger = require('../logger.cjs');

const HUME_API_KEY    = process.env.HUME_API_KEY || '';
const HUME_API_SECRET = process.env.HUME_API_SECRET || '';
const HUME_CONFIG_ID  = process.env.HUME_CONFIG_ID || '';
const THINKDROP_MAIN_PORT = parseInt(process.env.THINKDROP_MAIN_PORT || '3010', 10);

// EVI WebSocket endpoint
const EVI_WS_BASE = 'wss://api.hume.ai/v0/evi/chat';

// ---------------------------------------------------------------------------
// Access token (short-lived JWT) — needed for WebSocket auth
// ---------------------------------------------------------------------------

let _cachedToken = null;
let _tokenExpiry  = 0;

async function getAccessToken() {
  // Reuse if still valid (with 60s buffer)
  if (_cachedToken && Date.now() < _tokenExpiry - 60_000) return _cachedToken;

  return new Promise((resolve, reject) => {
    const credentials = Buffer.from(`${HUME_API_KEY}:${HUME_API_SECRET}`).toString('base64');
    const body = 'grant_type=client_credentials';

    const req = https.request({
      hostname: 'api.hume.ai',
      port: 443,
      path: '/oauth2-cc/token',
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Authorization': `Basic ${credentials}`,
        'Content-Length': Buffer.byteLength(body),
      },
    }, (res) => {
      let data = '';
      res.on('data', chunk => { data += chunk; });
      res.on('end', () => {
        try {
          const parsed = JSON.parse(data);
          if (!parsed.access_token) return reject(new Error(`Hume token error: ${data.slice(0, 200)}`));
          _cachedToken = parsed.access_token;
          _tokenExpiry = Date.now() + (parsed.expires_in || 3600) * 1000;
          logger.info('[HumeEVI] Access token refreshed');
          resolve(_cachedToken);
        } catch (e) {
          reject(new Error(`Hume token parse error: ${e.message}`));
        }
      });
    });

    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

// ---------------------------------------------------------------------------
// Persistent Session Manager
// One EVI WebSocket is kept open across many audio chunks / voice activations.
// An idle timer closes it after IDLE_CLOSE_MS of no audio — EVI will greet
// again on the next connection which is the correct UX after a long pause.
// ---------------------------------------------------------------------------

const IDLE_CLOSE_MS = 45_000; // close EVI session after 45s of no audio
const TURN_TIMEOUT_MS = 60_000; // max ms to wait for assistant_end per turn

// The single persistent session — module-level singleton
let _persistentSession = null;

class HumeEVISession {
  constructor() {
    this.ws           = null;
    this.closed       = false;
    this._idleTimer   = null;

    // ── Serial turn queue ────────────────────────────────────────────────────
    // Each entry: { audioBase64, onStateGraphTrigger, resolve, turnId }
    // Only one turn is active at a time. The rest wait in _queue.
    this._queue       = [];   // pending turns
    this._activeTurn  = null; // currently in-flight turn or null

    // Per-turn accumulator (owned by the active turn)
    this._turnId              = 0;   // monotonic counter — incremented per turn
    this._activeTurnId        = null;
    this._responseAudioChunks = [];
    this._responseLane        = 'fast';
    this._responseText        = '';
    this._userTranscript      = '';
    this._turnTimeoutHandle   = null;
  }

  async connect() {
    if (!HUME_API_KEY) throw new Error('HUME_API_KEY not set');

    let token;
    try {
      token = await getAccessToken();
    } catch (err) {
      logger.warn('[HumeEVI] Token fetch failed — using API key directly', { error: err.message });
      token = null;
    }

    const authParam = token
      ? `access_token=${encodeURIComponent(token)}`
      : `api_key=${encodeURIComponent(HUME_API_KEY)}`;
    const configParam = HUME_CONFIG_ID ? `&config_id=${encodeURIComponent(HUME_CONFIG_ID)}` : '';
    const url = `${EVI_WS_BASE}?${authParam}${configParam}`;

    logger.info('[HumeEVI] Connecting persistent EVI WebSocket');
    this.ws = new WebSocket(url);
    this.ws.on('message', (data) => this._handleMessage(data));
    this.ws.on('error', (err) => {
      logger.error('[HumeEVI] WebSocket error', { error: err.message });
    });
    this.ws.on('close', (code, reason) => {
      logger.info('[HumeEVI] WebSocket closed', { code, reason: reason?.toString() });
      this.closed = true;
      this._clearIdleTimer();
      if (_persistentSession === this) _persistentSession = null;
      this._resolveAllPending();
    });

    await new Promise((resolve, reject) => {
      const t = setTimeout(() => reject(new Error('EVI connect timeout')), 10000);
      this.ws.once('open', () => { clearTimeout(t); resolve(); });
      this.ws.once('error', (err) => { clearTimeout(t); reject(err); });
    });
    logger.info('[HumeEVI] Persistent WebSocket connected');
  }

  // Enqueue an audio chunk. Returns a Promise that resolves with EVI's reply.
  // Turns execute serially — concurrent calls wait until the previous resolves.
  sendAudio(audioBase64, callbacks) {
    if (this.closed) return Promise.reject(new Error('EVI session is closed'));

    return new Promise((resolve) => {
      this._queue.push({ audioBase64, callbacks, resolve });
      this._drainQueue();
    });
  }

  // Process the next queued turn if no turn is currently active.
  _drainQueue() {
    if (this._activeTurn || this._queue.length === 0) return;
    if (this.closed || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
      // Drain all queued turns with empty result
      while (this._queue.length) {
        const { resolve } = this._queue.shift();
        resolve({ audioBase64: null, audioFormat: 'mp3', transcript: '', responseText: '', lane: 'fast', durationEstimateMs: 0 });
      }
      return;
    }

    const turn = this._queue.shift();
    this._activeTurn = turn;

    // Assign a unique ID to this turn so stale assistant_end can't resolve it
    const myTurnId = ++this._turnId;
    this._activeTurnId        = myTurnId;
    this._responseAudioChunks = [];
    this._responseLane        = 'fast';
    this._responseText        = '';
    this._userTranscript      = '';

    // Per-turn timeout
    this._turnTimeoutHandle = setTimeout(() => {
      if (this._activeTurnId !== myTurnId) return;
      logger.warn('[HumeEVI] Turn timeout — resolving with collected audio', { turnId: myTurnId });
      this._resolveTurn(myTurnId);
    }, TURN_TIMEOUT_MS);

    // Reset idle timer on each new turn
    this._resetIdleTimer();

    this.ws.send(JSON.stringify({ type: 'audio_input', data: turn.audioBase64 }));
    logger.info('[HumeEVI] Turn started', { turnId: myTurnId, queueDepth: this._queue.length });
  }

  // Resolve the active turn and start the next one.
  _resolveTurn(turnId) {
    if (this._activeTurnId !== turnId) return; // stale — ignore

    clearTimeout(this._turnTimeoutHandle);
    this._turnTimeoutHandle = null;

    const turn = this._activeTurn;
    this._activeTurn    = null;
    this._activeTurnId  = null;

    const combinedAudio = this._responseAudioChunks.length > 0
      ? Buffer.concat(this._responseAudioChunks).toString('base64')
      : null;
    const totalBytes = this._responseAudioChunks.reduce((s, b) => s + b.length, 0);

    turn.resolve({
      audioBase64:        combinedAudio,
      audioFormat:        'mp3',
      transcript:         this._userTranscript,
      responseText:       this._responseText,
      lane:               this._responseLane,
      durationEstimateMs: totalBytes ? Math.round((totalBytes / 16000) * 1000) : 0,
    });

    // Process the next queued turn
    setImmediate(() => this._drainQueue());
  }

  close() {
    this._clearIdleTimer();
    if (this.ws && !this.closed) {
      try { this.ws.close(); } catch (_) {}
    }
    this.closed = true;
    if (_persistentSession === this) _persistentSession = null;
  }

  _resetIdleTimer() {
    this._clearIdleTimer();
    this._idleTimer = setTimeout(() => {
      logger.info('[HumeEVI] Idle timeout — closing persistent session');
      this.close();
    }, IDLE_CLOSE_MS);
  }

  _clearIdleTimer() {
    if (this._idleTimer) { clearTimeout(this._idleTimer); this._idleTimer = null; }
  }

  // Drain all pending turns with empty result (used on close)
  _resolveAllPending() {
    if (this._activeTurn) {
      this._resolveTurn(this._activeTurnId);
    }
    while (this._queue.length) {
      const { resolve } = this._queue.shift();
      resolve({ audioBase64: null, audioFormat: 'mp3', transcript: '', responseText: '', lane: 'fast', durationEstimateMs: 0 });
    }
  }

  // ── Message handler ────────────────────────────────────────────────────────

  async _handleMessage(raw) {
    let msg;
    try { msg = JSON.parse(raw.toString()); } catch (_) { return; }

    const type = msg.type;
    switch (type) {
      case 'user_message':
        this._userTranscript = msg.message?.content || '';
        logger.info('[HumeEVI] user_message', { text: this._userTranscript.substring(0, 80) });
        break;

      case 'assistant_message':
        this._responseText = msg.message?.content || '';
        logger.info('[HumeEVI] assistant_message', { text: this._responseText.substring(0, 80) });
        break;

      case 'audio_output':
        if (msg.data) {
          this._responseAudioChunks.push(Buffer.from(msg.data, 'base64'));
        }
        break;

      case 'assistant_end':
        logger.info('[HumeEVI] assistant_end', { activeTurnId: this._activeTurnId });
        this._resetIdleTimer();
        if (this._activeTurnId !== null) {
          this._resolveTurn(this._activeTurnId);
        }
        break;

      case 'tool_call':
        await this._handleToolCall(msg);
        break;

      case 'tool_error':
        logger.warn('[HumeEVI] tool_error', { error: msg.error });
        this._sendToolResponse(msg.tool_call_id, 'I encountered an error processing that request.');
        break;

      case 'error':
        logger.error('[HumeEVI] EVI error', { msg });
        if (msg.code === 'E0300' || msg.slug === 'zero_credits') {
          this.close();
        } else if (this._activeTurnId !== null) {
          this._resolveTurn(this._activeTurnId);
        }
        break;

      case 'chat_metadata':
        logger.info('[HumeEVI] chat_metadata', { chatId: msg.chat_id });
        break;

      default:
        break;
    }
  }

  async _handleToolCall(msg) {
    const toolCallId = msg.tool_call_id;
    const toolName   = msg.name;
    logger.info('[HumeEVI] tool_call received', { toolName, toolCallId });

    if (toolName === 'trigger_stategraph') {
      let userText = this._userTranscript;
      try {
        const params = JSON.parse(msg.parameters || '{}');
        if (params.user_intent) userText = params.user_intent;
      } catch (_) {}

      logger.info('[HumeEVI] Triggering StateGraph', { userText: userText.substring(0, 80) });

      // Mark lane + respond to EVI immediately so it can speak the holding phrase
      this._responseLane = 'stategraph';
      this._sendToolResponse(toolCallId, 'On it — working on that for you now.');

      const sgCb = this._activeTurn?.callbacks?.onStateGraphTrigger;
      if (sgCb) {
        sgCb(userText).catch(err => {
          logger.error('[HumeEVI] StateGraph trigger error', { error: err.message });
        });
      }
    } else {
      this._sendToolResponse(toolCallId, 'Done.');
    }
  }

  _sendToolResponse(toolCallId, content) {
    if (this.closed || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    this.ws.send(JSON.stringify({ type: 'tool_response', tool_call_id: toolCallId, content }));
  }
}

// ---------------------------------------------------------------------------
// EVI System Prompt + Tool Definition
// These are sent once when creating/updating the EVI configuration.
// In practice you set these in the Hume dashboard (app.hume.ai/evi/build)
// and reference the config_id via HUME_CONFIG_ID env var.
// This is exposed here for documentation / programmatic config creation.
// ---------------------------------------------------------------------------

const EVI_SYSTEM_PROMPT = `You are ThinkDrop, a smart personal AI assistant that lives on the user's desktop.
You can answer questions, have natural conversations, and perform actions on the user's computer.

When the user asks you to DO something on their computer — open apps, search the web, save files,
automate tasks, send emails, etc. — call the trigger_stategraph tool with their exact request.

For everything else (questions, chitchat, explanations) — just answer naturally and conversationally.
Keep responses concise. No markdown. Max 2 sentences for simple answers.`;

const EVI_TOOL_DEFINITION = {
  type: 'function',
  name: 'trigger_stategraph',
  description: 'Trigger ThinkDrop\'s automation engine to perform an action on the user\'s computer. Use this whenever the user asks you to DO something — open apps, search, save files, send messages, etc.',
  parameters: JSON.stringify({
    type: 'object',
    properties: {
      user_intent: {
        type: 'string',
        description: 'The user\'s request, verbatim or paraphrased clearly.',
      },
    },
    required: ['user_intent'],
  }),
};

// ---------------------------------------------------------------------------
// Public API — called by voice-provider.cjs
// ---------------------------------------------------------------------------

/**
 * Check if the Hume EVI provider is available (API key set).
 */
function isAvailable() {
  return !!(HUME_API_KEY && HUME_API_SECRET);
}

/**
 * Get or create the persistent EVI session.
 * Reuses the existing WebSocket if still open; creates a new one otherwise.
 */
async function _getOrCreateSession() {
  if (_persistentSession && !_persistentSession.closed) {
    return _persistentSession;
  }
  const session = new HumeEVISession();
  await session.connect();
  _persistentSession = session;
  return session;
}

/**
 * Send one audio chunk to the persistent EVI session and wait for EVI's reply.
 *
 * This is the main entry point from server.cjs _processWithHumeEVI.
 * The persistent session stays open across calls — EVI maintains full
 * conversation context including memory of previous turns.
 *
 * Returns:
 *   { audioBase64, audioFormat, transcript, responseText, lane, durationEstimateMs }
 */
async function processAudio({ audioBase64, onStateGraphTrigger }) {
  let session;
  try {
    session = await _getOrCreateSession();
  } catch (err) {
    throw new Error(`HumeEVI connect failed: ${err.message}`);
  }

  // Enqueue this turn. The queue ensures serial execution — only one turn
  // is active at a time. Per-turn timeout (60s) is managed inside _drainQueue.
  return session.sendAudio(audioBase64, {
    onStateGraphTrigger: onStateGraphTrigger || null,
  });
}

/**
 * Force-close the persistent session (e.g. on provider switch).
 */
function closePersistentSession() {
  if (_persistentSession) {
    _persistentSession.close();
    _persistentSession = null;
  }
}

module.exports = {
  isAvailable,
  processAudio,
  closePersistentSession,
  EVI_SYSTEM_PROMPT,
  EVI_TOOL_DEFINITION,
};
