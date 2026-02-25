/**
 * Voice Journal — ~/.thinkdrop/voice-state.json
 *
 * Shared state file between:
 *   - StateGraph (writes progress, reads signals)
 *   - Voice service (reads progress, writes signals and voice queue)
 *
 * Schema:
 * {
 *   stategraph: {
 *     status: 'idle' | 'running' | 'done' | 'error' | 'paused',
 *     intent: string,
 *     currentNode: string,
 *     nodeIndex: number,
 *     totalNodes: number,
 *     startedAt: ISO string,
 *     lastUpdate: ISO string,
 *     summary: string,
 *     traceSteps: Array<{ node, status, ms }>,
 *     sessionId: string,
 *   },
 *   signals: Array<{
 *     id: string,         // sig_<6hex>
 *     type: 'cancel' | 'pause' | 'resume' | 'inject',
 *     payload: any,       // for inject: { message: string }
 *     ts: ISO string,
 *     status: 'pending' | 'acknowledged' | 'done' | 'error',
 *   }>,
 *   voiceQueue: Array<{
 *     id: string,
 *     text: string,       // translated English text ready for StateGraph
 *     originalText: string,
 *     language: string,
 *     ts: ISO string,
 *     status: 'pending' | 'processing' | 'done',
 *   }>,
 *   voice: {
 *     status: 'idle' | 'listening' | 'processing' | 'speaking',
 *     lastSpokenAt: ISO string,
 *     activationMode: 'push-to-talk' | 'wake-word' | 'continuous',
 *     detectedLanguage: string,
 *   }
 * }
 */

'use strict';

const fs = require('fs');
const path = require('path');
const os = require('os');
const logger = require('./logger.cjs');

const JOURNAL_DIR = path.join(os.homedir(), '.thinkdrop');
const JOURNAL_PATH = process.env.VOICE_JOURNAL_PATH
  ? process.env.VOICE_JOURNAL_PATH.replace('~', os.homedir())
  : path.join(JOURNAL_DIR, 'voice-state.json');

const DEFAULT_STATE = {
  stategraph: {
    status: 'idle',
    intent: null,
    currentNode: null,
    nodeIndex: 0,
    totalNodes: 0,
    startedAt: null,
    lastUpdate: null,
    summary: '',
    traceSteps: [],
    sessionId: null,
  },
  signals: [],
  voiceQueue: [],
  voice: {
    status: 'idle',
    lastSpokenAt: null,
    activationMode: 'wake-word',
    detectedLanguage: 'en',
  },
};

function _ensureDir() {
  if (!fs.existsSync(JOURNAL_DIR)) {
    fs.mkdirSync(JOURNAL_DIR, { recursive: true });
  }
}

/**
 * Read the current journal state. Returns DEFAULT_STATE if file missing/corrupt.
 */
function read() {
  try {
    _ensureDir();
    if (!fs.existsSync(JOURNAL_PATH)) {
      return JSON.parse(JSON.stringify(DEFAULT_STATE));
    }
    const raw = fs.readFileSync(JOURNAL_PATH, 'utf8');
    return JSON.parse(raw);
  } catch (err) {
    logger.warn('[Journal] Failed to read, using default state', { error: err.message });
    return JSON.parse(JSON.stringify(DEFAULT_STATE));
  }
}

/**
 * Write the full state object to disk atomically.
 * Uses write-to-temp + rename to avoid partial reads.
 */
function write(state) {
  try {
    _ensureDir();
    const tmp = JOURNAL_PATH + '.tmp';
    fs.writeFileSync(tmp, JSON.stringify(state, null, 2), 'utf8');
    fs.renameSync(tmp, JOURNAL_PATH);
  } catch (err) {
    logger.error('[Journal] Write failed', { error: err.message });
  }
}

/**
 * Patch only specific top-level keys.
 */
function patch(updates) {
  const state = read();
  const next = { ...state };
  for (const [key, val] of Object.entries(updates)) {
    if (val && typeof val === 'object' && !Array.isArray(val)) {
      next[key] = { ...state[key], ...val };
    } else {
      next[key] = val;
    }
  }
  write(next);
  return next;
}

// ─── StateGraph-side helpers ──────────────────────────────────────────────────

/**
 * Called by StateGraph when a new run starts.
 */
function graphStarted({ intent, sessionId }) {
  patch({
    stategraph: {
      status: 'running',
      intent,
      sessionId,
      currentNode: null,
      nodeIndex: 0,
      totalNodes: 0,
      startedAt: new Date().toISOString(),
      lastUpdate: new Date().toISOString(),
      summary: `Starting ${intent}...`,
      traceSteps: [],
    },
  });
}

/**
 * Called by StateGraph after each node completes.
 */
function graphNodeDone({ node, nodeIndex, totalNodes, durationMs, status = 'done' }) {
  const state = read();
  const traceSteps = [...(state.stategraph.traceSteps || [])];
  traceSteps.push({ node, status, ms: durationMs });

  patch({
    stategraph: {
      ...state.stategraph,
      status: 'running',
      currentNode: node,
      nodeIndex,
      totalNodes,
      lastUpdate: new Date().toISOString(),
      summary: `Running ${node} (step ${nodeIndex} of ${totalNodes})`,
      traceSteps: traceSteps.slice(-30),
    },
  });
}

/**
 * Called by StateGraph when the run completes.
 */
function graphDone({ intent, summary = '' }) {
  patch({
    stategraph: {
      status: 'done',
      intent,
      currentNode: null,
      lastUpdate: new Date().toISOString(),
      summary: summary || `${intent} completed`,
    },
  });
}

/**
 * Called by StateGraph when the run errors.
 */
function graphError({ intent, error }) {
  patch({
    stategraph: {
      status: 'error',
      intent,
      currentNode: null,
      lastUpdate: new Date().toISOString(),
      summary: `Error in ${intent}: ${error}`,
    },
  });
}

// ─── Voice-side helpers ───────────────────────────────────────────────────────

/**
 * Write a signal for StateGraph to act on.
 * @param {'cancel'|'pause'|'resume'|'inject'} type
 * @param {any} payload
 */
function writeSignal(type, payload = {}) {
  const state = read();
  const id = `sig_${require('crypto').randomBytes(3).toString('hex')}`;
  const signal = {
    id,
    type,
    payload,
    ts: new Date().toISOString(),
    status: 'pending',
  };
  const signals = [...(state.signals || []), signal];
  patch({ signals });
  logger.info('[Journal] Signal written', { id, type });
  return id;
}

/**
 * Read pending signals. Used by StateGraph's poll loop.
 */
function readPendingSignals() {
  const state = read();
  return (state.signals || []).filter(s => s.status === 'pending');
}

/**
 * Acknowledge a signal (StateGraph marks it done).
 */
function acknowledgeSignal(signalId, status = 'done') {
  const state = read();
  const signals = (state.signals || []).map(s =>
    s.id === signalId ? { ...s, status } : s
  );
  const cleaned = signals.filter(s => s.status === 'pending' || Date.now() - new Date(s.ts).getTime() < 60000);
  patch({ signals: cleaned });
}

/**
 * Update voice status (for UI and status queries).
 */
function setVoiceStatus(status, extra = {}) {
  const state = read();
  patch({
    voice: {
      ...state.voice,
      status,
      ...extra,
    },
  });
}

/**
 * Get a human-readable status summary (for voice responses to status queries).
 */
function getStatusSummary() {
  const state = read();
  const sg = state.stategraph;

  if (!sg || sg.status === 'idle') {
    return 'ThinkDrop is idle, ready for your next command.';
  }

  if (sg.status === 'running') {
    const steps = sg.totalNodes > 0 ? ` (step ${sg.nodeIndex} of ${sg.totalNodes})` : '';
    return `Currently running ${sg.intent || 'a task'}${steps}. ${sg.summary || ''}`.trim();
  }

  if (sg.status === 'done') {
    return `Last task (${sg.intent || 'task'}) completed. ${sg.summary || ''}`.trim();
  }

  if (sg.status === 'error') {
    return `Last task encountered an error. ${sg.summary || ''}`.trim();
  }

  if (sg.status === 'paused') {
    return `Task paused at step ${sg.nodeIndex}. Say "resume" to continue.`;
  }

  return 'Status unknown.';
}

/**
 * Watch the journal for changes (uses fs.watch).
 * @param {Function} onChange - Called with (state) when file changes
 * @returns {Function} unwatch — call to stop watching
 */
function watch(onChange) {
  _ensureDir();
  let debounceTimer = null;

  const watcher = fs.watch(JOURNAL_PATH, (eventType) => {
    if (eventType === 'change') {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        try {
          const state = read();
          onChange(state);
        } catch (_) {}
      }, 100);
    }
  });

  return () => {
    clearTimeout(debounceTimer);
    watcher.close();
  };
}

module.exports = {
  read,
  write,
  patch,
  graphStarted,
  graphNodeDone,
  graphDone,
  graphError,
  writeSignal,
  readPendingSignals,
  acknowledgeSignal,
  setVoiceStatus,
  getStatusSummary,
  watch,
  JOURNAL_PATH,
};
