'use strict';

/**
 * fillers.cjs — Pre-generated, voice-matched filler library.
 *
 * Humans respond in ~0–200ms gaps; live TTS synthesis (~700ms+) can never hit
 * that window. So we batch-generate a categorized phrase set ONCE through the
 * active provider+voice, cache it on disk, and replay with ~0 latency over
 * the same worker audio path — same voice, same channel.
 *
 *   data/fillers/<provider>__<voice>[__<model>]/
 *     manifest.json   — { voiceKey, format, langs: { en: { cat: [{file,text}] } } }
 *     <lang>/<cat>/<i>.<ext>
 *
 * Text pools: fillerPhrases.json (conversational categories) +
 * comms-graph/src/handoffPhrases.json (intent_* acks — already multilingual).
 *
 * Generation is BACKGROUND: ensureLibrary() returns immediately and fills in
 * via a small concurrency pool; progressCb(done,total) drives a GhostLayer pill.
 */

const fs = require('fs');
const path = require('path');
const logger = require('./logger.cjs');
const voiceProvider = require('./voice-provider.cjs');

const FILLER_ROOT = path.join(__dirname, '..', 'data', 'fillers');
const PHRASES_PATH = path.join(__dirname, 'fillerPhrases.json');
const HANDOFF_PATH = path.join(__dirname, '..', '..', '..', 'comms-graph', 'src', 'handoffPhrases.json');

const LANGS = ['en', 'zh', 'es', 'fr', 'pt', 'ar', 'ja', 'ko', 'hi', 'de', 'it', 'ru'];
const GEN_CONCURRENCY = 4;

// Intent categories whose text comes from comms-graph's multilingual pool.
const INTENT_FROM_HANDOFF = {
  intent_web_search: 'web_search',
  intent_memory_retrieve: 'memory_retrieve',
  intent_screen_analysis: 'screen_analysis',
  intent_general_handoff: 'general_handoff',
};

// ── Phrase pool assembly ──────────────────────────────────────────────────────
let _pools = null;
function getPools() {
  if (_pools) return _pools;
  const base = JSON.parse(fs.readFileSync(PHRASES_PATH, 'utf8'));
  let handoff = {};
  try { handoff = JSON.parse(fs.readFileSync(HANDOFF_PATH, 'utf8')); } catch (e) {
    logger.warn('[Fillers] handoffPhrases.json unavailable', { error: e.message });
  }
  _pools = { ...base };
  for (const [fillerCat, handoffKey] of Object.entries(INTENT_FROM_HANDOFF)) {
    _pools[fillerCat] = handoff[handoffKey] || {};
  }
  return _pools;
}

// ── Voice key: manifest namespace per provider+voice+model ────────────────────
function getVoiceKey() {
  const provider = voiceProvider.getProviderName();
  const voice = process.env.OPENAI_TTS_VOICE
    || process.env.CARTESIA_VOICE_ID
    || process.env.RESEMBLE_VOICE_UUID
    || provider;
  const model = process.env.OPENAI_TTS_MODEL
    || process.env.CARTESIA_MODEL_ID
    || process.env.RESEMBLE_MODEL
    || '';
  // FS-safe key — voice ids can contain characters we don't want in paths.
  const raw = `${provider}:${voice}${model ? ':' + model : ''}`;
  return raw.replace(/[^\w.-]+/g, '_');
}

function libDir() { return path.join(FILLER_ROOT, getVoiceKey()); }
function manifestPath() { return path.join(libDir(), 'manifest.json'); }

function loadManifest() {
  try { return JSON.parse(fs.readFileSync(manifestPath(), 'utf8')); }
  catch (_) { return { voiceKey: getVoiceKey(), format: null, langs: {} }; }
}
function saveManifest(m) {
  fs.mkdirSync(libDir(), { recursive: true });
  fs.writeFileSync(manifestPath(), JSON.stringify(m, null, 1));
}

// ── Generation ────────────────────────────────────────────────────────────────
let _generating = false;
let _progress = { done: 0, total: 0, voiceKey: null };

/** Build the work list: {lang, cat, idx, text, rel} for every missing file. */
function buildWorklist(manifest) {
  const pools = getPools();
  const jobs = [];
  for (const lang of LANGS) {
    for (const [cat, byLang] of Object.entries(pools)) {
      const phrases = byLang[lang];
      if (!phrases || !phrases.length) continue;
      const existing = manifest.langs[lang]?.[cat] || [];
      phrases.forEach((text, idx) => {
        if (!existing[idx]) jobs.push({ lang, cat, idx, text });
      });
    }
  }
  return jobs;
}

/**
 * Ensure the filler library for the current voice exists. Returns instantly;
 * generation runs in the background with a concurrency pool. Safe to call
 * repeatedly — regenerates only missing files.
 */
function ensureLibrary(onProgress) {
  const manifest = loadManifest();
  const jobs = buildWorklist(manifest);
  _progress = { done: 0, total: jobs.length, voiceKey: manifest.voiceKey };
  if (!jobs.length) { onProgress && onProgress({ done: 0, total: 0, complete: true, voiceKey: manifest.voiceKey }); return; }
  if (_generating) return; // a regen for a different voice restarts on next call
  _generating = true;

  logger.info('[Fillers] generating library', { voiceKey: manifest.voiceKey, files: jobs.length });
  let i = 0;
  const tick = () => onProgress && onProgress({ ..._progress, voiceKey: manifest.voiceKey });

  const worker = async () => {
    while (i < jobs.length) {
      const job = jobs[i++];
      try {
        const result = await voiceProvider.synthesize({ text: job.text, language: job.lang });
        const ext = (result.format || 'mp3').replace(/\W/g, '');
        manifest.format = manifest.format || result.format || 'mp3';
        const rel = path.join(job.lang, job.cat, `${job.idx}.${ext}`);
        const abs = path.join(libDir(), rel);
        fs.mkdirSync(path.dirname(abs), { recursive: true });
        fs.writeFileSync(abs, result.audioBuffer);
        (manifest.langs[job.lang] ||= {})[job.cat] ||= [];
        manifest.langs[job.lang][job.cat][job.idx] = { file: rel, text: job.text };
      } catch (err) {
        // Never synthesize fillers through a robotic fallback — just mark absent.
        logger.warn('[Fillers] gen failed', { cat: job.cat, lang: job.lang, idx: job.idx, error: err.message });
      }
      _progress.done++;
      if (_progress.done % 5 === 0 || _progress.done === jobs.length) { saveManifest(manifest); tick(); }
    }
  };

  Promise.all(Array.from({ length: GEN_CONCURRENCY }, worker))
    .then(() => saveManifest(manifest))
    .catch(() => saveManifest(manifest))
    .finally(() => {
      _generating = false;
      logger.info('[Fillers] library ready', { voiceKey: manifest.voiceKey, generated: _progress.done });
      onProgress && onProgress({ ..._progress, complete: true, voiceKey: manifest.voiceKey });
    });
}

// ── Selection ─────────────────────────────────────────────────────────────────
const _lastIdx = {};

/**
 * Pick a cached filler clip. Rotation avoids immediate repeats.
 * Falls back to 'ack', then English, then null (caller decides).
 * @returns {{rel:string, text:string}|null}
 */
function pick(category, lang = 'en') {
  const manifest = loadManifest();
  const langs = manifest.langs;
  const l = (lang || 'en').split('-')[0].toLowerCase();
  const list = langs[l]?.[category]
    || langs[l]?.ack
    || langs.en?.[category]
    || langs.en?.ack
    || null;
  if (!list || !list.length) return null;
  const valid = list.filter(Boolean);
  if (!valid.length) return null;
  const key = `${l}:${category}`;
  let idx, last = _lastIdx[key] ?? -1;
  do { idx = Math.floor(Math.random() * valid.length); } while (valid.length > 1 && idx === last);
  _lastIdx[key] = idx;
  return { rel: valid[idx].file, text: valid[idx].text, category, lang: l };
}

function status() {
  const m = loadManifest();
  return { ..._progress, generating: _generating, voiceKey: getVoiceKey(), manifest: m };
}

/** Resolve a manifest rel path → absolute path under FILLER_ROOT (sanitized). */
function resolveRel(rel) {
  const abs = path.resolve(libDir(), rel);
  return abs.startsWith(libDir() + path.sep) ? abs : null;
}

module.exports = { ensureLibrary, pick, status, resolveRel, getVoiceKey, getPools, libDir };
