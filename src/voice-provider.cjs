'use strict';

/**
 * Voice Provider Switcher
 *
 * Selects and manages the active voice provider based on VOICE_PROVIDER env var.
 * Provides automatic fallback when a provider is unavailable.
 *
 * Provider chain (in fallback order):
 *   1. hume      — Hume EVI (speech-to-speech, most human-sounding, ~$0.06-0.07/min)
 *   2. inworld — Inworld/ElevenLabs TTS + Groq Whisper STT (current default)
 *   3. groq      — Groq Whisper STT + macOS say TTS (nearly free)
 *   4. macos     — macOS say TTS only (offline, zero cost, robotic)
 *
 * Configuration:
 *   VOICE_PROVIDER=hume|cartesia|inworld|groq|macos
 *   If not set, auto-selects based on available API keys (hume → inworld → groq → macos)
 *
 * Usage:
 *   const provider = require('./voice-provider.cjs');
 *
 *   // Standard STT+TTS providers (inworld, groq, macos):
 *   const sttResult = await provider.transcribe({ audioBase64, format });
 *   const ttsResult = await provider.synthesize({ text, language });
 *
 *   // Hume EVI — handled transparently via processAudio() in server.cjs;
 *   // the persistent WebSocket is managed internally by hume-evi.cjs.
 */

const logger = require('./logger.cjs');

// ── Provider modules ──────────────────────────────────────────────────────────
const humeEvi       = require('./providers/hume-evi.cjs');
const inworld       = require('./providers/inworld.cjs');
const cartesia      = require('./providers/cartesia.cjs');
const groqProvider  = require('./providers/groq.cjs');
const macosNative   = require('./providers/macos-native.cjs');

// ── Fallback chain ────────────────────────────────────────────────────────────
const FALLBACK_CHAIN = ['cartesia', 'inworld', 'groq', 'macos', 'hume'];

const PROVIDER_MAP = {
  hume:       humeEvi,
  inworld:    inworld,
  cartesia:   cartesia,
  groq:       groqProvider,
  macos:      macosNative,
};

// ── Active provider selection ─────────────────────────────────────────────────

let _activeProviderName = null;
let _activeProvider     = null;

function _selectProvider() {
  const requested = (process.env.VOICE_PROVIDER || '').toLowerCase().trim();

  if (requested && PROVIDER_MAP[requested]) {
    const p = PROVIDER_MAP[requested];
    if (p.isAvailable()) {
      logger.info(`[VoiceProvider] Using requested provider: ${requested}`);
      return { name: requested, provider: p };
    }
    logger.warn(`[VoiceProvider] Requested provider "${requested}" is unavailable — falling back`);
  }

  // Auto-select: walk the fallback chain
  for (const name of FALLBACK_CHAIN) {
    const p = PROVIDER_MAP[name];
    if (p && p.isAvailable()) {
      logger.info(`[VoiceProvider] Auto-selected provider: ${name}`);
      return { name, provider: p };
    }
  }

  // Should never reach here — macOS is always available on macOS
  logger.error('[VoiceProvider] No voice provider available!');
  return { name: 'macos', provider: macosNative };
}

function _ensureProvider() {
  if (!_activeProvider) {
    const { name, provider } = _selectProvider();
    _activeProviderName = name;
    _activeProvider     = provider;
  }
  return { name: _activeProviderName, provider: _activeProvider };
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Get the name of the currently active provider.
 * @returns {'hume'|'inworld'|'groq'|'macos'}
 */
function getProviderName() {
  return _ensureProvider().name;
}

/**
 * Returns true if the active provider is Hume EVI (speech-to-speech mode).
 * In EVI mode, the normal STT→LLM→TTS pipeline is replaced by EVI's WebSocket.
 */
function isHumeEVI() {
  return getProviderName() === 'hume';
}

/**
 * Force-switch to a specific provider at runtime.
 * Useful for the user toggling providers without restarting.
 * @param {'hume'|'cartesia'|'inworld'|'groq'|'macos'} name
 */
function setProvider(name) {
  const norm = (name || '').toLowerCase().trim();
  if (!PROVIDER_MAP[norm]) {
    throw new Error(`Unknown voice provider: ${name}. Valid: ${Object.keys(PROVIDER_MAP).join(', ')}`);
  }
  // Close the persistent EVI session if we're switching away from hume
  if (_activeProviderName === 'hume' && norm !== 'hume') {
    try { humeEvi.closePersistentSession(); } catch (_) {}
  }
  _activeProviderName = norm;
  _activeProvider     = PROVIDER_MAP[norm];
  logger.info(`[VoiceProvider] Switched to: ${norm}`);
  return { name: norm, available: _activeProvider.isAvailable() };
}

/**
 * List all providers and their availability status.
 */
function listProviders() {
  return FALLBACK_CHAIN.map(name => ({
    name,
    active: name === getProviderName(),
    available: PROVIDER_MAP[name]?.isAvailable() ?? false,
  }));
}

// ── STT (standard providers only) ────────────────────────────────────────────

/**
 * Transcribe audio to text.
 * NOT used in Hume EVI mode — EVI handles STT internally.
 *
 * Tries the active provider, falls back down the chain on error.
 *
 * @param {{ audioBase64: string, format?: string, languageHint?: string }} args
 * @returns {Promise<{ text: string, language: string, confidence: number, isFinal: boolean }>}
 */
async function transcribe(args) {
  const chain = _buildFallbackChain('transcribe');
  let lastError;
  for (const { name, provider } of chain) {
    if (!provider.transcribe) continue;
    try {
      logger.info(`[VoiceProvider] STT via ${name}`);
      return await provider.transcribe(args);
    } catch (err) {
      logger.warn(`[VoiceProvider] STT failed on ${name} — trying next`, { error: err.message });
      lastError = err;
    }
  }
  throw lastError || new Error('All STT providers failed');
}

// ── TTS (standard providers only) ────────────────────────────────────────────

/**
 * Synthesize text to speech.
 * NOT used in Hume EVI mode — EVI handles TTS internally.
 *
 * Tries the active provider, falls back down the chain on error.
 *
 * @param {{ text: string, language?: string, voiceId?: string }} args
 * @returns {Promise<{ audioBuffer: Buffer, format: string, durationEstimateMs: number }>}
 */
async function synthesize(args) {
  const chain = _buildFallbackChain('synthesize');
  let lastError;
  for (const { name, provider } of chain) {
    if (!provider.synthesize) continue;
    try {
      logger.info(`[VoiceProvider] TTS via ${name}`);
      return await provider.synthesize(args);
    } catch (err) {
      logger.warn(`[VoiceProvider] TTS failed on ${name} — trying next`, { error: err.message });
      lastError = err;
    }
  }
  throw lastError || new Error('All TTS providers failed');
}

// ── Hume EVI session API ──────────────────────────────────────────────────────

/**
 * Send audio to Hume EVI's persistent session and wait for the response.
 * Only valid when isHumeEVI() === true.
 */
async function eviProcessAudio(args) {
  if (!isHumeEVI()) throw new Error('eviProcessAudio() called but active provider is not Hume EVI');
  return humeEvi.processAudio(args);
}

// ── Internal helpers ──────────────────────────────────────────────────────────

function _buildFallbackChain(capability) {
  const { name: activeName } = _ensureProvider();
  // Start from active provider, then walk remaining fallback chain
  const activeIdx = FALLBACK_CHAIN.indexOf(activeName);
  const orderedNames = [
    ...FALLBACK_CHAIN.slice(activeIdx),
    ...FALLBACK_CHAIN.slice(0, activeIdx),
  ];
  return orderedNames
    .map(name => ({ name, provider: PROVIDER_MAP[name] }))
    .filter(({ provider }) => provider && provider.isAvailable());
}

// ── Init log ──────────────────────────────────────────────────────────────────
// Log available providers on module load
setTimeout(() => {
  const all = listProviders();
  logger.info('[VoiceProvider] Provider status on startup', {
    active: getProviderName(),
    available: all.filter(p => p.available).map(p => p.name),
    unavailable: all.filter(p => !p.available).map(p => p.name),
  });
}, 0);

module.exports = {
  getProviderName,
  isHumeEVI,
  setProvider,
  listProviders,
  transcribe,
  synthesize,
  eviProcessAudio,
};
