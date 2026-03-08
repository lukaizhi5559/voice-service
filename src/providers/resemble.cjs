'use strict';

/**
 * Resemble AI TTS Provider
 *
 * Uses Resemble AI's Chatterbox model (streaming synthesis endpoint).
 * Supports emotion and paralinguistic tags inline in text:
 *
 *   Vocalizations: [laugh] [sigh] [gasp] [crying]
 *   Emotions:      [happy] [angry] [sad] [excited] [empathetic]
 *   SSML:          <break time="500ms"/> <prosody pitch="high" rate="fast">
 *
 * The LLM is instructed to inject these tags into its responses based on
 * detected user emotion (from audio energy / tone analysis).
 *
 * API endpoints:
 *   Synthesis:  https://f.cluster.resemble.ai/synthesize
 *   Streaming:  https://f.cluster.resemble.ai/stream
 */

const https = require('https');
const logger = require('../logger.cjs');

const RESEMBLE_API_KEY  = process.env.RESEMBLE_API_KEY || '';
const RESEMBLE_VOICE_UUID = process.env.RESEMBLE_VOICE_UUID || '';
const RESEMBLE_PROJECT_UUID = process.env.RESEMBLE_PROJECT_UUID || '';
const RESEMBLE_SYNTHESIZE_URL = 'https://f.cluster.resemble.ai/synthesize';
const RESEMBLE_STREAM_URL = 'https://f.cluster.resemble.ai/stream';

// Per-language voice overrides — set e.g. RESEMBLE_VOICE_UUID_ZH=<chinese-voice-uuid>
// Falls back to RESEMBLE_VOICE_UUID if the language-specific var is not set.
const RESEMBLE_VOICE_MAP = {
  zh: process.env.RESEMBLE_VOICE_UUID_ZH || '',
  ja: process.env.RESEMBLE_VOICE_UUID_JA || '',
  ko: process.env.RESEMBLE_VOICE_UUID_KO || '',
  es: process.env.RESEMBLE_VOICE_UUID_ES || '',
  fr: process.env.RESEMBLE_VOICE_UUID_FR || '',
  de: process.env.RESEMBLE_VOICE_UUID_DE || '',
  pt: process.env.RESEMBLE_VOICE_UUID_PT || '',
  ar: process.env.RESEMBLE_VOICE_UUID_AR || '',
};

/**
 * Resolve the best voice UUID for a given language code.
 * Uses language-specific UUID if configured, otherwise falls back to default.
 */
function _resolveVoice(language, explicitVoiceId) {
  if (explicitVoiceId) return explicitVoiceId;
  const lang = (language || 'en').toLowerCase().substring(0, 2);
  return RESEMBLE_VOICE_MAP[lang] || RESEMBLE_VOICE_UUID;
}

/**
 * Check if Resemble is configured and available.
 */
function isAvailable() {
  return !!(RESEMBLE_API_KEY && RESEMBLE_VOICE_UUID);
}

/**
 * Emotion-state tags ([happy],[sad],[angry],[excited],[empathetic],[whispering],[shouting])
 * are NOT supported as inline text in the Resemble API — they are a Hub UI feature.
 * Only vocalizations are supported: [laugh][sigh][gasp][um][crying].
 * Strip emotion-state tags before sending; keep vocalizations.
 */
function _prepareText(text) {
  return text
    .replace(/\[(happy|sad|angry|excited|empathetic|whispering|shouting)\]/gi, '')
    .replace(/\s{2,}/g, ' ')
    .trim();
}

/**
 * Map emotion intensity (0–1) to Resemble exaggeration (0.3–0.8).
 * Neutral → 0.5, high intensity → 0.75.
 */
function _intensityToExaggeration(intensity) {
  if (!intensity || intensity === 0) return 0.5;
  return Math.min(0.8, 0.45 + (intensity * 0.35));
}

/**
 * Synthesize text to speech using Resemble AI (synchronous /synthesize endpoint).
 * Vocalizations ([laugh],[sigh],[gasp],[um]) are passed through — chatterbox-turbo processes them.
 * Emotion-state tags are stripped (not supported in API text — use exaggeration param instead).
 *
 * @param {{ text: string, language?: string, voiceId?: string, exaggeration?: number }} args
 * @returns {Promise<{ audioBuffer: Buffer, format: string, durationEstimateMs: number }>}
 */
async function synthesize({ text, language = 'en', voiceId, exaggeration } = {}) {
  if (!isAvailable()) throw new Error('Resemble AI not configured (missing RESEMBLE_API_KEY or RESEMBLE_VOICE_UUID)');

  const voice = _resolveVoice(language, voiceId);

  logger.info('[Resemble] Synthesizing TTS', {
    textLen: text.length,
    voice,
    language,
    hasVocalizations: /\[(laugh|sigh|gasp|um|crying)\]/i.test(text),
  });

  const cleanText = _prepareText(text);
  const exag = exaggeration != null ? exaggeration : 0.5;

  const bodyObj = {
    voice_uuid: voice,
    data: cleanText,
    sample_rate: 44100,
    output_format: 'wav',
    model: 'chatterbox-turbo',
    exaggeration: exag,
  };

  if (RESEMBLE_PROJECT_UUID) {
    bodyObj.project_uuid = RESEMBLE_PROJECT_UUID;
  }

  const bodyStr = JSON.stringify(bodyObj);
  const url = new URL(RESEMBLE_SYNTHESIZE_URL);

  const responseBuffer = await new Promise((resolve, reject) => {
    const chunks = [];
    const options = {
      hostname: url.hostname,
      path: url.pathname,
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${RESEMBLE_API_KEY}`,
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(bodyStr),
      },
    };

    const req = https.request(options, (res) => {
      if (res.statusCode !== 200) {
        let errBody = '';
        res.on('data', c => { errBody += c; });
        res.on('end', () => reject(new Error(`Resemble API error ${res.statusCode}: ${errBody.substring(0, 200)}`)));
        return;
      }
      res.on('data', chunk => chunks.push(chunk));
      res.on('end', () => resolve(Buffer.concat(chunks)));
    });

    req.on('error', reject);
    req.setTimeout(20000, () => req.destroy(new Error('Resemble TTS timeout')));
    req.write(bodyStr);
    req.end();
  });

  // Resemble /synthesize always returns JSON:
  //   { success: true, audio_content: "<base64-encoded WAV>", ... }
  // Parse and decode audio_content directly — no separate URL fetch needed.
  let json;
  try {
    json = JSON.parse(responseBuffer.toString('utf8'));
  } catch (_) {
    // Unexpected: raw binary response — treat as WAV directly
    const durationEstimateMs = _estimateDuration(responseBuffer, 'wav');
    logger.info('[Resemble] TTS complete (raw binary fallback)', { bytes: responseBuffer.length, durationEstimateMs });
    return { audioBuffer: responseBuffer, format: 'wav', durationEstimateMs };
  }

  if (!json.success) {
    throw new Error(`Resemble API error: ${JSON.stringify(json).substring(0, 200)}`);
  }

  // Primary path: audio_content is base64-encoded WAV
  if (json.audio_content) {
    const audioBuffer = Buffer.from(json.audio_content, 'base64');
    const durationMs = json.duration ? Math.ceil(json.duration * 1000) : _estimateDuration(audioBuffer, 'wav');
    logger.info('[Resemble] TTS complete', { bytes: audioBuffer.length, durationMs, model: 'chatterbox-turbo' });
    return { audioBuffer, format: 'wav', durationEstimateMs: durationMs };
  }

  // Legacy fallback: item.audio_src URL (older API versions)
  if (json.item?.audio_src) {
    const fetchedBuffer = await _fetchAudioUrl(json.item.audio_src);
    const durationEstimateMs = _estimateDuration(fetchedBuffer, 'wav');
    logger.info('[Resemble] TTS complete (URL fallback)', { bytes: fetchedBuffer.length, durationEstimateMs });
    return { audioBuffer: fetchedBuffer, format: 'wav', durationEstimateMs };
  }

  throw new Error(`Resemble API returned unexpected shape: ${JSON.stringify(json).substring(0, 200)}`);
}

/**
 * Fetch audio from a URL (used when Resemble returns an async URL).
 * @param {string} audioUrl
 * @returns {Promise<Buffer>}
 */
function _fetchAudioUrl(audioUrl) {
  return new Promise((resolve, reject) => {
    const url = new URL(audioUrl);
    const mod = url.protocol === 'https:' ? https : require('http');
    const chunks = [];
    const req = mod.get(audioUrl, (res) => {
      if (res.statusCode !== 200) {
        reject(new Error(`Failed to fetch audio URL: ${res.statusCode}`));
        return;
      }
      res.on('data', c => chunks.push(c));
      res.on('end', () => resolve(Buffer.concat(chunks)));
    });
    req.on('error', reject);
    req.setTimeout(15000, () => req.destroy(new Error('Audio URL fetch timeout')));
  });
}

/**
 * Rough duration estimate from buffer size.
 * MP3 at 128kbps ≈ 16000 bytes/s; 44100Hz ≈ 192kbps ≈ 24000 bytes/s.
 */
function _estimateDuration(buffer, format) {
  if (!buffer) return 0;
  const bytesPerSecond = format === 'mp3' ? 20000 : 88200;
  return Math.ceil((buffer.length / bytesPerSecond) * 1000);
}

/**
 * Resemble does not do STT — always throw so the fallback chain skips it.
 */
async function transcribe() {
  throw new Error('Resemble AI does not support STT');
}

/**
 * Streaming synthesis via /stream endpoint — returns chunked WAV.
 * Collects all chunks and resolves with full buffer for compatibility.
 * Use this for lower first-audio latency.
 *
 * @param {{ text: string, voiceId?: string, exaggeration?: number }} args
 * @returns {Promise<{ audioBuffer: Buffer, format: string, durationEstimateMs: number }>}
 */
async function synthesizeStream({ text, voiceId, exaggeration } = {}) {
  if (!isAvailable()) throw new Error('Resemble AI not configured');

  const voice = voiceId || RESEMBLE_VOICE_UUID;
  const cleanText = _prepareText(text);
  const exag = exaggeration != null ? exaggeration : 0.5;

  const bodyObj = {
    voice_uuid: voice,
    data: cleanText,
    model: 'chatterbox-turbo',
    exaggeration: exag,
  };

  const bodyStr = JSON.stringify(bodyObj);
  const url = new URL(RESEMBLE_STREAM_URL);

  logger.info('[Resemble] Streaming TTS', { textLen: cleanText.length, voice, exag });

  const audioBuffer = await new Promise((resolve, reject) => {
    const chunks = [];
    const options = {
      hostname: url.hostname,
      path: url.pathname,
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${RESEMBLE_API_KEY}`,
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(bodyStr),
      },
    };

    const req = https.request(options, (res) => {
      if (res.statusCode !== 200) {
        let errBody = '';
        res.on('data', c => { errBody += c; });
        res.on('end', () => reject(new Error(`Resemble stream error ${res.statusCode}: ${errBody.substring(0, 200)}`)));
        return;
      }
      res.on('data', chunk => chunks.push(chunk));
      res.on('end', () => resolve(Buffer.concat(chunks)));
    });

    req.on('error', reject);
    req.setTimeout(20000, () => req.destroy(new Error('Resemble stream timeout')));
    req.write(bodyStr);
    req.end();
  });

  const durationEstimateMs = _estimateDuration(audioBuffer, 'wav');
  logger.info('[Resemble] Stream TTS complete', { bytes: audioBuffer.length, durationEstimateMs });
  return { audioBuffer, format: 'wav', durationEstimateMs };
}

module.exports = { isAvailable, synthesize, synthesizeStream, transcribe, _prepareText, _intensityToExaggeration };
