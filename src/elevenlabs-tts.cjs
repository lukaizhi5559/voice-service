/**
 * TTS — Text-to-speech with multilingual support.
 *
 * Provider priority:
 *   1. Inworld TTS (inworld-tts-1.5-max) — primary
 *   2. ElevenLabs TTS — second fallback
 *   3. macOS `say` command — offline last resort
 */

'use strict';

const fs = require('fs');
const path = require('path');
const os = require('os');
const { execSync, spawn } = require('child_process');
const axios = require('axios');
const logger = require('./logger.cjs');

// ── Inworld TTS ──────────────────────────────────────────────────────────────
const INWORLD_TTS_URL = 'https://api.inworld.ai/tts/v1/voice';
const INWORLD_API_KEY = process.env.INWORLD_API_KEY;  // Basic (Base64) key from platform.inworld.ai
const INWORLD_VOICE_ID = process.env.INWORLD_VOICE_ID || 'Dennis';
const INWORLD_MODEL_ID = process.env.INWORLD_MODEL_ID || 'inworld-tts-1.5-max';

// ── ElevenLabs TTS ───────────────────────────────────────────────────────────
const ELEVENLABS_TTS_BASE = 'https://api.elevenlabs.io/v1/text-to-speech';
const API_KEY = process.env.ELEVENLABS_API_KEY;

const DEFAULT_VOICE_ID = process.env.ELEVENLABS_DEFAULT_VOICE_ID || 'pNInz6obpgDQGcFmaJgB';
const MODEL_ID = process.env.ELEVENLABS_MODEL_ID || 'eleven_turbo_v2_5';

/**
 * Voice IDs optimized per language family.
 * These are ElevenLabs pre-built voices — fall back to default for unlisted langs.
 */
const LANGUAGE_VOICE_MAP = {
  en: DEFAULT_VOICE_ID,
  zh: DEFAULT_VOICE_ID,
  es: DEFAULT_VOICE_ID,
  fr: DEFAULT_VOICE_ID,
  pt: DEFAULT_VOICE_ID,
  ar: DEFAULT_VOICE_ID,
  ja: DEFAULT_VOICE_ID,
  ko: DEFAULT_VOICE_ID,
  hi: DEFAULT_VOICE_ID,
  de: DEFAULT_VOICE_ID,
  it: DEFAULT_VOICE_ID,
  ru: DEFAULT_VOICE_ID,
};

/**
 * Synthesize text to speech and return audio buffer.
 *
 * @param {Object} args
 * @param {string} args.text           - Text to synthesize
 * @param {string} [args.language]     - Target language code (e.g. 'zh', 'es', 'en')
 * @param {string} [args.voiceId]      - Override voice ID
 * @param {number} [args.stability]    - 0.0–1.0 (default 0.5)
 * @param {number} [args.similarity]   - 0.0–1.0 (default 0.75)
 * @param {number} [args.style]        - 0.0–1.0 (default 0.0, use 0.3 for expressive)
 * @param {Function} [args.onChunk]    - Streaming callback: called with each Buffer chunk
 * @returns {Promise<{audioBuffer: Buffer, format: string, durationEstimateMs: number}>}
 */
async function synthesize({ text, language = 'en', voiceId, stability = 0.5, similarity = 0.75, style = 0.0, onChunk = null }) {
  // ── 1. Try Inworld TTS ───────────────────────────────────────────────────
  if (INWORLD_API_KEY) {
    try {
      return await _inworldTTS({ text, language });
    } catch (err) {
      logger.warn('[TTS] Inworld failed, trying ElevenLabs', { error: err.message });
    }
  }

  // ── 2. Try ElevenLabs ────────────────────────────────────────────────────
  if (API_KEY) {
    const selectedVoice = voiceId || LANGUAGE_VOICE_MAP[language] || DEFAULT_VOICE_ID;
    const url = `${ELEVENLABS_TTS_BASE}/${selectedVoice}${onChunk ? '/stream' : ''}`;
    const body = {
      text,
      model_id: MODEL_ID,
      voice_settings: { stability, similarity_boost: similarity, style, use_speaker_boost: true },
    };
    logger.info('[TTS] Synthesizing text (ElevenLabs)', {
      language, voiceId: selectedVoice, textPreview: text.substring(0, 80), streaming: !!onChunk,
    });
    try {
      if (onChunk) return await _streamingTTS(url, body, onChunk);
      return await _blockingTTS(url, body);
    } catch (error) {
      logger.error('[TTS] ElevenLabs failed, falling back to native', { error: error.message });
    }
  }

  // ── 3. macOS native fallback ─────────────────────────────────────────────
  logger.warn('[TTS] All cloud TTS unavailable — falling back to native macOS TTS');
  return synthesizeNative({ text, language });
}

/**
 * Inworld TTS — blocking synthesis returning mp3 buffer.
 */
async function _inworldTTS({ text, language = 'en' }) {
  // Hard cap: Inworld allows up to 2000 chars per request
  const cappedText = text.length > 1900 ? text.substring(0, 1900) + '...' : text;

  logger.info('[TTS] Synthesizing text (Inworld)', {
    language,
    voiceId: INWORLD_VOICE_ID,
    textPreview: cappedText.substring(0, 80),
  });

  const response = await axios.post(
    INWORLD_TTS_URL,
    {
      text: cappedText,
      voiceId: INWORLD_VOICE_ID,
      modelId: INWORLD_MODEL_ID,
    },
    {
      headers: {
        'Authorization': `Basic ${INWORLD_API_KEY}`,
        'Content-Type': 'application/json',
      },
      timeout: 15000,
    }
  );

  const audioContent = response.data?.audioContent;
  if (!audioContent) throw new Error('Inworld returned no audioContent');

  const audioBuffer = Buffer.from(audioContent, 'base64');
  const durationEstimateMs = Math.ceil((cappedText.length / 15) * 1000);
  logger.info('[TTS] Inworld audio generated', { bytes: audioBuffer.length, durationEstimateMs });
  return { audioBuffer, format: 'mp3', durationEstimateMs };
}

async function _blockingTTS(url, body) {
  const response = await axios.post(url, body, {
    headers: {
      'xi-api-key': API_KEY,
      'Content-Type': 'application/json',
      'Accept': 'audio/mpeg',
    },
    responseType: 'arraybuffer',
    timeout: 30000,
  });

  const audioBuffer = Buffer.from(response.data);
  const durationEstimateMs = Math.ceil((body.text.length / 15) * 1000);

  logger.info('[TTS] Audio generated', { bytes: audioBuffer.length, durationEstimateMs });

  return { audioBuffer, format: 'mp3', durationEstimateMs };
}

async function _streamingTTS(url, body, onChunk) {
  const response = await axios.post(url, body, {
    headers: {
      'xi-api-key': API_KEY,
      'Content-Type': 'application/json',
      'Accept': 'audio/mpeg',
    },
    responseType: 'stream',
    timeout: 30000,
  });

  const chunks = [];
  return new Promise((resolve, reject) => {
    response.data.on('data', (chunk) => {
      chunks.push(chunk);
      try { onChunk(chunk); } catch (_) {}
    });
    response.data.on('end', () => {
      const audioBuffer = Buffer.concat(chunks);
      const durationEstimateMs = Math.ceil((body.text.length / 15) * 1000);
      resolve({ audioBuffer, format: 'mp3', durationEstimateMs });
    });
    response.data.on('error', reject);
  });
}

/**
 * Native macOS TTS fallback using the `say` command.
 * Supports limited language/voice selection via macOS voices.
 */
async function synthesizeNative({ text, language = 'en' }) {
  const NATIVE_VOICES = {
    en: 'Samantha',
    zh: 'Tingting',
    es: 'Monica',
    fr: 'Thomas',
    pt: 'Luciana',
    ja: 'Kyoko',
    ko: 'Yuna',
    de: 'Anna',
    it: 'Alice',
    ru: 'Milena',
    ar: 'Maged',
    hi: 'Lekha',
  };

  const voice = NATIVE_VOICES[language] || 'Samantha';
  const tmpFile = path.join(os.tmpdir(), `thinkdrop-tts-${Date.now()}.aiff`);

  logger.info('[TTS] Using native macOS TTS', { voice, language });

  const wavFile = tmpFile.replace('.aiff', '.wav');
  return new Promise((resolve, reject) => {
    const proc = spawn('say', ['-v', voice, '-o', tmpFile, '--data-format=LEI16@22050', text]);
    proc.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`macOS say command failed with code ${code}`));
        return;
      }
      try {
        // Convert AIFF → WAV so browsers can decode it
        execSync(`afconvert -f WAVE -d LEI16@22050 "${tmpFile}" "${wavFile}"`, { timeout: 10000 });
        const audioBuffer = fs.readFileSync(wavFile);
        try { fs.unlinkSync(tmpFile); } catch (_) {}
        try { fs.unlinkSync(wavFile); } catch (_) {}
        resolve({ audioBuffer, format: 'wav', durationEstimateMs: text.length * 70 });
      } catch (err) {
        reject(err);
      }
    });
    proc.on('error', reject);
  });
}

/**
 * Play audio directly on macOS (for testing/dev — production plays via Electron renderer).
 */
async function playAudioBuffer({ audioBuffer, format = 'mp3' }) {
  const tmpFile = path.join(os.tmpdir(), `thinkdrop-play-${Date.now()}.${format}`);
  fs.writeFileSync(tmpFile, audioBuffer);
  try {
    execSync(`afplay "${tmpFile}"`, { timeout: 60000 });
  } finally {
    try { fs.unlinkSync(tmpFile); } catch (_) {}
  }
}

/**
 * Check if ElevenLabs TTS is available.
 */
async function isAvailable() {
  if (!API_KEY) return false;
  try {
    const res = await axios.get('https://api.elevenlabs.io/v1/user', {
      headers: { 'xi-api-key': API_KEY },
      timeout: 5000,
    });
    return res.status === 200;
  } catch (_) {
    return false;
  }
}

module.exports = { synthesize, synthesizeNative, playAudioBuffer, isAvailable };
