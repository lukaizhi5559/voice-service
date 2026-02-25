/**
 * ElevenLabs TTS — Text-to-speech with multilingual support.
 *
 * Uses ElevenLabs streaming TTS API with turbo model for low latency.
 * Returns audio as a Buffer (mp3) or streams chunks via callback.
 *
 * Supports automatic voice selection per language:
 *   - English: default configured voice
 *   - Other languages: same voice (ElevenLabs multilingual model handles accent/language)
 *
 * Fallback: macOS `say` command for offline operation.
 */

'use strict';

const fs = require('fs');
const path = require('path');
const os = require('os');
const { execSync, spawn } = require('child_process');
const axios = require('axios');
const logger = require('./logger.cjs');

const ELEVENLABS_TTS_BASE = 'https://api.elevenlabs.io/v1/text-to-speech';
const API_KEY = process.env.ELEVENLABS_API_KEY;

const DEFAULT_VOICE_ID = process.env.ELEVENLABS_DEFAULT_VOICE_ID || '21m00Tcm4TlvDq8ikWAM';
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
  if (!API_KEY) {
    logger.warn('[TTS] ELEVENLABS_API_KEY not set — falling back to native TTS');
    return synthesizeNative({ text, language });
  }

  const selectedVoice = voiceId || LANGUAGE_VOICE_MAP[language] || DEFAULT_VOICE_ID;

  const url = `${ELEVENLABS_TTS_BASE}/${selectedVoice}${onChunk ? '/stream' : ''}`;

  const body = {
    text,
    model_id: MODEL_ID,
    voice_settings: {
      stability,
      similarity_boost: similarity,
      style,
      use_speaker_boost: true,
    },
  };

  logger.info('[TTS] Synthesizing text', {
    language,
    voiceId: selectedVoice,
    textPreview: text.substring(0, 80),
    streaming: !!onChunk,
  });

  try {
    if (onChunk) {
      return await _streamingTTS(url, body, onChunk);
    } else {
      return await _blockingTTS(url, body);
    }
  } catch (error) {
    logger.error('[TTS] ElevenLabs failed, falling back to native', { error: error.message });
    return synthesizeNative({ text, language });
  }
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
