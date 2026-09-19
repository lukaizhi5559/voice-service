/**
 * STT — Speech-to-text with automatic language detection.
 *
 * Provider priority:
 *   1. Groq Whisper (whisper-large-v3-turbo) — primary, free tier, fast
 *   2. ElevenLabs Scribe — fallback
 */

'use strict';

const fs = require('fs');
const path = require('path');
const os = require('os');
const FormData = require('form-data');
const axios = require('axios');
const logger = require('../logger.cjs');

// ── Groq Whisper ─────────────────────────────────────────────────────────────
const GROQ_STT_URL = 'https://api.groq.com/openai/v1/audio/transcriptions';
const GROQ_API_KEY = process.env.GROQ_API_KEY;
const GROQ_MODEL = 'whisper-large-v3-turbo';

// ── ElevenLabs Scribe (fallback) ──────────────────────────────────────────────
const ELEVENLABS_STT_URL = 'https://api.elevenlabs.io/v1/speech-to-text';
const API_KEY = process.env.ELEVENLABS_API_KEY;

/**
 * Transcribe audio buffer or file path using ElevenLabs Scribe.
 *
 * @param {Object} args
 * @param {Buffer|string} args.audio   - Raw audio buffer OR path to audio file
 * @param {string} args.format         - Audio format: 'wav' | 'mp3' | 'webm' | 'ogg' (default: 'wav')
 * @param {string} [args.languageHint] - BCP-47 hint (e.g. 'zh', 'es') — Scribe auto-detects if omitted
 * @returns {Promise<{text: string, language: string, confidence: number, isFinal: boolean}>}
 */
async function transcribeAudio({ audio, format = 'wav', languageHint = null }) {
  let audioBuffer;

  if (typeof audio === 'string') {
    audioBuffer = fs.readFileSync(audio);
  } else if (Buffer.isBuffer(audio)) {
    audioBuffer = audio;
  } else if (audio instanceof Uint8Array) {
    audioBuffer = Buffer.from(audio);
  } else {
    throw new Error('[STT] audio must be a Buffer, Uint8Array, or file path string');
  }

  // ── 1. Try Groq Whisper ───────────────────────────────────────────────────
  if (GROQ_API_KEY) {
    try {
      return await _groqTranscribe({ audioBuffer, format, languageHint });
    } catch (err) {
      logger.warn('[STT] Groq Whisper failed, trying ElevenLabs Scribe', { error: err.message });
    }
  }

  // ── 2. Try ElevenLabs Scribe ─────────────────────────────────────────────
  if (API_KEY) {
    return await _elevenLabsTranscribe({ audioBuffer, format, languageHint });
  }

  throw new Error('[STT] No STT provider available — set GROQ_API_KEY or ELEVENLABS_API_KEY');
}

async function _groqTranscribe({ audioBuffer, format, languageHint }) {
  const tmpFile = path.join(os.tmpdir(), `thinkdrop-stt-${Date.now()}.${format}`);
  fs.writeFileSync(tmpFile, audioBuffer);

  try {
    const form = new FormData();
    form.append('file', fs.createReadStream(tmpFile), {
      filename: `audio.${format}`,
      contentType: _mimeType(format),
    });
    form.append('model', GROQ_MODEL);
    form.append('response_format', 'json');
    if (languageHint) form.append('language', languageHint);

    logger.info('[STT] Sending audio to Groq Whisper', {
      bytes: audioBuffer.length,
      format,
      languageHint: languageHint || 'auto',
    });

    const response = await axios.post(GROQ_STT_URL, form, {
      headers: {
        ...form.getHeaders(),
        'Authorization': `Bearer ${GROQ_API_KEY}`,
      },
      timeout: 30000,
    });

    const text = (response.data?.text || '').trim();
    const language = response.data?.language || languageHint || 'en';

    logger.info('[STT] Groq transcription complete', {
      language,
      textPreview: text.substring(0, 80),
    });

    return { text, language, confidence: 1.0, isFinal: true };
  } finally {
    if (fs.existsSync(tmpFile)) try { fs.unlinkSync(tmpFile); } catch (_) {}
  }
}

async function _elevenLabsTranscribe({ audioBuffer, format, languageHint }) {
  const tmpFile = path.join(os.tmpdir(), `thinkdrop-stt-${Date.now()}.${format}`);
  fs.writeFileSync(tmpFile, audioBuffer);

  try {
    const form = new FormData();
    form.append('file', fs.createReadStream(tmpFile), {
      filename: `audio.${format}`,
      contentType: _mimeType(format),
    });
    form.append('model_id', 'scribe_v1');
    form.append('tag_audio_events', 'false');
    if (languageHint) form.append('language_code', languageHint);

    logger.info('[STT] Sending audio to ElevenLabs Scribe', {
      bytes: audioBuffer.length,
      format,
      languageHint: languageHint || 'auto',
    });

    const response = await axios.post(ELEVENLABS_STT_URL, form, {
      headers: { ...form.getHeaders(), 'xi-api-key': API_KEY },
      timeout: 30000,
    });

    const { text, language_code, language_probability } = response.data;

    logger.info('[STT] ElevenLabs transcription complete', {
      language: language_code,
      confidence: language_probability,
      textPreview: (text || '').substring(0, 80),
    });

    return {
      text: (text || '').trim(),
      language: language_code || 'en',
      confidence: language_probability || 1.0,
      isFinal: true,
    };
  } finally {
    if (fs.existsSync(tmpFile)) try { fs.unlinkSync(tmpFile); } catch (_) {}
  }
}

/**
 * Transcribe base64-encoded audio (useful for WebSocket payloads).
 *
 * @param {Object} args
 * @param {string} args.audioBase64 - Base64-encoded audio data
 * @param {string} args.format      - Audio format
 * @param {string} [args.languageHint]
 */
async function transcribeBase64({ audioBase64, format = 'wav', languageHint = null }) {
  const audio = Buffer.from(audioBase64, 'base64');
  return transcribeAudio({ audio, format, languageHint });
}

function _mimeType(format) {
  const map = {
    wav: 'audio/wav',
    mp3: 'audio/mpeg',
    webm: 'audio/webm',
    ogg: 'audio/ogg',
    m4a: 'audio/mp4',
    flac: 'audio/flac',
  };
  return map[format] || 'audio/wav';
}

/**
 * Check if ElevenLabs STT is available (API key set + reachable).
 */
async function isAvailable() {
  if (GROQ_API_KEY) return true;
  if (!API_KEY) return false;
  try {
    const response = await axios.get('https://api.elevenlabs.io/v1/models', {
      headers: { 'xi-api-key': API_KEY },
      timeout: 5000,
    });
    return response.status === 200;
  } catch (_) {
    return false;
  }
}

module.exports = { transcribeAudio, transcribeBase64, isAvailable };
