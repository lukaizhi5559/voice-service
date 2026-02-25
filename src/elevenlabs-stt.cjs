/**
 * ElevenLabs Scribe STT — Speech-to-text with automatic language detection.
 *
 * Uses ElevenLabs Scribe model which supports 99 languages natively.
 * Returns transcribed text, detected language code, and confidence.
 *
 * Fallback: Web Speech API via renderer IPC (when ElevenLabs unavailable/offline).
 */

'use strict';

const fs = require('fs');
const path = require('path');
const os = require('os');
const FormData = require('form-data');
const axios = require('axios');
const logger = require('./logger.cjs');

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
  if (!API_KEY) {
    throw new Error('[STT] ELEVENLABS_API_KEY not set');
  }

  let audioBuffer;
  let tmpFile = null;

  if (typeof audio === 'string') {
    audioBuffer = fs.readFileSync(audio);
  } else if (Buffer.isBuffer(audio)) {
    audioBuffer = audio;
  } else if (audio instanceof Uint8Array) {
    audioBuffer = Buffer.from(audio);
  } else {
    throw new Error('[STT] audio must be a Buffer, Uint8Array, or file path string');
  }

  // Write to a temp file — ElevenLabs STT requires multipart upload
  tmpFile = path.join(os.tmpdir(), `thinkdrop-stt-${Date.now()}.${format}`);
  fs.writeFileSync(tmpFile, audioBuffer);

  try {
    const form = new FormData();
    form.append('file', fs.createReadStream(tmpFile), {
      filename: `audio.${format}`,
      contentType: _mimeType(format),
    });
    form.append('model_id', 'scribe_v1');
    form.append('tag_audio_events', 'false'); // disable (laughter)/(music)/(whirring) labels — key noise fix

    if (languageHint) {
      form.append('language_code', languageHint);
    }

    logger.info('[STT] Sending audio to ElevenLabs Scribe', {
      bytes: audioBuffer.length,
      format,
      languageHint: languageHint || 'auto',
    });

    const response = await axios.post(ELEVENLABS_STT_URL, form, {
      headers: {
        ...form.getHeaders(),
        'xi-api-key': API_KEY,
      },
      timeout: 30000,
    });

    const { text, language_code, language_probability } = response.data;

    logger.info('[STT] Transcription complete', {
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
    if (tmpFile && fs.existsSync(tmpFile)) {
      try { fs.unlinkSync(tmpFile); } catch (_) {}
    }
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
