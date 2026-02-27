/**
 * STT — Speech-to-text orchestrator.
 *
 * Provider priority:
 *   1. Groq Whisper (groq-stt.cjs) — primary, free tier, fast
 *   2. ElevenLabs Scribe            — fallback
 */

'use strict';

const fs = require('fs');
const path = require('path');
const os = require('os');
const FormData = require('form-data');
const axios = require('axios');
const logger = require('./logger.cjs');
const groqStt = require('./groq-stt.cjs');

// ── ElevenLabs Scribe (fallback) ──────────────────────────────────────────────
const ELEVENLABS_STT_URL = 'https://api.elevenlabs.io/v1/speech-to-text';
const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY;

/**
 * Transcribe audio buffer or file path.
 * Tries Groq Whisper first, falls back to ElevenLabs Scribe.
 *
 * @param {Object} args
 * @param {Buffer|string} args.audio   - Raw audio buffer OR path to audio file
 * @param {string} args.format         - Audio format: 'wav' | 'mp3' | 'webm' | 'ogg'
 * @param {string} [args.languageHint] - BCP-47 hint (e.g. 'en', 'es')
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
  if (groqStt.isAvailable()) {
    try {
      return await groqStt.transcribe({ audioBuffer, format, languageHint });
    } catch (err) {
      logger.warn('[STT] Groq Whisper failed, trying ElevenLabs Scribe', { error: err.message });
    }
  }

  // ── 2. Try ElevenLabs Scribe ─────────────────────────────────────────────
  if (ELEVENLABS_API_KEY) {
    return await _elevenLabsTranscribe({ audioBuffer, format, languageHint });
  }

  throw new Error('[STT] No STT provider available — set GROQ_API_KEY or ELEVENLABS_API_KEY');
}

/**
 * Transcribe base64-encoded audio.
 */
async function transcribeBase64({ audioBase64, format = 'wav', languageHint = null }) {
  const audio = Buffer.from(audioBase64, 'base64');
  return transcribeAudio({ audio, format, languageHint });
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
      headers: { ...form.getHeaders(), 'xi-api-key': ELEVENLABS_API_KEY },
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

async function isAvailable() {
  return groqStt.isAvailable() || !!ELEVENLABS_API_KEY;
}

module.exports = { transcribeAudio, transcribeBase64, isAvailable };
