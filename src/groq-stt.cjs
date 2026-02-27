/**
 * Groq Whisper STT — Speech-to-text via groq-sdk.
 *
 * Model: whisper-large-v3-turbo (fast, free tier)
 * Supports: webm, mp3, wav, ogg, m4a, flac
 */

'use strict';

const fs = require('fs');
const path = require('path');
const os = require('os');
const logger = require('./logger.cjs');

const GROQ_API_KEY = process.env.GROQ_API_KEY;
const GROQ_MODEL = process.env.GROQ_STT_MODEL || 'whisper-large-v3-turbo';

/**
 * Transcribe audio buffer using Groq Whisper.
 *
 * @param {Object} args
 * @param {Buffer} args.audioBuffer  - Raw audio bytes
 * @param {string} args.format       - Audio format: webm | mp3 | wav | ogg | m4a | flac
 * @param {string} [args.languageHint] - BCP-47 language code hint (e.g. 'en', 'es')
 * @returns {Promise<{text: string, language: string, confidence: number, isFinal: boolean}>}
 */
async function transcribe({ audioBuffer, format = 'wav', languageHint = null }) {
  if (!GROQ_API_KEY) throw new Error('[GroqSTT] GROQ_API_KEY not set');

  const Groq = require('groq-sdk');
  const groq = new Groq({ apiKey: GROQ_API_KEY });

  const tmpFile = path.join(os.tmpdir(), `thinkdrop-groq-stt-${Date.now()}.${format}`);
  fs.writeFileSync(tmpFile, audioBuffer);

  logger.info('[STT] Sending audio to Groq Whisper', {
    bytes: audioBuffer.length,
    format,
    model: GROQ_MODEL,
    languageHint: languageHint || 'auto',
  });

  try {
    const params = {
      file: fs.createReadStream(tmpFile),
      model: GROQ_MODEL,
      response_format: 'json',
      temperature: 0.0,
      // Lock to English — prevents language misdetect on short/noisy clips (especially AirPod BT audio).
      // prompt is a short prior-utterance hint to seed decoder context (not an instruction sentence —
      // long sentences leak into the output on silent audio).
      language: languageHint || 'en',
      prompt: 'okay, search, open, scroll, hey,',
    };

    const transcription = await groq.audio.transcriptions.create(params);

    const text = (transcription.text || '').trim();
    const language = transcription.language || languageHint || 'en';

    logger.info('[STT] Groq transcription complete', {
      language,
      textPreview: text.substring(0, 80),
    });

    return { text, language, confidence: 1.0, isFinal: true };
  } finally {
    if (fs.existsSync(tmpFile)) try { fs.unlinkSync(tmpFile); } catch (_) {}
  }
}

/**
 * Check if Groq STT is available.
 */
function isAvailable() {
  return !!GROQ_API_KEY;
}

module.exports = { transcribe, isAvailable };
