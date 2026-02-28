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
    // Language-appropriate decoder seed prompts — biases Whisper output toward the correct script.
    // Short words only: long sentences leak into output on silent audio.
    const LANG_PROMPTS = {
      zh: '好的，搜索，打开，',
      ja: 'はい、検索、開く、',
      ko: '네, 검색, 열기,',
      ar: 'حسنًا، بحث، افتح،',
      ru: 'хорошо, поиск, открыть,',
      hi: 'ठीक है, खोजें, खोलें,',
    };
    const effectiveLang = languageHint || 'en';
    const decoderPrompt = LANG_PROMPTS[effectiveLang] || 'okay, search, open, scroll, hey,';

    const params = {
      file: fs.createReadStream(tmpFile),
      model: GROQ_MODEL,
      response_format: 'json',
      temperature: 0.0,
      language: effectiveLang,
      prompt: decoderPrompt,
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
