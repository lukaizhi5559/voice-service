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
const GROQ_MODEL      = process.env.GROQ_STT_MODEL      || 'whisper-large-v3-turbo';
const GROQ_MODEL_AUTO = process.env.GROQ_STT_MODEL_AUTO || 'whisper-large-v3';

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

  const hasHintForLog = !!(languageHint && languageHint !== 'en');
  const modelForLog = hasHintForLog ? GROQ_MODEL : GROQ_MODEL_AUTO;
  logger.info('[STT] Sending audio to Groq Whisper', {
    bytes: audioBuffer.length,
    format,
    model: modelForLog,
    languageHint: languageHint || 'auto-detect',
  });

  try {
    // Language-appropriate decoder seed prompts — biases Whisper output toward the correct script.
    // Short words only: long sentences leak into output on silent audio.
    const LANG_PROMPTS = {
      zh: '好的，搜索，打开，Claude，Perplexity，Google，ChatGPT，YouTube，',
      ja: 'はい、検索、開く、Claude、Perplexity、Google、',
      ko: '네, 검색, 열기, Claude, Perplexity, Google,',
      ar: 'حسنًا، بحث، افتح، Claude، Perplexity، Google،',
      ru: 'хорошо, поиск, открыть, Claude, Perplexity, Google,',
      hi: 'ठीक है, खोजें, खोलें, Claude, Perplexity, Google,',
    };

    // When no hint is given, use large-v3 for auto-detection — it has better multilingual
    // accuracy than turbo (especially for CJK detection and language identification).
    // When a language hint IS given, turbo is fast enough and we lock the decoder.
    // Never pass language='en' explicitly — that disables multilingual detection.
    const hasHint = !!(languageHint && languageHint !== 'en');
    const selectedModel = hasHint ? GROQ_MODEL : GROQ_MODEL_AUTO;
    const decoderPrompt = hasHint ? (LANG_PROMPTS[languageHint] || 'okay, search, open, scroll, hey,') : 'okay, search, open, scroll, hey,';

    const params = {
      file: fs.createReadStream(tmpFile),
      model: selectedModel,
      response_format: 'json',
      temperature: 0.0,
      prompt: decoderPrompt,
      // Only lock to a language when explicitly hinted — omit for auto-detection
      ...(hasHint ? { language: languageHint } : {}),
    };

    const transcription = await groq.audio.transcriptions.create(params);

    const text = (transcription.text || '').trim();
    // Trust Groq's detected language — do not fall back to 'en' as that masks auto-detection.
    const language = transcription.language || languageHint || 'und';

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
