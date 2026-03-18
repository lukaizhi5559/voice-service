/**
 * Deepgram Nova-2 STT — Speech-to-text via Deepgram REST API.
 *
 * WHY DEEPGRAM OVER GROQ WHISPER:
 *   Groq Whisper hallucinates rare proper nouns catastrophically — "Armis" becomes
 *   "harmless", "K-R-M-E.", "armies", etc. Deepgram Nova-2 supports keyword boosting
 *   (`keywords=Armis:5`) which tells the model to strongly favor those exact spellings,
 *   dramatically improving accuracy for the wake word and app-specific vocabulary.
 *
 * MODEL: nova-2-general (best accuracy/speed balance as of 2025)
 *   Falls back to nova-2 if env var unset.
 *
 * KEYWORD BOOSTING:
 *   Set DEEPGRAM_KEYWORDS env var as comma-separated word:intensifier pairs, e.g.:
 *     DEEPGRAM_KEYWORDS=Armis:5,ThinkDrop:5,Perplexity:3,Claude:3
 *   Intensifiers are exponential — keep ≤5 to avoid false positives on common words.
 *
 * SETUP:
 *   1. Get a free API key at https://console.deepgram.com (free $200 credit on signup)
 *   2. Add DEEPGRAM_API_KEY=your_key to .env
 */

'use strict';

const fs   = require('fs');
const path = require('path');
const os   = require('os');
const https = require('https');
const FormData = require('form-data');
const logger = require('./logger.cjs');

const DEEPGRAM_API_KEY = process.env.DEEPGRAM_API_KEY;
// nova-2 (not nova-2-general) supports 30+ languages with detect_language=true
const DEEPGRAM_MODEL   = process.env.DEEPGRAM_STT_MODEL || 'nova-2';

// Keyword boosting — wire to DEEPGRAM_KEYWORDS env or hardcode app-specific terms.
// Format: "Word:intensifier" pairs, comma-separated.
const DEFAULT_KEYWORDS = 'Armis:5,ThinkDrop:5';
const KEYWORDS_RAW = process.env.DEEPGRAM_KEYWORDS || DEFAULT_KEYWORDS;

/**
 * Build the `keywords` query params array from the config string.
 * Returns an array of "keyword=word%3Aintensifier" strings for the URL.
 */
function _buildKeywordParams() {
  return KEYWORDS_RAW
    .split(',')
    .map(k => k.trim())
    .filter(Boolean)
    .map(k => `keywords=${encodeURIComponent(k)}`);
}

/**
 * Map audio format string to MIME type.
 */
function _mimeType(format) {
  const map = {
    webm: 'audio/webm',
    mp3:  'audio/mpeg',
    wav:  'audio/wav',
    ogg:  'audio/ogg',
    m4a:  'audio/mp4',
    flac: 'audio/flac',
    mp4:  'audio/mp4',
  };
  return map[format.split(';')[0].trim().toLowerCase()] || 'audio/webm';
}

/**
 * Transcribe audio buffer using Deepgram Nova-2.
 *
 * @param {Object} args
 * @param {Buffer} args.audioBuffer  - Raw audio bytes
 * @param {string} args.format       - Audio format: webm | mp3 | wav | ogg | m4a | flac
 * @param {string} [args.languageHint] - BCP-47 language code hint (e.g. 'en', 'es')
 * @returns {Promise<{text: string, language: string, confidence: number, isFinal: boolean}>}
 */
async function transcribe({ audioBuffer, format = 'webm', languageHint = null }) {
  if (!DEEPGRAM_API_KEY) throw new Error('[DeepgramSTT] DEEPGRAM_API_KEY not set');

  // Build query string
  const keywordParams = _buildKeywordParams();
  // Keyword boosting only helps for English (Deepgram's `keywords` param is English-only).
  // For non-English sessions, skip it to avoid wasting query budget.
  const isEnglishContext = !languageHint || languageHint === 'en';

  const queryParts = [
    `model=${DEEPGRAM_MODEL}`,
    'punctuate=true',
    'smart_format=true',
    // Pin language for non-Latin scripts; auto-detect for Latin-script languages
    ...(languageHint && languageHint !== 'en' ? [`language=${languageHint}`] : ['detect_language=true']),
    // Keyword boosting: English only (proper nouns like "Armis", "ThinkDrop")
    ...(isEnglishContext ? keywordParams : []),
  ];
  const query = queryParts.join('&');
  const url = `https://api.deepgram.com/v1/listen?${query}`;

  const mimeType = _mimeType(format);

  logger.info('[STT] Sending audio to Deepgram Nova-2', {
    bytes: audioBuffer.length,
    format,
    model: DEEPGRAM_MODEL,
    languageHint: languageHint || 'auto-detect',
    keywords: KEYWORDS_RAW,
  });

  return new Promise((resolve, reject) => {
    const urlObj = new URL(url);

    const options = {
      hostname: urlObj.hostname,
      path: urlObj.pathname + urlObj.search,
      method: 'POST',
      headers: {
        'Authorization': `Token ${DEEPGRAM_API_KEY}`,
        'Content-Type': mimeType,
        'Content-Length': audioBuffer.length,
      },
    };

    const req = https.request(options, (res) => {
      let body = '';
      res.on('data', chunk => { body += chunk; });
      res.on('end', () => {
        clearTimeout(timeout);
        if (res.statusCode !== 200) {
          return reject(new Error(`[DeepgramSTT] HTTP ${res.statusCode}: ${body.slice(0, 300)}`));
        }
        try {
          const parsed = JSON.parse(body);
          const alt = parsed?.results?.channels?.[0]?.alternatives?.[0];
          if (!alt) return reject(new Error('[DeepgramSTT] No transcription result in response'));

          const text = (alt.transcript || '').trim();
          const language = parsed?.results?.channels?.[0]?.detected_language
            || languageHint
            || 'en';
          const confidence = alt.confidence ?? 1.0;

          logger.info('[STT] Deepgram transcription complete', {
            language,
            confidence: confidence.toFixed(3),
            textPreview: text.substring(0, 80),
          });

          resolve({ text, language, confidence, isFinal: true });
        } catch (parseErr) {
          reject(new Error(`[DeepgramSTT] JSON parse error: ${parseErr.message}`));
        }
      });
    });

    req.on('error', err => reject(new Error(`[DeepgramSTT] Request error: ${err.message}`)));

    const timeout = setTimeout(() => {
      req.destroy();
      reject(new Error('[DeepgramSTT] Request timed out after 15s'));
    }, 15000);

    req.write(audioBuffer);
    req.end();
  });
}

/**
 * @returns {boolean}
 */
function isAvailable() {
  return !!DEEPGRAM_API_KEY;
}

module.exports = { transcribe, isAvailable };
