/**
 * Language Detection + Translation
 *
 * Detects language from STT output and translates:
 *   - Input: any language → English (for StateGraph)
 *   - Output: English → user's detected language (for TTS response)
 *
 * Strategy:
 *   1. ElevenLabs Scribe already provides language_code on transcription — use that first
 *   2. If language is English → pass through, no API call needed
 *   3. If non-English → translate using LLM (WebSocket backend) or simple prompt
 *
 * Supported languages (primary focus): zh, es, en, fr, pt, ar, ja, ko, hi, de, it, ru
 */

'use strict';

const axios = require('axios');
const logger = require('./logger.cjs');

const SUPPORTED_LANGUAGES = (process.env.SUPPORTED_LANGUAGES || 'en,zh,es,fr,pt,ar,ja,ko,hi,de,it,ru').split(',');

const LANGUAGE_NAMES = {
  en: 'English',
  zh: 'Chinese (Mandarin)',
  'zh-cn': 'Chinese (Mandarin)',
  'zh-tw': 'Chinese (Traditional)',
  es: 'Spanish',
  fr: 'French',
  pt: 'Portuguese',
  ar: 'Arabic',
  ja: 'Japanese',
  ko: 'Korean',
  hi: 'Hindi',
  de: 'German',
  it: 'Italian',
  ru: 'Russian',
};

/**
 * Normalize language codes — map variants to base codes.
 */
// ISO 639-2 (3-letter) → ISO 639-1 (2-letter) map for common languages
const ISO3_TO_ISO1 = {
  eng: 'en', deu: 'de', fra: 'fr', spa: 'es', por: 'pt',
  ara: 'ar', jpn: 'ja', kor: 'ko', hin: 'hi', ita: 'it',
  rus: 'ru', zho: 'zh', cmn: 'zh', yue: 'zh',
};

function normalizeLanguage(langCode) {
  if (!langCode) return 'en';
  const base = langCode.toLowerCase().split('-')[0];
  if (ISO3_TO_ISO1[base]) return ISO3_TO_ISO1[base];
  if (base === 'zh') return 'zh';
  return base;
}

/**
 * Translate text from any language to English using LLM backend.
 *
 * @param {Object} args
 * @param {string} args.text          - Text to translate
 * @param {string} args.fromLanguage  - Source language code (e.g. 'zh', 'es')
 * @param {string} [args.toLanguage]  - Target language (default: 'en')
 * @returns {Promise<{translated: string, fromLanguage: string, toLanguage: string}>}
 */
async function translate({ text, fromLanguage, toLanguage = 'en' }) {
  const from = normalizeLanguage(fromLanguage);
  const to = normalizeLanguage(toLanguage);

  if (from === to) {
    return { translated: text, fromLanguage: from, toLanguage: to };
  }

  const fromName = LANGUAGE_NAMES[from] || from;
  const toName = LANGUAGE_NAMES[to] || to;

  logger.info('[Translate]', { from: fromName, to: toName, textPreview: text.substring(0, 60) });

  const translated = await _translateViaLLM({ text, fromName, toName });

  return { translated: translated.trim(), fromLanguage: from, toLanguage: to };
}

/**
 * Translate using the LLM backend WebSocket.
 * Uses a tight, deterministic prompt so the LLM returns ONLY the translation.
 */
async function _translateViaLLM({ text, fromName, toName }) {
  const wsUrl = process.env.STATEGRAPH_WS_URL || 'ws://localhost:4000/ws/stream';
  const apiKey = process.env.STATEGRAPH_API_KEY || '';

  const prompt = `Translate the following ${fromName} text to ${toName}. Return ONLY the translation, no explanation, no quotes.\n\nText: ${text}`;

  try {
    const result = await _wsLLMRequest(wsUrl, apiKey, prompt);
    return result || text;
  } catch (error) {
    logger.error('[Translate] LLM translation failed', { error: error.message });
    return text;
  }
}

/**
 * Fire a single-shot LLM request over the backend WebSocket, return full text.
 */
function _wsLLMRequest(wsUrl, apiKey, prompt) {
  return new Promise((resolve, reject) => {
    let WebSocket;
    try {
      WebSocket = require('ws');
    } catch (_) {
      reject(new Error('ws module not available'));
      return;
    }

    const ws = new WebSocket(wsUrl, {
      headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : {},
    });

    const requestId = `translate_${Date.now()}`;
    let fullText = '';
    let settled = false;
    const timeout = setTimeout(() => {
      if (!settled) { settled = true; ws.close(); reject(new Error('Translation timeout')); }
    }, 15000);

    ws.on('open', () => {
      ws.send(JSON.stringify({
        id: requestId,
        type: 'llm_request',
        payload: {
          prompt,
          options: { maxTokens: 500, temperature: 0.1, taskType: 'translation' },
        },
        timestamp: Date.now(),
      }));
    });

    ws.on('message', (data) => {
      try {
        const msg = JSON.parse(data.toString());
        if (msg.type === 'llm_stream_chunk' && msg.payload?.text) {
          fullText += msg.payload.text;
        } else if (msg.type === 'llm_stream_end' || (msg.parentId === requestId && msg.type === 'llm_stream_end')) {
          if (!settled) { settled = true; clearTimeout(timeout); ws.close(); resolve(fullText); }
        } else if (msg.type === 'error') {
          if (!settled) { settled = true; clearTimeout(timeout); ws.close(); reject(new Error(msg.payload?.message || 'LLM error')); }
        }
      } catch (_) {}
    });

    ws.on('error', (err) => {
      if (!settled) { settled = true; clearTimeout(timeout); reject(err); }
    });

    ws.on('close', () => {
      if (!settled) { settled = true; clearTimeout(timeout); resolve(fullText || ''); }
    });
  });
}

/**
 * Determine if the given language requires translation to English for StateGraph.
 */
function needsTranslation(langCode) {
  return normalizeLanguage(langCode) !== 'en';
}

/**
 * Script-based language override: if Groq/STT reports 'en' but the text
 * contains a dominant non-Latin script, trust the script over the tag.
 * Groq Whisper-large-v3-turbo sometimes returns language='en' for mixed or
 * short Chinese utterances — this catches it.
 */
function detectScriptLanguage(text) {
  const cjk = (text.match(/[\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF]/g) || []).length;
  const arabic = (text.match(/[\u0600-\u06FF]/g) || []).length;
  const cyrillic = (text.match(/[\u0400-\u04FF]/g) || []).length;
  const devanagari = (text.match(/[\u0900-\u097F]/g) || []).length;
  const hangul = (text.match(/[\uAC00-\uD7AF\u1100-\u11FF]/g) || []).length;
  const hiragana = (text.match(/[\u3040-\u309F\u30A0-\u30FF]/g) || []).length;
  const total = text.replace(/\s/g, '').length || 1;

  if (cjk / total > 0.15) return hiragana > cjk * 0.3 ? 'ja' : 'zh';
  if (hangul / total > 0.15) return 'ko';
  if (arabic / total > 0.15) return 'ar';
  if (cyrillic / total > 0.15) return 'ru';
  if (devanagari / total > 0.15) return 'hi';
  return null;
}

async function toEnglish(sttResult) {
  const { text, language } = sttResult;
  let detectedLanguage = normalizeLanguage(language || 'en');

  // Override if STT reports English but text is clearly a non-Latin script
  if (detectedLanguage === 'en') {
    const scriptLang = detectScriptLanguage(text);
    if (scriptLang) {
      logger.info('[Translate] Script-override: STT said en but detected script', { scriptLang, textPreview: text.substring(0, 40) });
      detectedLanguage = scriptLang;
    }
  }

  if (!needsTranslation(detectedLanguage)) {
    return {
      englishText: text,
      originalText: text,
      detectedLanguage,
      wasTranslated: false,
    };
  }

  const { translated } = await translate({ text, fromLanguage: detectedLanguage, toLanguage: 'en' });

  return {
    englishText: translated,
    originalText: text,
    detectedLanguage,
    wasTranslated: true,
  };
}

/**
 * Translate English response back to the user's language for TTS.
 *
 * @param {string} englishText      - Answer in English
 * @param {string} targetLanguage   - User's detected language code
 * @returns {Promise<string>}
 */
async function fromEnglish(englishText, targetLanguage) {
  const lang = normalizeLanguage(targetLanguage);
  if (lang === 'en') return englishText;

  // Skip API call if text is already in the target script (avoid Chinese→Chinese round-trip).
  // This happens when the LLM already replied in the target language (e.g. fast lane with
  // language suffix injected into system prompt).
  const alreadyInScript = detectScriptLanguage(englishText);
  if (alreadyInScript === lang) return englishText;
  // CJK family: if target is zh/ja/ko and text has CJK, trust it
  const CJK_LANGS = new Set(['zh', 'ja', 'ko']);
  if (CJK_LANGS.has(lang) && alreadyInScript && CJK_LANGS.has(alreadyInScript)) return englishText;

  const { translated } = await translate({ text: englishText, fromLanguage: 'en', toLanguage: lang });
  return translated;
}

module.exports = { translate, toEnglish, fromEnglish, normalizeLanguage, needsTranslation, LANGUAGE_NAMES, SUPPORTED_LANGUAGES };
