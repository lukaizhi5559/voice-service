'use strict';

/**
 * Cartesia voice provider
 *
 * STT:  Groq Whisper large-v3-turbo
 * TTS:  Cartesia Sonic — fast, multilingual, high quality
 *
 * Uses Cartesia REST API directly (axios) for CJS compatibility.
 * Docs: https://docs.cartesia.ai/api-reference/tts/bytes
 */

const stt = require('../stt.cjs');
const logger = require('../logger.cjs');
const axios = require('axios');

const CARTESIA_API_KEY = process.env.CARTESIA_API_KEY;
const CARTESIA_API_VERSION = '2025-04-16';
const CARTESIA_TTS_URL = 'https://api.cartesia.ai/tts/bytes';

// Cartesia Sonic multilingual model
const CARTESIA_MODEL_ID = process.env.CARTESIA_MODEL_ID || 'sonic-2';

// Default voice — Cartesia "Barbershop Man" (deep, clear English)
// Change via CARTESIA_VOICE_ID env var
const DEFAULT_VOICE_ID = process.env.CARTESIA_VOICE_ID || 'a0e99841-438c-4a64-b679-ae501e7d6091';

// Language code map — Cartesia uses BCP-47
const LANG_MAP = {
  en: 'en', zh: 'zh', es: 'es', fr: 'fr', pt: 'pt',
  de: 'de', ja: 'ja', ko: 'ko', hi: 'hi', it: 'it',
  ru: 'ru', ar: 'ar', nl: 'nl', pl: 'pl', sv: 'sv',
  tr: 'tr', id: 'id', vi: 'vi',
};

function isAvailable() {
  return !!(process.env.GROQ_API_KEY || process.env.CARTESIA_API_KEY);
}

/**
 * Transcribe base64 audio → text using Groq Whisper.
 */
async function transcribe({ audioBase64, format = 'wav', languageHint = null }) {
  return stt.transcribeBase64({ audioBase64, format, languageHint });
}

/**
 * Synthesize text → audio buffer using Cartesia Sonic.
 */
async function synthesize({ text, language = 'en', voiceId }) {
  if (!CARTESIA_API_KEY) throw new Error('CARTESIA_API_KEY not set');

  const lang = LANG_MAP[(language || 'en').toLowerCase().split('-')[0]] || 'en';
  const voice = voiceId || DEFAULT_VOICE_ID;

  logger.info('[CartesiaProvider] synthesize', {
    textLen: text.length,
    language: lang,
    voiceId: voice,
    textPreview: text.substring(0, 80),
  });

  const response = await axios.post(CARTESIA_TTS_URL, {
    model_id: CARTESIA_MODEL_ID,
    transcript: text,
    voice: { mode: 'id', id: voice },
    output_format: { container: 'mp3', encoding: 'mp3', sample_rate: 44100 },
    language: lang,
  }, {
    headers: {
      'X-API-Key': CARTESIA_API_KEY,
      'Cartesia-Version': CARTESIA_API_VERSION,
      'Content-Type': 'application/json',
    },
    responseType: 'arraybuffer',
    timeout: 30000,
  });

  const audioBuffer = Buffer.from(response.data);
  const durationEstimateMs = Math.ceil((text.length / 15) * 1000);
  logger.info('[CartesiaProvider] audio generated', { bytes: audioBuffer.length, durationEstimateMs });
  return { audioBuffer, format: 'mp3', durationEstimateMs };
}

module.exports = { isAvailable, transcribe, synthesize };
