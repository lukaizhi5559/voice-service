'use strict';

/**
 * OpenAI TTS provider
 *
 * TTS:  OpenAI /v1/audio/speech — 57 languages, natural quality (~300ms)
 * STT:  Groq Whisper (reused from existing stt.cjs)
 *
 * Used as the fallback TTS when Kokoro WASM can't handle a language.
 * Returns audioBase64 (mp3) for the renderer to play via AudioContext.
 */

const https = require('https');
const logger = require('../logger.cjs');

const OPENAI_API_KEY = process.env.OPENAI_API_KEY || '';

// tts-1 = fast + cheap; tts-1-hd = higher quality (~2× latency)
const TTS_MODEL = process.env.OPENAI_TTS_MODEL || 'tts-1';
const TTS_VOICE = process.env.OPENAI_TTS_VOICE || 'nova';  // nova is warm, natural, multilingual

// OpenAI maps input language automatically — no explicit lang param needed.
// Full list: https://platform.openai.com/docs/guides/text-to-speech/supported-languages

function synthesize({ text, language }) {
  if (!OPENAI_API_KEY) {
    return Promise.reject(new Error('OPENAI_API_KEY not set'));
  }
  if (!text || !text.trim()) {
    return Promise.reject(new Error('text is required'));
  }

  const body = JSON.stringify({
    model: TTS_MODEL,
    input: text,
    voice: TTS_VOICE,
    response_format: 'mp3',
  });

  return new Promise((resolve, reject) => {
    const req = https.request({
      hostname: 'api.openai.com',
      path: '/v1/audio/speech',
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${OPENAI_API_KEY}`,
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(body),
      },
    }, (res) => {
      if (res.statusCode !== 200) {
        const chunks = [];
        res.on('data', c => chunks.push(c));
        res.on('end', () => {
          const msg = Buffer.concat(chunks).toString();
          reject(new Error(`OpenAI TTS HTTP ${res.statusCode}: ${msg}`));
        });
        return;
      }

      const chunks = [];
      res.on('data', chunk => chunks.push(chunk));
      res.on('end', () => {
        const audioBuffer = Buffer.concat(chunks);
        logger.info('[OpenAI TTS] synthesized', { bytes: audioBuffer.length, lang: language || 'auto', voice: TTS_VOICE });
        resolve({
          audioBuffer,
          format: 'mp3',
          sampleRate: 24000,
        });
      });
      res.on('error', reject);
    });

    req.on('error', reject);
    req.on('timeout', () => { req.destroy(); reject(new Error('OpenAI TTS timeout')); });
    req.setTimeout(15000);
    req.write(body);
    req.end();
  });
}

function isAvailable() {
  return !!OPENAI_API_KEY;
}

module.exports = { synthesize, isAvailable };
