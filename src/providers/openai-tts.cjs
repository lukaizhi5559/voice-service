'use strict';

/**
 * OpenAI TTS provider
 *
 * TTS:  OpenAI /v1/audio/speech — gpt-4o-mini-tts is the model behind
 *       ChatGPT voice mode. Voices marin/cedar are the latest ChatGPT
 *       voices; `instructions` lets us steer tone ("warm, natural").
 *
 * Returns the full audio buffer; callers that stream can pass onChunk()
 * to receive data as it arrives.
 */

const https = require('https');
const logger = require('../logger.cjs');

const OPENAI_API_KEY = process.env.OPENAI_API_KEY || '';

// gpt-4o-mini-tts = ChatGPT voice model; marin/cedar = recommended voices
const TTS_MODEL = process.env.OPENAI_TTS_MODEL || 'gpt-4o-mini-tts';
const TTS_VOICE = process.env.OPENAI_TTS_VOICE || 'marin';
const TTS_INSTRUCTIONS = process.env.OPENAI_TTS_INSTRUCTIONS || '';
const TTS_FORMAT = process.env.OPENAI_TTS_FORMAT || 'mp3'; // mp3 | wav | pcm | opus

// OpenAI maps input language automatically — no explicit lang param needed.
// Full list: https://platform.openai.com/docs/guides/text-to-speech/supported-languages

function synthesize({ text, language, onChunk }) {
  if (!OPENAI_API_KEY) {
    return Promise.reject(new Error('OPENAI_API_KEY not set'));
  }
  if (!text || !text.trim()) {
    return Promise.reject(new Error('text is required'));
  }

  const payload = {
    model: TTS_MODEL,
    input: text,
    voice: TTS_VOICE,
    response_format: TTS_FORMAT,
  };
  // `instructions` is supported on gpt-4o-mini-tts (steers tone/delivery)
  if (TTS_INSTRUCTIONS && TTS_MODEL.includes('gpt-4o')) {
    payload.instructions = TTS_INSTRUCTIONS;
  }
  const body = JSON.stringify(payload);

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
      res.on('data', chunk => {
        chunks.push(chunk);
        if (onChunk) { try { onChunk(chunk); } catch (_) {} }
      });
      res.on('end', () => {
        const audioBuffer = Buffer.concat(chunks);
        logger.info('[OpenAI TTS] synthesized', { bytes: audioBuffer.length, lang: language || 'auto', voice: TTS_VOICE, model: TTS_MODEL });
        resolve({
          audioBuffer,
          format: TTS_FORMAT,
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
