/**
 * TTS — Inworld TTS with macOS native fallback.
 *
 * Provider priority:
 *   1. Inworld TTS (inworld-tts-1.5-max) — English only
 *   2. macOS `say` command — offline last resort
 *
 * For multilingual TTS use the Cartesia provider instead.
 */

'use strict';

const fs = require('fs');
const path = require('path');
const os = require('os');
const { execSync, spawn } = require('child_process');
const axios = require('axios');
const logger = require('./logger.cjs');

// ── Inworld TTS ────────────────────────────────────────────────────────────
const INWORLD_TTS_URL = 'https://api.inworld.ai/tts/v1/voice';
const INWORLD_API_KEY = process.env.INWORLD_API_KEY;
const INWORLD_VOICE_ID = process.env.INWORLD_VOICE_ID || 'Dennis';
const INWORLD_MODEL_ID = process.env.INWORLD_MODEL_ID || 'inworld-tts-1.5-max';

/**
 * Synthesize text to speech and return audio buffer.
 *
 * @param {Object} args
 * @param {string} args.text           - Text to synthesize
 * @param {string} [args.language]     - Target language code (e.g. 'zh', 'es', 'en')
 * @param {string} [args.voiceId]      - Override voice ID
 * @param {number} [args.stability]    - 0.0–1.0 (default 0.5)
 * @param {number} [args.similarity]   - 0.0–1.0 (default 0.75)
 * @param {number} [args.style]        - 0.0–1.0 (default 0.0, use 0.3 for expressive)
 * @param {Function} [args.onChunk]    - Streaming callback: called with each Buffer chunk
 * @returns {Promise<{audioBuffer: Buffer, format: string, durationEstimateMs: number}>}
 */
async function synthesize({ text, language = 'en' }) {
  // ── 1. Try Inworld TTS (English only) ───────────────────────────────────────────
  if (INWORLD_API_KEY) {
    try {
      return await _inworldTTS({ text, language });
    } catch (err) {
      logger.warn('[TTS] Inworld failed, falling back to macOS', { error: err.message });
    }
  }

  // ── 2. macOS native fallback ────────────────────────────────────────────────────────
  logger.warn('[TTS] Inworld unavailable — falling back to native macOS TTS');
  return synthesizeNative({ text, language });
}

/**
 * Inworld TTS — blocking synthesis returning mp3 buffer.
 */
async function _inworldTTS({ text, language = 'en' }) {
  // Hard cap: Inworld allows up to 2000 chars per request
  const cappedText = text.length > 1900 ? text.substring(0, 1900) + '...' : text;

  logger.info('[TTS] Synthesizing text (Inworld)', {
    language,
    voiceId: INWORLD_VOICE_ID,
    textPreview: cappedText.substring(0, 80),
  });

  const response = await axios.post(
    INWORLD_TTS_URL,
    {
      text: cappedText,
      voiceId: INWORLD_VOICE_ID,
      modelId: INWORLD_MODEL_ID,
    },
    {
      headers: {
        'Authorization': `Basic ${INWORLD_API_KEY}`,
        'Content-Type': 'application/json',
      },
      timeout: 15000,
    }
  );

  const audioContent = response.data?.audioContent;
  if (!audioContent) throw new Error('Inworld returned no audioContent');

  const audioBuffer = Buffer.from(audioContent, 'base64');
  const durationEstimateMs = Math.ceil((cappedText.length / 15) * 1000);
  logger.info('[TTS] Inworld audio generated', { bytes: audioBuffer.length, durationEstimateMs });
  return { audioBuffer, format: 'mp3', durationEstimateMs };
}

/**
 * Native macOS TTS fallback using the `say` command.
 * Supports limited language/voice selection via macOS voices.
 */
async function synthesizeNative({ text, language = 'en' }) {
  const NATIVE_VOICES = {
    en: 'Samantha',
    zh: 'Tingting',
    es: 'Monica',
    fr: 'Thomas',
    pt: 'Luciana',
    ja: 'Kyoko',
    ko: 'Yuna',
    de: 'Anna',
    it: 'Alice',
    ru: 'Milena',
    ar: 'Maged',
    hi: 'Lekha',
  };

  const voice = NATIVE_VOICES[language] || 'Samantha';
  const tmpFile = path.join(os.tmpdir(), `thinkdrop-tts-${Date.now()}.aiff`);

  logger.info('[TTS] Using native macOS TTS', { voice, language });

  const mp3File = tmpFile.replace('.aiff', '.mp3');
  return new Promise((resolve, reject) => {
    // Use '--' separator to guard against special characters in text being interpreted as flags
    const proc = spawn('say', ['-v', voice, '-o', tmpFile, '--', text]);
    proc.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`macOS say command failed with code ${code}`));
        return;
      }
      try {
        // Try MP3 via ffmpeg first (better browser compatibility)
        let audioBuffer;
        let format = 'aiff';
        try {
          execSync(`ffmpeg -y -i "${tmpFile}" -codec:a libmp3lame -qscale:a 4 "${mp3File}" 2>/dev/null`, { timeout: 10000 });
          if (fs.existsSync(mp3File)) {
            audioBuffer = fs.readFileSync(mp3File);
            format = 'mp3';
          }
        } catch (_) {
          // ffmpeg unavailable — fall back to raw AIFF
        }
        if (!audioBuffer) {
          audioBuffer = fs.readFileSync(tmpFile);
          format = 'aiff';
        }
        try { fs.unlinkSync(tmpFile); } catch (_) {}
        try { fs.unlinkSync(mp3File); } catch (_) {}
        resolve({ audioBuffer, format, durationEstimateMs: text.split(/\s+/).length * 460 });
      } catch (err) {
        reject(err);
      }
    });
    proc.on('error', reject);
  });
}

/**
 * Play audio directly on macOS (for testing/dev — production plays via Electron renderer).
 */
async function playAudioBuffer({ audioBuffer, format = 'mp3' }) {
  const tmpFile = path.join(os.tmpdir(), `thinkdrop-play-${Date.now()}.${format}`);
  fs.writeFileSync(tmpFile, audioBuffer);
  try {
    execSync(`afplay "${tmpFile}"`, { timeout: 60000 });
  } finally {
    try { fs.unlinkSync(tmpFile); } catch (_) {}
  }
}

module.exports = { synthesize, synthesizeNative, playAudioBuffer };
