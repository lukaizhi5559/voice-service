'use strict';

/**
 * Groq voice provider
 *
 * STT:  Groq Whisper large-v3-turbo (fast, nearly free)
 * TTS:  macOS `say` command (offline, no cost, robotic but functional)
 *
 * This is the cheapest functional provider — use as fallback when
 * ElevenLabs/Inworld TTS keys are unavailable or over quota.
 */

const groqStt = require('../groq-stt.cjs');
const macosNative = require('./macos-native.cjs');
const logger = require('../logger.cjs');

function isAvailable() {
  return groqStt.isAvailable();
}

/**
 * Transcribe base64 audio → text using Groq Whisper.
 */
async function transcribe({ audioBase64, format = 'wav', languageHint = null }) {
  const audioBuffer = Buffer.from(audioBase64, 'base64');
  logger.info('[GroqProvider] transcribe', { bytes: audioBuffer.length, format });
  return groqStt.transcribe({ audioBuffer, format, languageHint });
}

/**
 * Synthesize text → audio using macOS `say`.
 * Groq does not offer a TTS API, so we fall back to macOS native.
 */
async function synthesize({ text, language = 'en' }) {
  logger.info('[GroqProvider] synthesize via macOS say', { textLen: text.length });
  return macosNative.synthesize({ text, language });
}

module.exports = { isAvailable, transcribe, synthesize };
