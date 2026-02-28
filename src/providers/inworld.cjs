'use strict';

/**
 * ElevenLabs/Inworld voice provider
 *
 * Wraps the existing Inworld (primary) → ElevenLabs (fallback) → macOS say (last resort) TTS
 * pipeline combined with Groq Whisper → ElevenLabs Scribe STT.
 *
 * This is the current default behaviour extracted into a provider module
 * so it can participate in the fallback chain alongside Hume EVI.
 *
 * No new logic — just delegation to the existing stt.cjs + elevenlabs-tts.cjs modules.
 */

const stt = require('../stt.cjs');
const tts = require('../inworld-tts.cjs');
const logger = require('../logger.cjs');

function isAvailable() {
  // Available if either Groq or ElevenLabs STT keys are set
  return !!(process.env.GROQ_API_KEY || process.env.ELEVENLABS_API_KEY);
}

/**
 * Transcribe base64 audio → text.
 * Delegates to Groq Whisper → ElevenLabs Scribe fallback chain.
 */
async function transcribe({ audioBase64, format = 'wav', languageHint = null }) {
  return stt.transcribeBase64({ audioBase64, format, languageHint });
}

/**
 * Synthesize text → audio buffer.
 * Delegates to Inworld → ElevenLabs → macOS say fallback chain.
 */
async function synthesize({ text, language = 'en', voiceId, stability, similarity, style }) {
  logger.info('[ElevenLabsProvider] synthesize', { textLen: text.length, language });
  return tts.synthesize({ text, language, voiceId, stability, similarity, style });
}

module.exports = { isAvailable, transcribe, synthesize };
