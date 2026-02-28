'use strict';

/**
 * macOS Native voice provider
 *
 * STT:  Not available natively from Node (use Groq or ElevenLabs for STT).
 *       This provider is TTS-only — pair with groq.cjs for full STT+TTS.
 * TTS:  macOS `say` command — completely offline, no API key, zero cost.
 *       Output is AIFF → converted to MP3 via ffmpeg or raw AIFF buffer.
 *
 * Voice selection:
 *   Default voice is whatever the user has set in System Preferences > Accessibility.
 *   Override with MACOS_SAY_VOICE env var (e.g. "Samantha", "Alex", "Daniel").
 */

const { execFile, execSync } = require('child_process');
const fs   = require('fs');
const path = require('path');
const os   = require('os');
const logger = require('../logger.cjs');

const MACOS_SAY_VOICE = process.env.MACOS_SAY_VOICE || '';

// Language → voice map (macOS built-in voices)
const LANG_VOICE_MAP = {
  en:  MACOS_SAY_VOICE || 'Samantha',
  es:  'Monica',
  fr:  'Thomas',
  de:  'Anna',
  ja:  'Kyoko',
  zh:  'Ting-Ting',
  ko:  'Yuna',
  pt:  'Joana',
  it:  'Alice',
  ru:  'Milena',
  ar:  'Tarik',
  hi:  'Lekha',
};

function isAvailable() {
  return process.platform === 'darwin';
}

/**
 * Synthesize text → audio buffer using macOS `say`.
 * Returns { audioBuffer, format: 'mp3' | 'aiff', durationEstimateMs }
 */
async function synthesize({ text, language = 'en' }) {
  if (!isAvailable()) {
    throw new Error('macOS native TTS is only available on macOS');
  }

  const langKey = (language || 'en').toLowerCase().substring(0, 2);
  const voice = LANG_VOICE_MAP[langKey] || MACOS_SAY_VOICE || 'Samantha';
  const tmpAiff = path.join(os.tmpdir(), `thinkdrop_say_${Date.now()}.aiff`);
  const tmpMp3  = path.join(os.tmpdir(), `thinkdrop_say_${Date.now()}.mp3`);

  logger.info('[macOSNative] synthesize', { textLen: text.length, voice, language });

  // Generate AIFF via `say`
  await new Promise((resolve, reject) => {
    const args = ['-v', voice, '-o', tmpAiff, '--', text];
    execFile('say', args, { timeout: 15000 }, (err) => {
      if (err) reject(new Error(`say command failed: ${err.message}`));
      else resolve();
    });
  });

  // Try to convert to MP3 via ffmpeg (better compatibility)
  let audioBuffer;
  let format = 'aiff';
  try {
    execSync(`ffmpeg -y -i "${tmpAiff}" -codec:a libmp3lame -qscale:a 4 "${tmpMp3}" 2>/dev/null`, { timeout: 10000 });
    if (fs.existsSync(tmpMp3)) {
      audioBuffer = fs.readFileSync(tmpMp3);
      format = 'mp3';
    }
  } catch (_) {
    // ffmpeg not available — use raw AIFF
    logger.info('[macOSNative] ffmpeg not available — using AIFF format');
  }

  if (!audioBuffer) {
    audioBuffer = fs.readFileSync(tmpAiff);
    format = 'aiff';
  }

  // Cleanup
  try { fs.unlinkSync(tmpAiff); } catch (_) {}
  try { fs.unlinkSync(tmpMp3); } catch (_) {}

  // Rough duration estimate: ~130 words per minute
  const wordCount = text.split(/\s+/).length;
  const durationEstimateMs = Math.round((wordCount / 130) * 60 * 1000);

  logger.info('[macOSNative] synthesize done', { bytes: audioBuffer.length, format, durationEstimateMs });

  return { audioBuffer, format, durationEstimateMs };
}

module.exports = { isAvailable, synthesize };
