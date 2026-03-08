'use strict';

/**
 * Audio Emotion Detector
 *
 * Analyzes raw PCM energy characteristics from a decoded audio buffer to infer
 * the speaker's emotional state. Used to inject emotion context into the LLM
 * system prompt so the LLM can wrap its TTS response with appropriate Resemble AI
 * emotion/paralinguistic tags ([happy], [excited], [angry], [sad], [empathetic], etc.)
 *
 * Detection approach:
 *   - RMS energy → overall loudness
 *   - Peak-to-RMS ratio (crest factor) → dynamic range (yelling = low crest, compressed signal)
 *   - Short-window RMS variance → speaking pace / agitation
 *   - Base64 audio length heuristic (fast shortcut before full decode)
 *
 * Returns: { emotion: string, intensity: 'low'|'medium'|'high', tags: string[] }
 *
 * Emotion → Resemble tag mapping:
 *   angry/yelling  → [angry]
 *   excited        → [excited]
 *   happy          → [happy]
 *   sad            → [sad]
 *   empathetic     → [empathetic]
 *   neutral        → (no tag)
 */

const logger = require('./logger.cjs');

// ── Thresholds (tuned for 16-bit PCM / WAV from browser VAD) ─────────────────
const ENERGY_HIGH   = 0.25;   // RMS > this → loud / yelling
const ENERGY_LOW    = 0.04;   // RMS < this → quiet / soft / possibly sad
const CREST_YELL    = 3.5;    // peak/RMS < this AND loud → likely yelling (compressed)
const VARIANCE_HIGH = 0.015;  // RMS variance > this → agitated / excited / angry
const VARIANCE_LOW  = 0.003;  // RMS variance < this → flat / sad / neutral

/**
 * Analyze a base64-encoded audio buffer and return detected emotion.
 *
 * @param {string} audioBase64 - Base64 encoded audio (WAV/WebM/MP4)
 * @param {{ format?: string, transcript?: string }} opts
 * @returns {{ emotion: string, intensity: 'low'|'medium'|'high', tags: string[], context: string }}
 */
function detectEmotion(audioBase64, opts = {}) {
  try {
    if (!audioBase64) return _neutral();

    const { transcript = '' } = opts;

    // ── Fast path: text-based cues (exclamation marks, all-caps, question cascade) ──
    const textEmotion = _detectTextCues(transcript);

    // ── Audio energy analysis ──────────────────────────────────────────────────
    const buffer = Buffer.from(audioBase64, 'base64');

    // Skip non-PCM headers — WAV has a 44-byte header, WebM/MP4 are compressed.
    // For compressed formats we can still do a coarse energy estimate from the
    // raw byte values (works well enough for loudness detection).
    const startOffset = _detectWavHeader(buffer) ? 44 : 0;
    const samples = _extractSamples(buffer, startOffset);

    if (samples.length < 100) {
      // Too short to analyze — rely on text cues only
      return textEmotion || _neutral();
    }

    const rms = _computeRMS(samples);
    const peak = _computePeak(samples);
    const crest = peak > 0 ? peak / (rms + 1e-10) : 10;
    const variance = _computeWindowedVariance(samples);

    logger.debug('[AudioEmotion] Analysis', {
      rms: rms.toFixed(4),
      peak: peak.toFixed(4),
      crest: crest.toFixed(2),
      variance: variance.toFixed(5),
      transcriptLen: transcript.length,
    });

    // ── Decision tree ──────────────────────────────────────────────────────────

    // Yelling / angry — loud AND compressed (low crest factor) AND high variance
    if (rms > ENERGY_HIGH && crest < CREST_YELL && variance > VARIANCE_HIGH) {
      return _result('angry', 'high', ['[angry]'],
        'The user appears to be speaking loudly or with frustration. Mirror their intensity with direct, confident energy — no hedging.');
    }

    // Loud and excited (high crest — dynamic, not compressed yelling)
    if (rms > ENERGY_HIGH && crest >= CREST_YELL && variance > VARIANCE_HIGH) {
      return _result('excited', 'high', ['[excited]'],
        'The user sounds energized and enthusiastic. Match that energy — be upbeat and engaged.');
    }

    // Loud and somewhat agitated (medium variance)
    if (rms > ENERGY_HIGH && variance >= VARIANCE_LOW) {
      // Combine with text cues if available
      if (textEmotion && textEmotion.emotion === 'angry') {
        return _result('angry', 'medium', ['[angry]'],
          'The user sounds somewhat raised or tense. Stay calm and direct.');
      }
      return _result('excited', 'medium', ['[excited]'],
        'The user sounds expressive and engaged. Be responsive and energetic.');
    }

    // Soft and flat — possible sadness or tiredness
    if (rms < ENERGY_LOW && variance < VARIANCE_LOW) {
      if (textEmotion && (textEmotion.emotion === 'sad' || textEmotion.emotion === 'empathetic')) {
        return textEmotion;
      }
      return _result('empathetic', 'low', ['[empathetic]'],
        'The user is speaking softly. Be gentle, warm, and supportive.');
    }

    // Normal energy with high variance — enthusiastic normal conversation
    if (variance > VARIANCE_HIGH) {
      return _result('happy', 'medium', ['[happy]'],
        'The user sounds upbeat and conversational. Be warm and friendly.');
    }

    // Text cues win if audio is neutral
    if (textEmotion) return textEmotion;

    return _neutral();
  } catch (err) {
    logger.debug('[AudioEmotion] Analysis failed (non-critical)', { error: err.message });
    return _neutral();
  }
}

// ── Text cue analysis ──────────────────────────────────────────────────────────

function _detectTextCues(transcript) {
  if (!transcript || transcript.length < 2) return null;

  const t = transcript.trim();

  // Multiple exclamation marks → excited or angry
  const exclaims = (t.match(/!/g) || []).length;
  if (exclaims >= 2) {
    return _result('excited', 'medium', ['[excited]'],
      'The user seems excited (multiple exclamation marks). Match that enthusiasm.');
  }

  // ALL CAPS words → yelling or strong emphasis
  const words = t.split(/\s+/);
  const capsWords = words.filter(w => w.length > 2 && w === w.toUpperCase() && /[A-Z]/.test(w));
  if (capsWords.length >= 2) {
    return _result('angry', 'medium', ['[angry]'],
      'The user is emphasizing strongly (ALL CAPS). Be direct and match their intensity without escalating.');
  }

  // Sad/frustration indicators
  if (/\b(ugh|argh|frustrated|annoyed|upset|so (tired|done|over)|can'?t believe|seriously\?|wtf|this is (crazy|ridiculous|stupid|awful|terrible))\b/i.test(t)) {
    return _result('empathetic', 'medium', ['[empathetic]'],
      'The user sounds frustrated or upset. Lead with empathy before being practical.');
  }

  // Happy/positive indicators
  if (/\b(awesome|amazing|fantastic|love it|great|wonderful|excellent|perfect|so (good|happy|excited)|yes!|heck yeah|let'?s go)\b/i.test(t)) {
    return _result('happy', 'medium', ['[happy]'],
      'The user sounds happy and positive. Be warm and celebratory.');
  }

  return null;
}

// ── PCM sample extraction ──────────────────────────────────────────────────────

function _detectWavHeader(buffer) {
  if (buffer.length < 12) return false;
  return buffer.slice(0, 4).toString('ascii') === 'RIFF';
}

function _extractSamples(buffer, startOffset = 0) {
  const samples = [];
  // Read as signed 16-bit LE integers (most common WAV format from browsers)
  for (let i = startOffset; i < buffer.length - 1; i += 2) {
    const sample = buffer.readInt16LE(i) / 32768.0; // normalize to [-1, 1]
    samples.push(sample);
  }
  return samples;
}

// ── DSP helpers ────────────────────────────────────────────────────────────────

function _computeRMS(samples) {
  if (!samples.length) return 0;
  const sumSq = samples.reduce((acc, s) => acc + s * s, 0);
  return Math.sqrt(sumSq / samples.length);
}

function _computePeak(samples) {
  return samples.reduce((max, s) => Math.max(max, Math.abs(s)), 0);
}

/**
 * Compute variance of RMS values across short windows (~20ms at 16kHz = 320 samples).
 * High variance = lots of dynamic change = excited/agitated speech.
 */
function _computeWindowedVariance(samples, windowSize = 320) {
  const rmsValues = [];
  for (let i = 0; i < samples.length - windowSize; i += windowSize) {
    const window = samples.slice(i, i + windowSize);
    rmsValues.push(_computeRMS(window));
  }
  if (rmsValues.length < 3) return 0;
  const mean = rmsValues.reduce((a, b) => a + b, 0) / rmsValues.length;
  const variance = rmsValues.reduce((acc, v) => acc + (v - mean) ** 2, 0) / rmsValues.length;
  return variance;
}

// ── Result builders ────────────────────────────────────────────────────────────

function _result(emotion, intensity, tags, context) {
  return { emotion, intensity, tags, context };
}

function _neutral() {
  return { emotion: 'neutral', intensity: 'low', tags: [], context: '' };
}

module.exports = { detectEmotion };
