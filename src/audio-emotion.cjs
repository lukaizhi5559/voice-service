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
 * @returns {{ emotion: string, intensity: 'low'|'medium'|'high', tags: string[], context: string, speakerProfile: { gender: 'male'|'female'|'child'|'unknown', ageGroup: 'child'|'adult', volume: 'loud'|'normal'|'quiet'|'whisper', address: 'sir'|"ma'am"|'friend'|'' } }}
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
    const speakerProfile = _estimateSpeakerProfile(samples, 16000, rms);

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
      return _withProfile(_result('angry', 'high', ['[angry]'],
        'The user appears to be speaking loudly or with frustration. Mirror their intensity with direct, confident energy — no hedging.'), speakerProfile);
    }

    // Loud and excited (high crest — dynamic, not compressed yelling)
    if (rms > ENERGY_HIGH && crest >= CREST_YELL && variance > VARIANCE_HIGH) {
      return _withProfile(_result('excited', 'high', ['[excited]'],
        'The user sounds energized and enthusiastic. Match that energy — be upbeat and engaged.'), speakerProfile);
    }

    // Loud and somewhat agitated (medium variance)
    if (rms > ENERGY_HIGH && variance >= VARIANCE_LOW) {
      if (textEmotion && textEmotion.emotion === 'angry') {
        return _withProfile(_result('angry', 'medium', ['[angry]'],
          'The user sounds somewhat raised or tense. Stay calm and direct.'), speakerProfile);
      }
      return _withProfile(_result('excited', 'medium', ['[excited]'],
        'The user sounds expressive and engaged. Be responsive and energetic.'), speakerProfile);
    }

    // Soft and flat — possible sadness or tiredness
    if (rms < ENERGY_LOW && variance < VARIANCE_LOW) {
      if (textEmotion && (textEmotion.emotion === 'sad' || textEmotion.emotion === 'empathetic')) {
        return _withProfile(textEmotion, speakerProfile);
      }
      return _withProfile(_result('empathetic', 'low', ['[empathetic]'],
        'The user is speaking softly. Be gentle, warm, and supportive.'), speakerProfile);
    }

    // Normal energy with high variance — enthusiastic normal conversation
    if (variance > VARIANCE_HIGH) {
      return _withProfile(_result('happy', 'medium', ['[happy]'],
        'The user sounds upbeat and conversational. Be warm and friendly.'), speakerProfile);
    }

    // Text cues win if audio is neutral
    if (textEmotion) return _withProfile(textEmotion, speakerProfile);

    return _withProfile(_neutral(), speakerProfile);
  } catch (err) {
    logger.debug('[AudioEmotion] Analysis failed (non-critical)', { error: err.message });
    return _neutral();
  }
}

/**
 * Estimate speaker profile from PCM samples.
 *
 * Technique: Zero-Crossing Rate (ZCR) as an F0 proxy.
 * Zero-crossings per second ≈ 2× fundamental frequency for voiced speech.
 *
 *   Child   : ZCR-based F0 ≈ 250-500 Hz  → ZCR > ~500 crossings/sec
 *   Female  : ZCR-based F0 ≈ 165-255 Hz  → ZCR ~330-510
 *   Male    : ZCR-based F0 ≈  85-180 Hz  → ZCR ~170-360
 *
 * Ranges overlap, so we use median ZCR across voiced frames (RMS > noise floor)
 * to get a more stable estimate. This is a heuristic — it works well for clear
 * single-speaker audio at 16kHz.
 *
 * Volume detection: based on RMS vs thresholds (already computed in detectEmotion).
 *
 * @param {Float32Array|number[]} samples  - Normalized PCM [-1, 1]
 * @param {number} sampleRate              - Sample rate in Hz (default 16000)
 * @param {number} rms                     - Already-computed RMS
 * @returns {{ gender: string, ageGroup: string, volume: string, address: string, f0Estimate: number }}
 */
function _estimateSpeakerProfile(samples, sampleRate, rms) {
  sampleRate = sampleRate || 16000;

  // ── Volume classification ────────────────────────────────────────────────────
  let volume;
  if (rms > ENERGY_HIGH) {
    volume = 'loud';
  } else if (rms < 0.01) {
    volume = 'whisper';
  } else if (rms < ENERGY_LOW) {
    volume = 'quiet';
  } else {
    volume = 'normal';
  }

  // ── Zero-crossing rate on voiced frames ─────────────────────────────────────
  const FRAME_SIZE   = 512;          // ~32ms at 16kHz
  const NOISE_FLOOR  = 0.015;        // minimum RMS to consider a frame voiced
  const ZCR_CHILD    = 500;          // crossings/sec above this → child
  const ZCR_FEMALE_LO = 300;         // female range lower bound
  const ZCR_MALE_HI  = 380;          // male/female overlap upper bound for male majority

  const frameZcrs = [];

  for (let i = 0; i + FRAME_SIZE < samples.length; i += FRAME_SIZE) {
    // Only analyze voiced frames
    let frameSumSq = 0;
    for (let j = i; j < i + FRAME_SIZE; j++) frameSumSq += samples[j] * samples[j];
    const frameRms = Math.sqrt(frameSumSq / FRAME_SIZE);
    if (frameRms < NOISE_FLOOR) continue;

    // Count zero crossings in this frame
    let crossings = 0;
    for (let j = i + 1; j < i + FRAME_SIZE; j++) {
      if ((samples[j] >= 0 && samples[j - 1] < 0) || (samples[j] < 0 && samples[j - 1] >= 0)) {
        crossings++;
      }
    }
    // ZCR in crossings per second
    const frameDurationSec = FRAME_SIZE / sampleRate;
    frameZcrs.push(crossings / frameDurationSec);
  }

  if (frameZcrs.length === 0) {
    return { gender: 'unknown', ageGroup: 'adult', volume, address: '', f0Estimate: 0 };
  }

  // Use median ZCR across voiced frames (robust against outlier frames)
  frameZcrs.sort((a, b) => a - b);
  const mid = Math.floor(frameZcrs.length / 2);
  const medianZcr = frameZcrs.length % 2 !== 0
    ? frameZcrs[mid]
    : (frameZcrs[mid - 1] + frameZcrs[mid]) / 2;

  // F0 estimate: ZCR ≈ 2 * F0 for voiced speech
  const f0Estimate = medianZcr / 2;

  // ── Classification ────────────────────────────────────────────────────────────
  let gender, ageGroup, address;

  if (medianZcr > ZCR_CHILD) {
    // High ZCR → child (or very high female — check RMS energy too)
    // Children also tend to have more variable energy; default to child
    gender   = 'child';
    ageGroup = 'child';
    address  = 'friend';
  } else if (medianZcr >= ZCR_FEMALE_LO) {
    // Mid-to-high ZCR range → likely female adult
    gender   = 'female';
    ageGroup = 'adult';
    address  = "ma'am";
  } else if (medianZcr < ZCR_FEMALE_LO && medianZcr > 0) {
    // Lower ZCR → likely male adult
    gender   = 'male';
    ageGroup = 'adult';
    address  = 'sir';
  } else {
    gender   = 'unknown';
    ageGroup = 'adult';
    address  = '';
  }

  logger.debug('[AudioEmotion] SpeakerProfile', {
    medianZcr: medianZcr.toFixed(1),
    f0Estimate: f0Estimate.toFixed(1),
    gender, ageGroup, volume,
    voicedFrames: frameZcrs.length,
  });

  return { gender, ageGroup, volume, address, f0Estimate };
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

const _defaultProfile = { gender: 'unknown', ageGroup: 'adult', volume: 'normal', address: '', f0Estimate: 0 };

function _result(emotion, intensity, tags, context) {
  return { emotion, intensity, tags, context, speakerProfile: _defaultProfile };
}

function _neutral() {
  return { emotion: 'neutral', intensity: 'low', tags: [], context: '', speakerProfile: _defaultProfile };
}

function _withProfile(result, profile) {
  return Object.assign({}, result, { speakerProfile: profile || _defaultProfile });
}

module.exports = { detectEmotion, _estimateSpeakerProfile };
