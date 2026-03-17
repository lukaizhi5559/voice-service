'use strict';

/**
 * voice-fingerprint.cjs
 *
 * Speaker identification via spectral feature vectors stored in DuckDB.
 *
 * FEATURE VECTOR (5 dimensions):
 *   [0] zcr_median        — zero-crossing rate median across voiced frames
 *   [1] spectral_centroid — brightness of the voice (Hz estimate)
 *   [2] rms_mean          — average loudness
 *   [3] rms_variance      — loudness variability (speaking dynamism)
 *   [4] f0_estimate       — fundamental frequency estimate (zcr_median / 2)
 *
 * MATCHING:
 *   Cosine similarity between the incoming feature vector and stored fingerprints.
 *   Threshold: >= 0.92 → confirmed match (same speaker)
 *              0.80–0.92 → probable match (possible same speaker)
 *              < 0.80 → unknown speaker
 *
 * ENROLLMENT:
 *   First utterance from a speaker creates a new fingerprint row.
 *   Subsequent utterances update the running average (exponential moving average,
 *   alpha = 0.1) so the fingerprint drifts slowly toward current voice characteristics
 *   (e.g. aging, microphone change, illness).
 *
 * MULTI-USER:
 *   Each speaker gets a unique speaker_id (UUID). ThinkDrop stores speaker_name
 *   (e.g. "Primary User", "Guest", "Child — Emma"). The pipeline injects the
 *   matched speaker name into the personality overlay.
 *
 * LIMITATIONS (honest):
 *   - ZCR/spectral features are ~70-80% accurate for well-separated voices.
 *   - Same-gender adults with similar voices can collide.
 *   - Noisy/distant microphone degrades matching significantly.
 *   - This is NOT a cryptographic identity — it's a best-effort heuristic.
 */

const http   = require('http');
const crypto = require('crypto');
const logger = require('./logger.cjs');

const MEMORY_SERVICE_PORT = parseInt(process.env.MEMORY_SERVICE_PORT || '3001', 10);
const MEM_API_KEY = process.env.MCP_USER_MEMORY_API_KEY || process.env.USER_MEMORY_API_KEY || process.env.MCP_API_KEY || '';

// ── Matching thresholds ────────────────────────────────────────────────────────
const MATCH_CONFIRMED  = 0.92;   // cosine similarity — high confidence same speaker
const MATCH_PROBABLE   = 0.80;   // probable match — same speaker likely
const EMA_ALPHA        = 0.10;   // exponential moving average weight for updates

// ── HTTP helper ────────────────────────────────────────────────────────────────

function memPost(action, payload) {
  return new Promise((resolve) => {
    const envelope = {
      version: 'mcp.v1',
      service: 'user-memory',
      action,
      payload: payload || {},
      requestId: 'vfp_' + Date.now(),
    };
    const body = JSON.stringify(envelope);
    const headers = { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) };
    if (MEM_API_KEY) headers['Authorization'] = 'Bearer ' + MEM_API_KEY;
    const req = http.request({
      hostname: '127.0.0.1', port: MEMORY_SERVICE_PORT,
      path: '/' + action, method: 'POST', headers, timeout: 6000,
    }, (res) => {
      let raw = '';
      res.on('data', c => { raw += c; });
      res.on('end', () => { try { resolve(JSON.parse(raw)); } catch (_) { resolve(null); } });
    });
    req.on('error', () => resolve(null));
    req.on('timeout', () => { req.destroy(); resolve(null); });
    req.write(body);
    req.end();
  });
}

// ── Feature extraction ─────────────────────────────────────────────────────────

/**
 * Extract a 5-dim feature vector from PCM samples.
 * Uses the same ZCR approach as _estimateSpeakerProfile but also computes
 * spectral centroid (brightness) as an additional discriminating feature.
 *
 * @param {number[]} samples   Normalized PCM [-1, 1]
 * @param {number}   sampleRate Default 16000
 * @returns {number[]} [zcr_median, spectral_centroid, rms_mean, rms_variance, f0_estimate]
 */
function extractFeatures(samples, sampleRate) {
  sampleRate = sampleRate || 16000;

  if (!samples || samples.length < 512) {
    return null;
  }

  const FRAME_SIZE  = 512;
  const NOISE_FLOOR = 0.015;

  const frameZcrs         = [];
  const frameCentroids    = [];
  const frameRms          = [];

  for (let i = 0; i + FRAME_SIZE < samples.length; i += FRAME_SIZE) {
    // RMS for this frame
    let sumSq = 0;
    for (let j = i; j < i + FRAME_SIZE; j++) sumSq += samples[j] * samples[j];
    const rms = Math.sqrt(sumSq / FRAME_SIZE);
    if (rms < NOISE_FLOOR) continue; // skip silent frames

    frameRms.push(rms);

    // Zero-crossing rate
    let crossings = 0;
    for (let j = i + 1; j < i + FRAME_SIZE; j++) {
      if ((samples[j] >= 0) !== (samples[j - 1] >= 0)) crossings++;
    }
    frameZcrs.push((crossings / FRAME_SIZE) * sampleRate);

    // Spectral centroid approximation via weighted sum of |sample| * position
    // This is a time-domain proxy, not a true FFT centroid, but it's fast and
    // captures brightness reasonably well for discrimination purposes.
    let weightedSum = 0;
    let totalMag    = 0;
    for (let j = i; j < i + FRAME_SIZE; j++) {
      const mag = Math.abs(samples[j]);
      weightedSum += mag * (j - i);
      totalMag    += mag;
    }
    frameCentroids.push(totalMag > 0 ? (weightedSum / totalMag) / FRAME_SIZE : 0.5);
  }

  if (frameZcrs.length < 3) return null;

  // Median ZCR
  const sortedZcr = [...frameZcrs].sort((a, b) => a - b);
  const mid = Math.floor(sortedZcr.length / 2);
  const zcrMedian = sortedZcr.length % 2 !== 0
    ? sortedZcr[mid]
    : (sortedZcr[mid - 1] + sortedZcr[mid]) / 2;

  // Mean spectral centroid
  const centroidMean = frameCentroids.reduce((a, b) => a + b, 0) / frameCentroids.length;

  // RMS mean and variance
  const rmsMean = frameRms.reduce((a, b) => a + b, 0) / frameRms.length;
  const rmsVariance = frameRms.reduce((acc, v) => acc + (v - rmsMean) ** 2, 0) / frameRms.length;

  const f0Estimate = zcrMedian / 2;

  return [zcrMedian, centroidMean, rmsMean, rmsVariance, f0Estimate];
}

// ── Vector math ────────────────────────────────────────────────────────────────

function cosineSimilarity(a, b) {
  if (!a || !b || a.length !== b.length) return 0;
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot   += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function emaUpdate(stored, incoming, alpha) {
  return stored.map((v, i) => v * (1 - alpha) + (incoming[i] || 0) * alpha);
}

// ── Public API ─────────────────────────────────────────────────────────────────

/**
 * Match an audio buffer against all stored fingerprints.
 * Returns the best-matching speaker or null if no match above threshold.
 *
 * @param {number[]} samples    PCM samples [-1, 1]
 * @param {number}   sampleRate
 * @returns {Promise<{ speaker_id, speaker_name, gender, age_group, confidence, match_level } | null>}
 */
async function matchSpeaker(samples, sampleRate) {
  try {
    const features = extractFeatures(samples, sampleRate);
    if (!features) return null;

    const res = await memPost('fingerprint.list', {});
    const fingerprints = (res && res.data && res.data.fingerprints) ? res.data.fingerprints : [];
    if (fingerprints.length === 0) return null;

    let bestMatch = null;
    let bestScore = 0;

    for (const fp of fingerprints) {
      let stored;
      try { stored = JSON.parse(fp.features_json); } catch (_) { continue; }
      const score = cosineSimilarity(features, stored);
      if (score > bestScore) {
        bestScore = score;
        bestMatch = fp;
      }
    }

    if (!bestMatch || bestScore < MATCH_PROBABLE) return null;

    const matchLevel = bestScore >= MATCH_CONFIRMED ? 'confirmed' : 'probable';
    logger.info('[VoiceFingerprint] Match', {
      speaker: bestMatch.speaker_name,
      score: bestScore.toFixed(3),
      level: matchLevel,
    });

    return {
      speaker_id:   bestMatch.speaker_id,
      speaker_name: bestMatch.speaker_name,
      gender:       bestMatch.gender,
      age_group:    bestMatch.age_group,
      confidence:   bestScore,
      match_level:  matchLevel,
      angry_count:  bestMatch.angry_count || 0,
      loud_count:   bestMatch.loud_count  || 0,
      whisper_count: bestMatch.whisper_count || 0,
    };
  } catch (e) {
    logger.warn('[VoiceFingerprint] matchSpeaker failed', { error: e.message });
    return null;
  }
}

/**
 * Enroll or update a speaker fingerprint.
 * If speaker_id is null, creates a new speaker. Otherwise updates the existing one.
 * Uses EMA to blend new features into the running average.
 *
 * @param {number[]} samples
 * @param {number}   sampleRate
 * @param {string|null} speakerId     null = auto-assign new UUID
 * @param {string}   speakerName      e.g. 'Primary User', 'Guest'
 * @param {string}   gender           'male'|'female'|'child'|'unknown'
 * @param {string}   ageGroup         'adult'|'child'
 * @param {Object}   voicePatternUpdate  { angry?: boolean, loud?: boolean, whisper?: boolean }
 * @returns {Promise<{ speaker_id, enrolled: boolean, updated: boolean }>}
 */
async function enrollOrUpdate(samples, sampleRate, speakerId, speakerName, gender, ageGroup, voicePatternUpdate) {
  try {
    const features = extractFeatures(samples, sampleRate);
    if (!features) return { speaker_id: null, enrolled: false, updated: false };

    const sid = speakerId || ('spk_' + crypto.randomUUID().replace(/-/g, '').slice(0, 16));

    const res = await memPost('fingerprint.enroll', {
      speaker_id:    sid,
      speaker_name:  speakerName || 'Primary User',
      features:      features,
      gender:        gender || 'unknown',
      age_group:     ageGroup || 'adult',
      ema_alpha:     EMA_ALPHA,
      voice_pattern: voicePatternUpdate || {},
    });

    return {
      speaker_id: sid,
      enrolled:   !!(res && res.data && res.data.enrolled),
      updated:    !!(res && res.data && res.data.updated),
    };
  } catch (e) {
    logger.warn('[VoiceFingerprint] enrollOrUpdate failed', { error: e.message });
    return { speaker_id: null, enrolled: false, updated: false };
  }
}

/**
 * Extract features from a base64-encoded audio buffer.
 * Helper that decodes base64 → PCM samples first.
 *
 * @param {string} audioBase64
 * @param {number} sampleRate
 * @returns {number[]|null}
 */
function extractFeaturesFromBase64(audioBase64, sampleRate) {
  try {
    if (!audioBase64) return null;
    const buffer = Buffer.from(audioBase64, 'base64');
    // Skip WAV header if present
    const isWav = buffer.length >= 4 && buffer.slice(0, 4).toString('ascii') === 'RIFF';
    const startOffset = isWav ? 44 : 0;
    const samples = [];
    for (let i = startOffset; i < buffer.length - 1; i += 2) {
      samples.push(buffer.readInt16LE(i) / 32768.0);
    }
    return extractFeatures(samples, sampleRate || 16000);
  } catch (_) {
    return null;
  }
}

module.exports = { matchSpeaker, enrollOrUpdate, extractFeatures, extractFeaturesFromBase64, cosineSimilarity, MATCH_CONFIRMED, MATCH_PROBABLE };
