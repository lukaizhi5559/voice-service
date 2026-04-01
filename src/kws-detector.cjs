/**
 * KWS Detector — Audio-domain keyword spotting via sherpa-onnx.
 *
 * WHY THIS EXISTS:
 *   Groq Whisper hallucinates the wake word "Armis" as common English words
 *   ("harvest", "I promise", "permiss") because "Armis" is rare in its training
 *   corpus. Post-STT text matching can never recover from this — the correct
 *   phoneme sequence arrives as entirely different words.
 *
 *   This module runs keyword spotting DIRECTLY on the raw audio PCM samples,
 *   BEFORE calling Groq. It uses a tiny (3.3MB) transducer-based KWS model
 *   from sherpa-onnx (Apache-2.0, fully offline, no API key needed).
 *
 * HOW IT WORKS:
 *   1. Receives WebM/WAV audio buffer from the voice pipeline
 *   2. Converts to 16kHz mono PCM via ffmpeg (always available on macOS)
 *   3. Feeds Float32 PCM samples to sherpa-onnx KeywordSpotter
 *   4. Returns { detected: true, keyword: 'armis' } or { detected: false }
 *
 * INTEGRATION POINT (server.cjs):
 *   In wake-word mode (not pushToTalk), this module acts as a GATE before STT:
 *   - No keyword in audio → skip Groq entirely (saves cost + avoids hallucination)
 *   - Keyword confirmed → full STT pipeline proceeds (wake check already satisfied)
 *
 * GRACEFUL FALLBACK:
 *   If the model files aren't downloaded yet (run `node scripts/download-kws-model.js`),
 *   isAvailable() returns false and the existing text-based wake word detection
 *   in wake-word.cjs continues to work unchanged.
 *
 * SETUP:
 *   node scripts/download-kws-model.js   # one-time download (~3.3MB)
 */

'use strict';

const fs = require('fs');
const path = require('path');
const os = require('os');
const { spawn } = require('child_process');
const logger = require('./logger.cjs');

const MODEL_DIR = path.join(__dirname, '..', 'models', 'kws');
const CONFIG_PATH = path.join(MODEL_DIR, 'model-config.json');

// ── State ─────────────────────────────────────────────────────────────────────
let _spotter = null;
let _available = null;       // null = not checked, true/false after first check
let _initError = null;
let _ffmpegPath = null;

// ── ffmpeg resolution ─────────────────────────────────────────────────────────

/**
 * Resolve the ffmpeg binary path.
 * Checks: env FFMPEG_PATH, then common macOS brew paths, then PATH.
 * @returns {string|null}
 */
function _resolveFfmpeg() {
  if (_ffmpegPath) return _ffmpegPath;

  const candidates = [
    process.env.FFMPEG_PATH,
    '/opt/homebrew/bin/ffmpeg',     // macOS Apple Silicon (brew)
    '/usr/local/bin/ffmpeg',        // macOS Intel (brew)
    '/usr/bin/ffmpeg',              // Linux
  ].filter(Boolean);

  for (const p of candidates) {
    if (fs.existsSync(p)) {
      _ffmpegPath = p;
      return _ffmpegPath;
    }
  }

  // Last resort: check PATH via 'which' (synchronous enough for init)
  try {
    const { execSync } = require('child_process');
    const found = execSync('which ffmpeg 2>/dev/null', { encoding: 'utf8' }).trim();
    if (found) {
      _ffmpegPath = found;
      return _ffmpegPath;
    }
  } catch (_) {}

  return null;
}

// ── Model file resolution ─────────────────────────────────────────────────────

/**
 * Check whether all model files exist.
 * @returns {boolean}
 */
function isAvailable() {
  if (_available !== null) return _available;

  // Quick check: config file written by download script
  if (!fs.existsSync(CONFIG_PATH)) {
    _available = false;
    return false;
  }

  try {
    const config = JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf8'));
    const { encoder, decoder, joiner, tokens, keywords } = config.files || {};
    const allExist = [encoder, decoder, joiner, tokens, keywords].every(
      (f) => f && fs.existsSync(path.join(MODEL_DIR, f))
    );
    _available = allExist;
    if (!allExist) {
      logger.warn('[KWS] Model files incomplete — run: node scripts/download-kws-model.js');
    }
    return _available;
  } catch (err) {
    _available = false;
    logger.warn('[KWS] Could not read model config', { error: err.message });
    return false;
  }
}

// ── Lazy init ─────────────────────────────────────────────────────────────────

/**
 * Initialize sherpa-onnx KeywordSpotter.
 * Called lazily on first use. Thread-safe (Node.js is single-threaded).
 * @returns {boolean} whether init succeeded
 */
function _initialize() {
  if (_spotter) return true;
  if (_initError) return false;

  if (!isAvailable()) return false;

  try {
    const sherpa = require('sherpa-onnx-node');
    const config = JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf8'));
    const f = config.files;

    const kwsConfig = {
      featConfig: { sampleRate: 16000, featureDim: 80 },
      modelConfig: {
        transducer: {
          encoder: path.join(MODEL_DIR, f.encoder),
          decoder: path.join(MODEL_DIR, f.decoder),
          joiner:  path.join(MODEL_DIR, f.joiner),
        },
        tokens:     path.join(MODEL_DIR, f.tokens),
        numThreads: 2,
        debug:      0,
        provider:   'cpu',
        modelType:  'zipformer2',
      },
      maxActivePaths:    8,
      numTrailingBlanks: 2,
      keywordsScore:     3.0,
      keywordsThreshold: 0.05,
      keywordsFile:      path.join(MODEL_DIR, f.keywords),
    };

    _spotter = new sherpa.KeywordSpotter(kwsConfig);

    logger.info('[KWS] KeywordSpotter initialized', {
      model: config.model,
      keywordsFile: f.keywords,
    });
    return true;
  } catch (err) {
    _initError = err;
    logger.error('[KWS] Failed to initialize KeywordSpotter', { error: err.message });
    return false;
  }
}

// ── Audio decoding ─────────────────────────────────────────────────────────────

/**
 * Convert audio buffer (any format) to Float32Array PCM at 16kHz mono via ffmpeg.
 *
 * @param {Buffer} audioBuffer - Raw audio bytes
 * @param {string} inputFormat - 'webm' | 'wav' | 'mp3' | 'ogg' | 'webm;codecs=opus'
 * @returns {Promise<Float32Array>}
 */
function _decodeAudioToFloat32(audioBuffer, inputFormat) {
  return new Promise((resolve, reject) => {
    const ffmpeg = _resolveFfmpeg();
    if (!ffmpeg) {
      return reject(new Error('[KWS] ffmpeg not found — install via: brew install ffmpeg'));
    }

    // Normalise format string (strip codec suffix)
    const fmt = inputFormat.split(';')[0].trim().toLowerCase();

    // Write input to temp file
    const tmpIn  = path.join(os.tmpdir(), `thinkdrop-kws-in-${Date.now()}.${fmt}`);
    const tmpOut = path.join(os.tmpdir(), `thinkdrop-kws-out-${Date.now()}.wav`);
    fs.writeFileSync(tmpIn, audioBuffer);

    // ffmpeg: decode input → 16kHz, mono, 16-bit PCM WAV
    const args = [
      '-y',                 // overwrite output
      '-i', tmpIn,          // input file
      '-ar', '16000',       // resample to 16kHz
      '-ac', '1',           // mono
      '-sample_fmt', 's16', // 16-bit signed PCM
      '-f', 'wav',          // WAV container
      tmpOut,               // output file
    ];

    const proc = spawn(ffmpeg, args, { stdio: ['ignore', 'ignore', 'pipe'] });
    let stderr = '';
    proc.stderr.on('data', (d) => { stderr += d.toString(); });

    const timeout = setTimeout(() => {
      proc.kill();
      _cleanup(tmpIn, tmpOut);
      reject(new Error('[KWS] ffmpeg timeout'));
    }, 8000);

    proc.on('close', (code) => {
      clearTimeout(timeout);
      _cleanup(tmpIn);

      if (code !== 0) {
        _cleanup(tmpOut);
        return reject(new Error(`[KWS] ffmpeg exited ${code}: ${stderr.slice(-200)}`));
      }

      try {
        const wavBuf = fs.readFileSync(tmpOut);
        _cleanup(tmpOut);

        // Parse PCM from WAV container.
        // Standard WAV header: 44 bytes. We skip to the 'data' chunk.
        const offset = _findWavDataOffset(wavBuf);
        const numSamples = (wavBuf.length - offset) / 2; // 16-bit = 2 bytes/sample
        const samples = new Float32Array(numSamples);
        for (let i = 0; i < numSamples; i++) {
          samples[i] = wavBuf.readInt16LE(offset + i * 2) / 32768.0;
        }

        resolve(samples);
      } catch (err) {
        reject(new Error(`[KWS] WAV parse error: ${err.message}`));
      }
    });

    proc.on('error', (err) => {
      clearTimeout(timeout);
      _cleanup(tmpIn, tmpOut);
      reject(new Error(`[KWS] ffmpeg spawn error: ${err.message}`));
    });
  });
}

/**
 * Find the offset of PCM data in a WAV buffer.
 * Scans for the 'data' chunk header rather than assuming a fixed 44-byte header.
 * @param {Buffer} buf
 * @returns {number} byte offset to start of PCM data
 */
function _findWavDataOffset(buf) {
  // Minimum WAV header is 44 bytes; scan up to 256 bytes for the 'data' chunk
  for (let i = 12; i < Math.min(buf.length - 8, 256); i++) {
    if (buf[i] === 0x64 && buf[i+1] === 0x61 && buf[i+2] === 0x74 && buf[i+3] === 0x61) {
      // Found 'data' chunk — data starts 8 bytes later (4 id + 4 size)
      return i + 8;
    }
  }
  // Fallback: standard 44-byte header
  return 44;
}

function _cleanup(...paths) {
  for (const p of paths) {
    try { if (p && fs.existsSync(p)) fs.unlinkSync(p); } catch (_) {}
  }
}

// ── Core detection ────────────────────────────────────────────────────────────

/**
 * Run keyword spotting on audio data.
 *
 * @param {string} audioBase64 - Base64-encoded audio
 * @param {string} format      - Audio format: 'webm' | 'wav' | 'mp3' | etc.
 * @returns {Promise<{ detected: boolean, keyword: string|null, available: boolean, durationMs?: number }>}
 */
async function detectKeyword(audioBase64, format = 'webm') {
  if (!isAvailable() || !_initialize()) {
    return { detected: false, keyword: null, available: false };
  }

  const t0 = Date.now();

  try {
    const audioBuffer = Buffer.from(audioBase64, 'base64');

    // Decode audio to float32 PCM @ 16kHz mono
    let samples;
    try {
      samples = await _decodeAudioToFloat32(audioBuffer, format);
    } catch (decodeErr) {
      logger.warn('[KWS] Audio decode failed — falling back to text detection', {
        error: decodeErr.message,
        format,
      });
      return { detected: false, keyword: null, available: true, decodeError: true };
    }

    if (!samples || samples.length < 1600) {
      // Less than 0.1s of audio — too short for reliable detection
      logger.debug('[KWS] Audio too short for KWS', { samples: samples?.length });
      return { detected: false, keyword: null, available: true };
    }

    // Prepend 800ms of silence so Zipformer2's 640ms left-context window fills
    // before the first speech frame arrives.  Without this, audio that starts
    // immediately with the keyword has no prior frames to attend to and the
    // encoder never accumulates enough score to cross the threshold.
    const LEAD_SILENCE_SAMPLES = 16000 * 0.8; // 800ms @ 16kHz
    const padded = new Float32Array(LEAD_SILENCE_SAMPLES + samples.length);
    padded.set(samples, LEAD_SILENCE_SAMPLES); // leading zeros + original audio
    samples = padded;

    // ── Chunked streaming detection ────────────────────────────────────────────
    // Feed audio in 160ms chunks (2560 samples = 16 frames × 160 samples/frame).
    // This matches the Zipformer2 model's natural decode_chunk_len of 32 frames
    // and ensures the encoder's sliding context window fills properly.
    //
    // WHY chunked, not batch:
    //   Feeding all samples at once via acceptWaveform() + inputFinished() causes
    //   isReady() to return false too quickly — the encoder processes too few
    //   positions and keyword scores never accumulate to the detection threshold.
    //   Chunked feeding triggers proper positional decoding at each 160ms step.
    //
    // Trailing silence:
    //   The VAD in VoiceButton.tsx waits VAD_SILENCE_MS (1200ms) of silence before
    //   sending the audio blob, so the blob already contains ~1.2s of trailing
    //   silence — enough to flush the decoder's 640ms left-context window.
    const CHUNK_SAMPLES = 2560; // 160ms @ 16kHz
    const stream = _spotter.createStream();
    let detectedKeyword = null;

    for (let i = 0; i < samples.length; i += CHUNK_SAMPLES) {
      const chunk = samples.slice(i, Math.min(i + CHUNK_SAMPLES, samples.length));
      stream.acceptWaveform({ sampleRate: 16000, samples: chunk });
      while (_spotter.isReady(stream)) {
        _spotter.decode(stream);
      }
      const midResult = _spotter.getResult(stream);
      if (midResult && midResult.keyword && midResult.keyword.trim()) {
        detectedKeyword = midResult.keyword.trim();
        break; // Found — no need to process remaining audio
      }
    }

    // Flush final frames
    if (!detectedKeyword) {
      stream.inputFinished();
      while (_spotter.isReady(stream)) {
        _spotter.decode(stream);
      }
      const finalResult = _spotter.getResult(stream);
      if (finalResult && finalResult.keyword && finalResult.keyword.trim()) {
        detectedKeyword = finalResult.keyword.trim();
      }
    }

    const durationMs = Date.now() - t0;
    const detected = !!detectedKeyword;
    const keyword  = detectedKeyword;

    // Log RMS so we can verify audio has speech content
    const rms = Math.sqrt(samples.reduce((s, x) => s + x * x, 0) / samples.length);
    logger.info('[KWS] Detection result', {
      detected,
      keyword,
      durationMs,
      samplesLen: samples.length,
      audioDurationSec: (samples.length / 16000).toFixed(2),
      rms: rms.toFixed(4),
    });

    // KWS_DEBUG=1 → save decoded WAV for offline analysis with kws-api-diag.js
    if (!detected && process.env.KWS_DEBUG) {
      try {
        const dbgPath = path.join(os.tmpdir(), `kws-fail-${Date.now()}.wav`);
        const pcm = Buffer.allocUnsafe(samples.length * 2);
        for (let i = 0; i < samples.length; i++) {
          const s16 = Math.max(-32768, Math.min(32767, Math.round(samples[i] * 32767)));
          pcm.writeInt16LE(s16, i * 2);
        }
        // Write minimal WAV header + PCM data
        const hdr = Buffer.allocUnsafe(44);
        hdr.write('RIFF', 0); hdr.writeUInt32LE(36 + pcm.length, 4);
        hdr.write('WAVE', 8); hdr.write('fmt ', 12);
        hdr.writeUInt32LE(16, 16); hdr.writeUInt16LE(1, 20); hdr.writeUInt16LE(1, 22);
        hdr.writeUInt32LE(16000, 24); hdr.writeUInt32LE(32000, 28);
        hdr.writeUInt16LE(2, 32); hdr.writeUInt16LE(16, 34);
        hdr.write('data', 36); hdr.writeUInt32LE(pcm.length, 40);
        fs.writeFileSync(dbgPath, Buffer.concat([hdr, pcm]));
        logger.info('[KWS] DEBUG audio saved', { path: dbgPath });
      } catch (_) {}
    }

    return { detected, keyword, available: true, durationMs };
  } catch (err) {
    logger.error('[KWS] Detection error', { error: err.message });
    return { detected: false, keyword: null, available: true, error: err.message };
  }
}

/**
 * Reset internal state (useful after model files are (re)downloaded without restart).
 */
function reset() {
  _spotter = null;
  _available = null;
  _initError = null;
}

module.exports = { isAvailable, detectKeyword, reset };
