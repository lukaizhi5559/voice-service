'use strict';
/**
 * kws-audio-diag.js — Diagnoses audio quality coming into the KWS pipeline.
 * 
 * Usage: node tests/kws-audio-diag.js <path-to-webm-or-wav>
 * 
 * If no file given, generates a synthetic 440Hz sine wave (should produce some features)
 * and tests the KWS with various thresholds.
 */
const s = require('../node_modules/sherpa-onnx-node');
const path = require('path');
const fs = require('fs');
const os = require('os');
const { execSync } = require('child_process');

const MODEL_DIR = path.join(__dirname, '..', 'models', 'kws');

// ── Build KWS ─────────────────────────────────────────────────────────────────
const kws = new s.KeywordSpotter({
  featConfig: { sampleRate: 16000, featureDim: 80 },
  modelConfig: {
    transducer: {
      encoder: path.join(MODEL_DIR, 'encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx'),
      decoder: path.join(MODEL_DIR, 'decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx'),
      joiner:  path.join(MODEL_DIR, 'joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx'),
    },
    tokens: path.join(MODEL_DIR, 'tokens.txt'),
    numThreads: 2, debug: 0, provider: 'cpu',
  },
  maxActivePaths: 4, numTrailingBlanks: 1,
  keywordsScore: 2.0,
  keywordsThreshold: 0.05,
  keywordsFile: path.join(MODEL_DIR, 'keywords.txt'),
});

// ── Helper: detect with chunking (simulates streaming) ───────────────────────
function detectChunked(samples, chunkSize = 1600) {
  const stream = kws.createStream();
  for (let i = 0; i < samples.length; i += chunkSize) {
    const chunk = samples.slice(i, Math.min(i + chunkSize, samples.length));
    stream.acceptWaveform({ sampleRate: 16000, samples: chunk });
    while (kws.isReady(stream)) kws.decode(stream);
    const r = kws.getResult(stream);
    if (r && r.keyword && r.keyword.trim()) {
      return { detected: true, keyword: r.keyword.trim(), atSample: i };
    }
  }
  stream.inputFinished();
  while (kws.isReady(stream)) kws.decode(stream);
  const final = kws.getResult(stream);
  return { detected: !!(final && final.keyword && final.keyword.trim()), keyword: final?.keyword?.trim() || null };
}

// ── Helper: RMS of float32 samples ───────────────────────────────────────────
function rms(samples) {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  return Math.sqrt(sum / samples.length);
}

// ── Helper: decode audio file to PCM ─────────────────────────────────────────
function decodeAudio(inputPath) {
  const outWav = path.join(os.tmpdir(), `kws-diag-${Date.now()}.wav`);
  execSync(`ffmpeg -y -i "${inputPath}" -ar 16000 -ac 1 -sample_fmt s16 -f wav "${outWav}" 2>/dev/null`, { stdio: 'pipe' });
  const wavBuf = fs.readFileSync(outWav);
  fs.unlinkSync(outWav);
  
  // Find data chunk
  let offset = 44;
  for (let i = 12; i < Math.min(wavBuf.length - 8, 256); i++) {
    if (wavBuf[i] === 0x64 && wavBuf[i+1] === 0x61 && wavBuf[i+2] === 0x74 && wavBuf[i+3] === 0x61) {
      offset = i + 8; break;
    }
  }
  const numSamples = (wavBuf.length - offset) / 2;
  const samples = new Float32Array(numSamples);
  for (let i = 0; i < numSamples; i++) {
    samples[i] = wavBuf.readInt16LE(offset + i * 2) / 32768.0;
  }
  return samples;
}

// ── Main ──────────────────────────────────────────────────────────────────────
const inputFile = process.argv[2];

if (inputFile) {
  console.log(`\nTesting with audio file: ${inputFile}`);
  const samples = decodeAudio(inputFile);
  console.log(`  Decoded: ${samples.length} samples = ${(samples.length/16000).toFixed(2)}s`);
  console.log(`  RMS: ${rms(samples).toFixed(6)} (>0.001 = audible signal)`);
  const maxAbs = Math.max(...Array.from(samples).map(Math.abs));
  console.log(`  Peak: ${maxAbs.toFixed(4)}`);
  
  const result = detectChunked(samples);
  console.log(`  KWS result: detected=${result.detected}, keyword=${result.keyword}`);
} else {
  // Test 1: silence
  console.log('\n[Test 1] Silence (0.5s):');
  const silence = new Float32Array(8000).fill(0);
  const r1 = detectChunked(silence);
  console.log(`  RMS=0.000000 | detected=${r1.detected}`);

  // Test 2: synthetic tone (440Hz)
  console.log('\n[Test 2] Synthetic 440Hz sine (2s):');
  const tone = new Float32Array(32000);
  for (let i = 0; i < 32000; i++) tone[i] = 0.3 * Math.sin(2 * Math.PI * 440 * i / 16000);
  console.log(`  RMS=${rms(tone).toFixed(6)}`);
  const r2 = detectChunked(tone);
  console.log(`  detected=${r2.detected}, keyword=${r2.keyword}`);

  console.log('\n[Info] Pass a WebM/WAV file as argument to test real audio:');
  console.log('  node tests/kws-audio-diag.js /path/to/audio.webm');
}
