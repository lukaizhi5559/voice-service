'use strict';
/**
 * kws-threshold-test.js — Tests KWS at various thresholds with TTS audio
 * to determine what threshold is needed for detection.
 */
const s = require('../node_modules/sherpa-onnx-node');
const path = require('path');
const fs = require('fs');
const os = require('os');
const { execSync } = require('child_process');

const MODEL_DIR = path.join(__dirname, '..', 'models', 'kws');

function buildKws(threshold, score) {
  return new s.KeywordSpotter({
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
    keywordsScore: score,
    keywordsThreshold: threshold,
    keywordsFile: path.join(MODEL_DIR, 'keywords.txt'),
  });
}

function wavToSamples(wavPath) {
  const wavBuf = fs.readFileSync(wavPath);
  let offset = 44;
  for (let i = 12; i < Math.min(wavBuf.length - 8, 256); i++) {
    if (wavBuf[i] === 0x64 && wavBuf[i+1] === 0x61 && wavBuf[i+2] === 0x74 && wavBuf[i+3] === 0x61) {
      offset = i + 8; break;
    }
  }
  const n = (wavBuf.length - offset) / 2;
  const samples = new Float32Array(n);
  for (let i = 0; i < n; i++) samples[i] = wavBuf.readInt16LE(offset + i * 2) / 32768.0;
  return samples;
}

// Batch detect: feed all at once
function detectBatch(kws, samples) {
  const stream = kws.createStream();
  stream.acceptWaveform({ sampleRate: 16000, samples });
  stream.inputFinished();
  while (kws.isReady(stream)) kws.decode(stream);
  const r = kws.getResult(stream);
  return r?.keyword?.trim() || '';
}

// Chunked detect: 2560 samples = 160ms (model's natural chunk - 16 frames × 160 samples/frame)
function detectChunked(kws, samples, chunkSize = 2560) {
  const stream = kws.createStream();
  for (let i = 0; i < samples.length; i += chunkSize) {
    const chunk = samples.slice(i, Math.min(i + chunkSize, samples.length));
    stream.acceptWaveform({ sampleRate: 16000, samples: chunk });
    while (kws.isReady(stream)) kws.decode(stream);
    const r = kws.getResult(stream);
    if (r?.keyword?.trim()) return r.keyword.trim();
  }
  stream.inputFinished();
  while (kws.isReady(stream)) kws.decode(stream);
  const r = kws.getResult(stream);
  return r?.keyword?.trim() || '';
}

// Build test WAVs
const testCases = [
  { phrase: 'armis', label: 'armis' },
  { phrase: 'hey armis', label: 'hey armis' },
  { phrase: 'think drop', label: 'think drop' },
  { phrase: 'hey thinkdrop', label: 'hey thinkdrop' },
  { phrase: 'hello world', label: 'hello world (negative)' },
];

console.log('\n=== KWS Threshold Scan ===\n');

for (const tc of testCases) {
  const aiff = path.join(os.tmpdir(), `kws-tts-${tc.phrase.replace(/ /g,'_')}.aiff`);
  const wav  = path.join(os.tmpdir(), `kws-tts-${tc.phrase.replace(/ /g,'_')}.wav`);
  try {
    execSync(`say -o "${aiff}" "${tc.phrase}" 2>/dev/null`);
    execSync(`ffmpeg -y -i "${aiff}" -ar 16000 -ac 1 -sample_fmt s16 "${wav}" 2>/dev/null`);
  } catch(e) { console.error(`  TTS failed for "${tc.phrase}":`, e.message); continue; }

  const samples = wavToSamples(wav);
  const rms = Math.sqrt(Array.from(samples).reduce((s,x) => s + x*x, 0) / samples.length);
  process.stdout.write(`\n[${tc.label}] RMS=${rms.toFixed(3)} ${samples.length} samples\n`);

  // Test thresholds from aggressive to conservative
  const configs = [
    { threshold: 0.001, score: 3.0 },
    { threshold: 0.01,  score: 2.0 },
    { threshold: 0.05,  score: 1.5 },
    { threshold: 0.25,  score: 1.0 },
  ];

  for (const cfg of configs) {
    const kws = buildKws(cfg.threshold, cfg.score);
    const batch   = detectBatch(kws, samples);
    const chunked = detectChunked(kws, samples);
    const hit = batch || chunked;
    console.log(`  thresh=${cfg.threshold} score=${cfg.score}: batch="${batch}" chunked="${chunked}" ${hit ? '✓ DETECTED' : '✗'}`);
  }
}
