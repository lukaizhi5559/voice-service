'use strict';
/**
 * kws-snakecase-test.js — Tests KWS with snake_case property names
 * (as used in the official sherpa-onnx nodejs-addon-examples)
 */
const s = require('../node_modules/sherpa-onnx-node');
const path = require('path');
const fs = require('fs');
const os = require('os');
const { execSync } = require('child_process');

const MODEL_DIR = path.join(__dirname, '..', 'models', 'kws');

const kws = new s.KeywordSpotter({
  feat_config: { sample_rate: 16000, feature_dim: 80 },
  model_config: {
    transducer: {
      encoder: path.join(MODEL_DIR, 'encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx'),
      decoder: path.join(MODEL_DIR, 'decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx'),
      joiner:  path.join(MODEL_DIR, 'joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx'),
    },
    tokens: path.join(MODEL_DIR, 'tokens.txt'),
    num_threads: 2, debug: 0, provider: 'cpu',
  },
  max_active_paths: 4,
  num_trailing_blanks: 1,
  keywords_score: 1.5,
  keywords_threshold: 0.25,
  keywords_file: path.join(MODEL_DIR, 'keywords.txt'),
});

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

function detectBatch(samples) {
  const stream = kws.createStream();
  stream.acceptWaveform({ sampleRate: 16000, samples });
  stream.inputFinished();
  while (kws.isReady(stream)) kws.decode(stream);
  const r = kws.getResult(stream);
  return r?.keyword?.trim() || '';
}

function detectChunked(samples, chunkSize = 2560) {
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

const testCases = [
  'armis', 'hey armis', 'think drop', 'hey thinkdrop', 'hello',
];

console.log('\n=== KWS Snake_case Config Test ===\n');

for (const phrase of testCases) {
  const aiff = `/tmp/kws-sc-${phrase.replace(/ /g,'_')}.aiff`;
  const wav  = `/tmp/kws-sc-${phrase.replace(/ /g,'_')}.wav`;
  execSync(`say -o "${aiff}" "${phrase}" 2>/dev/null`);
  execSync(`ffmpeg -y -i "${aiff}" -ar 16000 -ac 1 -sample_fmt s16 "${wav}" 2>/dev/null`);
  const samples = wavToSamples(wav);
  const rms = Math.sqrt(Array.from(samples).reduce((s,x) => s + x*x, 0) / samples.length);
  const batch   = detectBatch(samples);
  const chunked = detectChunked(samples);
  const hit = batch || chunked;
  console.log(`[${phrase}] RMS=${rms.toFixed(3)} → batch="${batch}" chunked="${chunked}" ${hit ? '✓ DETECTED' : '✗'}`);
}
