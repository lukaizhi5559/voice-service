'use strict';
/**
 * kws-padded-test.js — Test with 1s silence padding before/after to flush the
 * transducer decoder's context window.
 */
const s = require('../node_modules/sherpa-onnx-node');
const path = require('path');
const fs = require('fs');
const os = require('os');
const { execSync } = require('child_process');

const MODEL_DIR = path.join(__dirname, '..', 'models', 'kws');

const recognizer = new s.OnlineRecognizer({
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
  decodingMethod: 'greedy_search', maxActivePaths: 4, enableEndpoint: 0,
});

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
  keywordsScore: 1.5, keywordsThreshold: 0.001,
  keywordsFile: path.join(MODEL_DIR, 'keywords.txt'),
});

function wavToSamples(wavPath) {
  const wavBuf = fs.readFileSync(wavPath);
  let offset = 44;
  for (let i = 12; i < Math.min(wavBuf.length-8, 256); i++) {
    if (wavBuf[i]===0x64&&wavBuf[i+1]===0x61&&wavBuf[i+2]===0x74&&wavBuf[i+3]===0x61) { offset=i+8; break; }
  }
  const n = (wavBuf.length - offset) / 2;
  const samples = new Float32Array(n);
  for (let i = 0; i < n; i++) samples[i] = wavBuf.readInt16LE(offset + i*2) / 32768.0;
  return samples;
}

// Pad with silence: 0.5s before, 1s after to allow transducer to flush context
function padSilence(samples, preSeconds = 0.5, postSeconds = 1.0) {
  const pre  = new Float32Array(Math.round(preSeconds  * 16000));
  const post = new Float32Array(Math.round(postSeconds * 16000));
  const out  = new Float32Array(pre.length + samples.length + post.length);
  out.set(pre, 0);
  out.set(samples, pre.length);
  out.set(post, pre.length + samples.length);
  return out;
}

function transcribe(samples) {
  const stream = recognizer.createStream();
  const CHUNK = 2560;
  for (let i = 0; i < samples.length; i += CHUNK) {
    stream.acceptWaveform({ sampleRate: 16000, samples: samples.slice(i, i + CHUNK) });
    while (recognizer.isReady(stream)) recognizer.decode(stream);
  }
  stream.inputFinished();
  while (recognizer.isReady(stream)) recognizer.decode(stream);
  return recognizer.getResult(stream);
}

function detectKws(samples) {
  const CHUNK = 2560;
  const stream = kws.createStream();
  for (let i = 0; i < samples.length; i += CHUNK) {
    stream.acceptWaveform({ sampleRate: 16000, samples: samples.slice(i, i + CHUNK) });
    while (kws.isReady(stream)) kws.decode(stream);
    const r = kws.getResult(stream);
    if (r?.keyword?.trim()) return r.keyword.trim();
  }
  stream.inputFinished();
  while (kws.isReady(stream)) kws.decode(stream);
  const r = kws.getResult(stream);
  return r?.keyword?.trim() || '';
}

const phrases = ['armis', 'hey armis', 'think drop', 'thinkdrop', 'the', 'hello'];

console.log('\n=== Padded Audio Test (silence pre+post flush) ===\n');

for (const phrase of phrases) {
  const aiff = `/tmp/kwspad-${phrase.replace(/ /g,'_')}.aiff`;
  const wav  = `/tmp/kwspad-${phrase.replace(/ /g,'_')}.wav`;
  execSync(`say -o "${aiff}" "${phrase}" 2>/dev/null`);
  execSync(`ffmpeg -y -i "${aiff}" -ar 16000 -ac 1 -sample_fmt s16 "${wav}" 2>/dev/null`);
  const raw = wavToSamples(wav);
  const padded = padSilence(raw);

  const asr = transcribe(padded);
  const kwsHit = detectKws(padded);

  console.log(`[${phrase}]  ${raw.length} samples + padding = ${padded.length}`);
  console.log(`  ASR: "${asr?.text}"  tokens: [${asr?.tokens?.join(', ')}]`);
  console.log(`  KWS: "${kwsHit || '(none)'}"`);
  console.log();
}
