'use strict';
/**
 * kws-debug-test.js — debug=1 to see model internal output
 */
const s = require('../node_modules/sherpa-onnx-node');
const path = require('path');
const fs = require('fs');
const os = require('os');
const { execSync } = require('child_process');

const MODEL_DIR = path.join(__dirname, '..', 'models', 'kws');

const kws = new s.KeywordSpotter({
  featConfig: { sampleRate: 16000, featureDim: 80 },
  modelConfig: {
    transducer: {
      encoder: path.join(MODEL_DIR, 'encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx'),
      decoder: path.join(MODEL_DIR, 'decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx'),
      joiner:  path.join(MODEL_DIR, 'joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx'),
    },
    tokens: path.join(MODEL_DIR, 'tokens.txt'),
    numThreads: 2, debug: 1, provider: 'cpu',
    modelingUnit: 'bpe',
  },
  maxActivePaths: 4, numTrailingBlanks: 1,
  keywordsScore: 1.5,
  keywordsThreshold: 0.001,
  keywordsFile: path.join(MODEL_DIR, 'keywords.txt'),
});

console.log('Model loaded. Testing with "armis" TTS...\n');

const aiff = '/tmp/kws-debug-armis.aiff';
const wav  = '/tmp/kws-debug-armis.wav';
execSync(`say -o "${aiff}" "armis" 2>/dev/null`);
execSync(`ffmpeg -y -i "${aiff}" -ar 16000 -ac 1 -sample_fmt s16 "${wav}" 2>/dev/null`);

const wavBuf = fs.readFileSync(wav);
let offset = 44;
for (let i = 12; i < Math.min(wavBuf.length-8, 256); i++) {
  if (wavBuf[i]===0x64&&wavBuf[i+1]===0x61&&wavBuf[i+2]===0x74&&wavBuf[i+3]===0x61) { offset=i+8; break; }
}
const n = (wavBuf.length - offset) / 2;
const samples = new Float32Array(n);
for (let i = 0; i < n; i++) samples[i] = wavBuf.readInt16LE(offset + i*2) / 32768.0;
console.log(`Samples: ${samples.length} (${(samples.length/16000).toFixed(2)}s)\n`);

// Batch approach
const stream = kws.createStream();
stream.acceptWaveform({ sampleRate: 16000, samples });
stream.inputFinished();
let iters = 0;
while (kws.isReady(stream)) { kws.decode(stream); iters++; }
const result = kws.getResult(stream);
console.log(`\nDone. iters=${iters} result=`, JSON.stringify(result));
