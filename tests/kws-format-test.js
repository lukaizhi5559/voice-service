'use strict';
/**
 * kws-format-test.js — Test different keyword file formats to find what works
 */
const s = require('../node_modules/sherpa-onnx-node');
const path = require('path');
const fs = require('fs');
const os = require('os');
const { execSync } = require('child_process');

const MODEL_DIR = path.join(__dirname, '..', 'models', 'kws');

// Helper: write a keywords buf and test detection
function makeKws(keywordsBuf, threshold = 0.001, score = 1.5) {
  return new s.KeywordSpotter({
    featConfig: { sampleRate: 16000, featureDim: 80 },
    modelConfig: {
      transducer: {
        encoder: path.join(MODEL_DIR, 'encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx'),
        decoder: path.join(MODEL_DIR, 'decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx'),
        joiner:  path.join(MODEL_DIR, 'joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx'),
      },
      tokens: path.join(MODEL_DIR, 'tokens.txt'),
      numThreads: 1, debug: 0, provider: 'cpu',
    },
    maxActivePaths: 4, numTrailingBlanks: 1,
    keywordsScore: score, keywordsThreshold: threshold,
    keywordsBuf: keywordsBuf,
    keywordsBufSize: Buffer.byteLength(keywordsBuf, 'utf8'),
  });
}

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

function detectBatch(kws, samples) {
  const stream = kws.createStream();
  stream.acceptWaveform({ sampleRate: 16000, samples });
  stream.inputFinished();
  while (kws.isReady(stream)) kws.decode(stream);
  const r = kws.getResult(stream);
  return r?.keyword?.trim() || '';
}

// Prepare audio files
function makeTts(phrase) {
  const key = phrase.replace(/ /g, '_');
  const aiff = `/tmp/kwsfmt-${key}.aiff`;
  const wav  = `/tmp/kwsfmt-${key}.wav`;
  try {
    execSync(`say -o "${aiff}" "${phrase}" 2>/dev/null`);
    execSync(`ffmpeg -y -i "${aiff}" -ar 16000 -ac 1 -sample_fmt s16 "${wav}" 2>/dev/null`);
    return wavToSamples(wav);
  } catch(e) { return null; }
}

console.log('\n=== KWS Keyword Format Test ===\n');

const phrases = {
  'the':       makeTts('the'),
  'armis':     makeTts('armis'),
  'think drop': makeTts('think drop'),
};

// Format A: BPE pieces with ▁ word-boundary (original approach)
const fmtA = [
  '▁THE @the',
  '▁A R M IS @armis',
  '▁THINK ▁DR O P @think_drop',
].join('\n') + '\n';

// Format B: individual chars with ▁ as separate token
const fmtB = [
  '▁ T H E @the',
  '▁ A R M I S @armis',
  '▁ T H I N K ▁ D R O P @think_drop',
].join('\n') + '\n';

// Format C: pure BPE pieces, no ▁ on first piece (continuation tokens)
const fmtC = [
  'T H E @the',
  'A R M I S @armis',
  'T H I N K D R O P @think_drop',
].join('\n') + '\n';

// Format D: using known vocabulary entries verbatim
// THE = ▁THE (token 5) — single token
const fmtD = [
  '▁THE @the',
  '▁A R M I S @armis',
  '▁THINK D R O P @think_drop',
].join('\n') + '\n';

const formats = [
  { name: 'A (BPE pieces + ▁ word marker)', buf: fmtA },
  { name: 'B (▁ separate + individual chars)', buf: fmtB },
  { name: 'C (no ▁, individual chars)', buf: fmtC },
  { name: 'D (▁THE single token + char rest)', buf: fmtD },
];

for (const fmt of formats) {
  console.log(`\n[Format ${fmt.name}]`);
  let kws;
  try { kws = makeKws(fmt.buf); } catch(e) { console.log(`  ERROR: ${e.message}`); continue; }

  for (const [phrase, samples] of Object.entries(phrases)) {
    if (!samples) { console.log(`  ${phrase}: no audio`); continue; }
    const hit = detectBatch(kws, samples);
    console.log(`  "${phrase}" → "${hit || '(none)'}" ${hit ? '✓' : '✗'}`);
  }
}
