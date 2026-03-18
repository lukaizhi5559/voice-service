/**
 * kws-api-diag.js — Manual KWS detection test
 *
 * Usage:
 *   node tests/kws-api-diag.js                  # synthesise phrases with macOS `say`
 *   node tests/kws-api-diag.js /path/to/file.wav # test a specific WAV file
 *
 * Run from: /Users/lukaizhi/Desktop/projects/thinkdrop/mcp-services/voice-service
 */
'use strict';

const path    = require('path');
const fs      = require('fs');
const os      = require('os');
const { execSync, spawnSync } = require('child_process');

// ── Paths (absolute so CWD doesn't matter) ───────────────────────────────────
const ROOT      = path.resolve(__dirname, '..');
const MODEL_DIR = path.join(ROOT, 'models', 'kws');
const sherpa    = require(path.join(ROOT, 'node_modules', 'sherpa-onnx-node'));

// ── Build KWS directly (bypass kws-detector.cjs for raw diagnostics) ─────────
// Read model paths from model-config.json so this stays in sync with the live config
const modelCfg = JSON.parse(fs.readFileSync(path.join(MODEL_DIR, 'model-config.json'), 'utf8'));
const mf = modelCfg.files;
console.log(`[diag] Using model: ${modelCfg.model}`);

const kws = new sherpa.KeywordSpotter({
  featConfig: { sampleRate: 16000, featureDim: 80 },
  modelConfig: {
    transducer: {
      encoder: path.join(MODEL_DIR, mf.encoder),
      decoder: path.join(MODEL_DIR, mf.decoder),
      joiner:  path.join(MODEL_DIR, mf.joiner),
    },
    tokens:     path.join(MODEL_DIR, mf.tokens),
    numThreads: 2,
    debug:      0,
    provider:   'cpu',
  },
  maxActivePaths:    8,
  numTrailingBlanks: 2,
  keywordsScore:     2.0,
  keywordsThreshold: 0.05,
  keywordsFile:      path.join(MODEL_DIR, mf.keywords),
});

console.log('\n[diag] KeywordSpotter created');
console.log('[diag] keywords.txt:');
fs.readFileSync(path.join(MODEL_DIR, 'keywords.txt'), 'utf8')
  .split('\n').filter(Boolean)
  .forEach(l => console.log('        ', l));

// ── Helpers ───────────────────────────────────────────────────────────────────

function ffmpegPath() {
  for (const p of ['/opt/homebrew/bin/ffmpeg', '/usr/local/bin/ffmpeg', '/usr/bin/ffmpeg']) {
    if (fs.existsSync(p)) return p;
  }
  return 'ffmpeg';
}

function wavToFloat32(wavPath) {
  const tmpOut = path.join(os.tmpdir(), `kws-diag-${Date.now()}.wav`);
  const r = spawnSync(ffmpegPath(), [
    '-y', '-i', wavPath,
    '-ar', '16000', '-ac', '1', '-sample_fmt', 's16', '-f', 'wav', tmpOut,
  ], { stdio: ['ignore', 'ignore', 'pipe'] });
  if (r.status !== 0) throw new Error('ffmpeg failed: ' + r.stderr?.toString().slice(-300));

  const buf = fs.readFileSync(tmpOut);
  fs.unlinkSync(tmpOut);

  // Find 'data' chunk
  let offset = 44;
  for (let i = 12; i < Math.min(buf.length - 8, 256); i++) {
    if (buf[i] === 0x64 && buf[i+1] === 0x61 && buf[i+2] === 0x74 && buf[i+3] === 0x61) {
      offset = i + 8; break;
    }
  }
  const n = (buf.length - offset) / 2;
  const samples = new Float32Array(n);
  for (let i = 0; i < n; i++) samples[i] = buf.readInt16LE(offset + i * 2) / 32768.0;
  return samples;
}

function synthToWav(text) {
  const tmp = path.join(os.tmpdir(), `kws-say-${Date.now()}.aiff`);
  const wav = tmp.replace('.aiff', '.wav');
  // macOS say → AIFF → WAV via ffmpeg
  // NOTE: no manual padding here — kws-detector.cjs adds 800ms leading silence internally
  const r1 = spawnSync('say', ['-o', tmp, text]);
  if (r1.status !== 0) throw new Error('say failed');
  // Add 1.5s trailing silence to flush the decoder (simulates VAD tail silence)
  const r2 = spawnSync(ffmpegPath(), [
    '-y',
    '-i', tmp,                                                // speech
    '-f', 'lavfi', '-i', 'aevalsrc=0:s=16000:c=1:d=1.5',   // 1.5s trailing silence
    '-filter_complex', '[0:a][1:a]concat=n=2:v=0:a=1[out]',
    '-map', '[out]',
    '-ar', '16000', '-ac', '1', '-sample_fmt', 's16', '-f', 'wav', wav,
  ], { stdio: ['ignore', 'ignore', 'pipe'] });
  fs.unlinkSync(tmp);
  if (r2.status !== 0) throw new Error('ffmpeg (say→wav) failed: ' + r2.stderr?.toString().slice(-200));
  return wav;
}

// ── Detection ─────────────────────────────────────────────────────────────────

function runKWS(samples, label) {
  // Mirror the 800ms prepend in kws-detector.cjs
  const LEAD = 16000 * 0.8;
  const padded = new Float32Array(LEAD + samples.length);
  padded.set(samples, LEAD);
  samples = padded;

  const CHUNK = 2560; // 160ms
  const stream = kws.createStream();
  let found = null;
  let iters = 0;

  for (let i = 0; i < samples.length && !found; i += CHUNK) {
    const chunk = samples.slice(i, Math.min(i + CHUNK, samples.length));
    stream.acceptWaveform({ sampleRate: 16000, samples: chunk });
    while (kws.isReady(stream)) { kws.decode(stream); iters++; }
    const r = kws.getResult(stream);
    if (r && r.keyword && r.keyword.trim()) found = r.keyword.trim();
  }

  if (!found) {
    stream.inputFinished();
    while (kws.isReady(stream)) { kws.decode(stream); iters++; }
    const r = kws.getResult(stream);
    if (r && r.keyword && r.keyword.trim()) found = r.keyword.trim();
  }

  const dur = (samples.length / 16000).toFixed(2);
  const status = found ? `✅  DETECTED: "${found}"` : '❌  not detected';
  console.log(`\n[${label}] ${dur}s, ${samples.length} samples, ${iters} decode iters`);
  console.log(`  → ${status}`);
  return found;
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main() {
  const userFile = process.argv[2];

  if (userFile) {
    // Test a specific file
    console.log(`\n[diag] Testing file: ${userFile}`);
    const samples = wavToFloat32(userFile);
    runKWS(samples, path.basename(userFile));
    return;
  }

  // Synthesize test phrases with macOS `say`
  const phrases = [
    'Armis',
    'Hey Armis',
    'OK Armis',
    'ThinkDrop',
    'Hey ThinkDrop',
    'Armies',               // alternative — keyword ▁A R M IES @armies
    'hello world',          // negative — should NOT detect
    'the weather today',    // negative — should NOT detect
  ];

  console.log('\n[diag] Synthesizing phrases with macOS `say` and testing KWS...\n');

  for (const phrase of phrases) {
    let wavPath;
    try {
      wavPath = synthToWav(phrase);
      const samples = wavToFloat32(wavPath);
      runKWS(samples, `say: "${phrase}"`);
    } catch (err) {
      console.log(`\n[say: "${phrase}"] ERROR: ${err.message}`);
    } finally {
      if (wavPath && fs.existsSync(wavPath)) fs.unlinkSync(wavPath);
    }
  }

  console.log('\n[diag] Done.\n');
}

main().catch(err => { console.error('[diag] Fatal:', err.message); process.exit(1); });
