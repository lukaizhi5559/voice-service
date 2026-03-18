/**
 * Manual test for deepgram-stt.cjs
 * Uses macOS `say` to synthesize speech, then sends it to Deepgram Nova-2 transcription.
 *
 * Usage:
 *   node tests/test-deepgram-stt.cjs
 *
 * Requirements:
 *   - DEEPGRAM_API_KEY set in .env (voice-service)
 */

'use strict';

require('dotenv').config({ path: require('path').join(__dirname, '../.env') });

const { execSync } = require('child_process');
const fs   = require('fs');
const path = require('path');
const os   = require('os');

const { transcribe, isAvailable } = require('../src/deepgram-stt.cjs');

if (!isAvailable()) {
  console.error('❌  DEEPGRAM_API_KEY not set. Add it to mcp-services/voice-service/.env');
  process.exit(1);
}

const PHRASES = [
  { text: 'Armis, what is the time?',               lang: null },
  { text: 'Hey ThinkDrop, remind me tomorrow',       lang: null },
  { text: 'OK Armis, how is the weather today?',     lang: null },
  { text: 'The quick brown fox jumps over the lazy dog', lang: null },
];

async function runTest() {
  console.log(`\n=== Deepgram Nova-2 STT Test ===\n`);

  let passed = 0;

  for (const { text, lang } of PHRASES) {
    const tmpWav = path.join(os.tmpdir(), `dg-test-${Date.now()}.wav`);
    try {
      // Synthesize speech to WAV using macOS `say`
      execSync(`say -r 160 -o "${tmpWav}" --data-format=LEF32@16000 "${text}"`, { stdio: 'pipe' });

      // Read raw PCM from macOS AIFF-C (say produces AIFF) — convert to 16-bit PCM WAV via afconvert
      const tmpPcm = path.join(os.tmpdir(), `dg-pcm-${Date.now()}.wav`);
      execSync(`afconvert -f WAVE -d LEI16@16000 -c 1 "${tmpWav}" "${tmpPcm}"`, { stdio: 'pipe' });

      const audioBuf = fs.readFileSync(tmpPcm);
      const result = await transcribe({ audioBuffer: audioBuf, format: 'wav', languageHint: lang || null });

      const detected = result.text.toLowerCase();
      const expected = text.toLowerCase();
      const ok = detected.length > 0;

      console.log(`${ok ? '✅' : '❌'}  Input:    "${text}"`);
      console.log(`    Output:   "${result.text}"`);
      console.log(`    Language: ${result.language}  Confidence: ${(result.confidence * 100).toFixed(1)}%\n`);

      if (ok) passed++;

      fs.unlinkSync(tmpWav);
      fs.unlinkSync(tmpPcm);
    } catch (err) {
      console.error(`❌  ERROR for "${text}": ${err.message}\n`);
      try { fs.unlinkSync(tmpWav); } catch {}
    }
  }

  console.log(`=== Results: ${passed}/${PHRASES.length} transcribed ===\n`);
}

runTest().catch(err => {
  console.error('Fatal:', err);
  process.exit(1);
});
