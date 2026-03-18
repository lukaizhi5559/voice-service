#!/usr/bin/env node
/**
 * download-kws-model.js
 *
 * Downloads the sherpa-onnx keyword-spotting (KWS) model for ThinkDrop wake word detection.
 * Model: sherpa-onnx-kws-zipformer-gigaspeech-3.3M (Apache-2.0, ~3.3MB compressed)
 *
 * What this script does:
 *   1. Downloads the model tarball from GitHub releases (~3.3MB)
 *   2. Extracts the 3 ONNX model files + tokens.txt
 *   3. Writes models/kws/keywords.txt with pre-computed BPE tokenizations
 *      (tokenizations verified via greedy matching against the model's tokens.txt)
 *
 * BPE tokenizations (greedy longest-match against model's 500-token vocab):
 *   ARMIS        → ▁A R M IS
 *   HEY ARMIS    → ▁HE Y ▁A R M IS
 *   HI ARMIS     → ▁HI ▁A R M IS
 *   OK ARMIS     → ▁O K ▁A R M IS
 *   THINKDROP    → ▁THINK D RO P
 *   THINK DROP   → ▁THINK ▁DR O P
 *   HEY THINKDROP → ▁HE Y ▁THINK D RO P
 *   HEY THINK DROP → ▁HE Y ▁THINK ▁DR O P
 *   OK THINKDROP → ▁O K ▁THINK D RO P
 *
 * Run once before starting the voice-service:
 *   node scripts/download-kws-model.js
 *   (or: npm run setup:kws)
 */

'use strict';

const https = require('https');
const http = require('http');
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const MODEL_DIR = path.join(__dirname, '..', 'models', 'kws');

// GitHub releases URL — publicly accessible, no auth needed (unlike HuggingFace)
const TARBALL_URL = 'https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2';
const TARBALL_PREFIX = 'sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01';

// Files to extract from the tarball (int8 quantized = smaller, same accuracy)
const EXTRACT_FILES = [
  'encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx',
  'decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx',
  'joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx',
  'tokens.txt',
];

// ── Pre-computed keywords.txt ─────────────────────────────────────────────────
// BPE tokenizations verified via greedy longest-match against the model's 500-token vocab.
// Format: "<BPE tokens> @<label>"
// The @label is returned by result.keyword for human-readable log messages.
//
// Threshold note: lower = more sensitive (more triggers), higher = more conservative.
// 0.25 is the model's suggested default. Currently we use the global keywordsThreshold
// in kws-detector.cjs config, so these per-keyword thresholds are informational.
const KEYWORDS_TXT = `▁A R M IS @armis
▁HE Y ▁A R M IS @hey_armis
▁HI ▁A R M IS @hi_armis
▁O K ▁A R M IS @ok_armis
▁THINK ▁DR O P @think_drop
▁THINK D RO P @thinkdrop
▁HE Y ▁THINK ▁DR O P @hey_think_drop
▁HE Y ▁THINK D RO P @hey_thinkdrop
▁O K ▁THINK D RO P @ok_thinkdrop
▁A R M IES @armies
▁A R M US @armus
`;

// ── Helpers ───────────────────────────────────────────────────────────────────

function ensureDir(dir) {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
    console.log(`  Created directory: ${dir}`);
  }
}

/**
 * Download a URL to a local file path, following redirects up to 5 hops.
 * @param {string} url
 * @param {string} destPath
 * @param {number} [hops]
 * @returns {Promise<void>}
 */
function download(url, destPath, hops = 0) {
  if (hops > 5) return Promise.reject(new Error('Too many redirects'));
  return new Promise((resolve, reject) => {
    const proto = url.startsWith('https') ? https : http;

    const request = proto.get(url, { headers: { 'User-Agent': 'ThinkDrop/1.0' } }, (response) => {
      const sc = response.statusCode;
      if (sc >= 301 && sc <= 308 && response.headers.location) {
        return download(response.headers.location, destPath, hops + 1).then(resolve).catch(reject);
      }

      if (sc !== 200) {
        return reject(new Error(`HTTP ${sc} for ${url}`));
      }

      const totalBytes = parseInt(response.headers['content-length'] || '0', 10);
      let downloaded = 0;
      const file = fs.createWriteStream(destPath);

      response.on('data', (chunk) => {
        downloaded += chunk.length;
        if (totalBytes > 0) {
          const pct = Math.round((downloaded / totalBytes) * 100);
          process.stdout.write(`\r    ${pct}% (${(downloaded / 1024 / 1024).toFixed(1)} MB / ${(totalBytes / 1024 / 1024).toFixed(1)} MB)  `);
        } else {
          process.stdout.write(`\r    ${(downloaded / 1024 / 1024).toFixed(1)} MB downloaded...  `);
        }
      });

      response.pipe(file);
      file.on('finish', () => { file.close(); process.stdout.write('\n'); resolve(); });
      file.on('error', (err) => { fs.unlink(destPath, () => {}); reject(err); });
    });

    request.on('error', reject);
    request.setTimeout(60000, () => { request.destroy(); reject(new Error(`Timeout downloading ${url}`)); });
  });
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main() {
  console.log('');
  console.log('ThinkDrop KWS Model Setup');
  console.log('─'.repeat(50));
  console.log(`  Model: sherpa-onnx-kws-zipformer-gigaspeech-3.3M`);
  console.log(`  Destination: ${MODEL_DIR}`);
  console.log('');

  ensureDir(MODEL_DIR);

  // Check if model files already exist
  const allExist = EXTRACT_FILES.every((f) => fs.existsSync(path.join(MODEL_DIR, f)));
  if (allExist) {
    console.log('  ✓ Model files already present — skipping download');
  } else {
    // ── Download tarball ─────────────────────────────────────────────────────
    const tmpTarball = path.join(MODEL_DIR, '_download.tar.bz2');
    console.log(`  Downloading model tarball...`);
    console.log(`  Source: ${TARBALL_URL}`);
    try {
      await download(TARBALL_URL, tmpTarball);
      const sizeMB = (fs.statSync(tmpTarball).size / 1024 / 1024).toFixed(1);
      console.log(`  ✓ Downloaded ${sizeMB} MB`);
    } catch (err) {
      console.error(`  ✗ Download failed: ${err.message}`);
      process.exit(1);
    }

    // ── Extract needed files ─────────────────────────────────────────────────
    console.log(`  Extracting model files...`);
    try {
      const tarPaths = EXTRACT_FILES.map((f) => `${TARBALL_PREFIX}/${f}`).join(' ');
      execSync(
        `tar -xjf "${tmpTarball}" --strip-components=1 -C "${MODEL_DIR}" ${tarPaths}`,
        { stdio: 'pipe' }
      );
      fs.unlinkSync(tmpTarball);
      console.log(`  ✓ Extracted ${EXTRACT_FILES.length} files`);
    } catch (err) {
      console.error(`  ✗ Extraction failed: ${err.message}`);
      if (fs.existsSync(tmpTarball)) fs.unlinkSync(tmpTarball);
      process.exit(1);
    }

    // Verify extraction
    for (const f of EXTRACT_FILES) {
      const fp = path.join(MODEL_DIR, f);
      if (!fs.existsSync(fp)) {
        console.error(`  ✗ Missing after extraction: ${f}`);
        process.exit(1);
      }
      const size = fs.statSync(fp).size;
      console.log(`  ✓ ${f} (${(size / 1024).toFixed(0)} KB)`);
    }
  }

  console.log('');

  // ── Write keywords.txt ─────────────────────────────────────────────────────
  const keywordsPath = path.join(MODEL_DIR, 'keywords.txt');
  fs.writeFileSync(keywordsPath, KEYWORDS_TXT, 'utf8');
  console.log(`  ✓ Written keywords.txt (${KEYWORDS_TXT.split('\n').filter(Boolean).length} keywords)`);
  console.log('');
  console.log('  Keywords:');
  for (const line of KEYWORDS_TXT.split('\n').filter(Boolean)) {
    // Extract @label for display
    const label = (line.match(/@(\S+)/) || ['', line])[1];
    console.log(`    ${label}`);
  }
  console.log('');

  // ── Write config.json ──────────────────────────────────────────────────────
  const configPath = path.join(MODEL_DIR, 'model-config.json');
  fs.writeFileSync(configPath, JSON.stringify({
    model: 'sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01',
    downloadedAt: new Date().toISOString(),
    files: {
      encoder:  'encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx',
      decoder:  'decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx',
      joiner:   'joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx',
      tokens:   'tokens.txt',
      keywords: 'keywords.txt',
    },
  }, null, 2), 'utf8');

  console.log('  ✓ Setup complete!');
  console.log('');
  console.log('  The voice-service KWS is now ready. Restart the voice-service to activate.');
  console.log('  The KWS gate will run before Groq STT in wake-word mode.');
  console.log('');
}

main().catch((err) => {
  console.error('Error:', err.message);
  process.exit(1);
});
