#!/usr/bin/env node
'use strict';
const fs = require('fs');
const kws = require('../src/kws-detector.cjs');

async function main() {
  const wavBuf = fs.readFileSync('/tmp/0.wav');
  const base64 = wavBuf.toString('base64');
  console.log('WAV size:', wavBuf.length, 'bytes');
  console.log('Testing KWS against test WAV (should NOT detect armis/thinkdrop)...');

  const r = await kws.detectKeyword(base64, 'wav');
  console.log('Result:', JSON.stringify(r));
  if (!r.detected) {
    console.log('PASS: no false positive detected');
  } else {
    console.log('FAIL: unexpected detection:', r.keyword);
  }
}

main().catch(e => { console.error('Error:', e.message); process.exit(1); });
