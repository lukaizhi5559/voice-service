'use strict';

/**
 * wake-word.cjs — revived from deprecated/ for sleep-mode wake-phrase gating.
 * detect(transcript) → { detected, matchedPhrase, score, type:'wake'|'cancel'|'status' }
 * The heavy lifting (Fuse fuzzy match + Whisper-hallucination regexes) lives in
 * the archived implementation — unchanged and battle-tested.
 */
module.exports = require('./deprecated/wake-word.cjs');
