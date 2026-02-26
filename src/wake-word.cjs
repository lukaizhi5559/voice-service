/**
 * Wake Word Detector
 *
 * Flexible phrase matching — no single wake phrase required.
 * Uses fuzzy matching so natural speech variations work:
 *   "ThinkDrop" / "Think Drop" / "hey ThinkDrop" / "yo ThinkDrop"
 *   "you awake" / "are you there" / "wake up" / "listen up"
 *   "I need you" / "hey assistant" / "ThinkiDrop" (typo forgiveness)
 *
 * Strategy:
 *   1. Configurable phrase list from env (WAKE_WORD_PHRASES)
 *   2. Fuse.js fuzzy match on the STT transcript
 *   3. Sensitivity threshold (0.0 = exact, 1.0 = anything matches)
 *   4. Also detects CANCEL/STOP phrases to abort active StateGraph runs
 *
 * This module does NOT do audio detection — it operates on already-transcribed text.
 * Audio activation is handled by the renderer (push-to-talk) or a lightweight audio
 * watcher that sends small chunks to STT and this module checks the result.
 */

'use strict';

const Fuse = require('fuse.js');
const logger = require('./logger.cjs');

const DEFAULT_WAKE_PHRASES = [
  'thinkdrop',
  'think drop',
  'hey thinkdrop',
  'hey think drop',
  'yo thinkdrop',
  'ok thinkdrop',
  'you awake',
  'are you there',
  'wake up',
  'listen up',
  'hey assistant',
  'thinky drop',
  'thinkidrop',
];

const DEFAULT_CANCEL_PHRASES = [
  'cancel that',
  'cancel this',
  'cancel task',
  'abort that',
  'abort task',
  'abort everything',
  'stop that',
  'stop everything',
  'stop the task',
  'nevermind',
  'never mind',
  'forget it',
];

const DEFAULT_STATUS_PHRASES = [
  'how is it going',
  'status',
  'what are you doing',
  "how's it going",
  'what is happening',
  'are you done',
  'still working',
  'progress',
  'how far along',
  'where are you at',
];

let _wakeFuse = null;
let _cancelFuse = null;
let _statusFuse = null;
let _initialized = false;

function _getWakePhrases() {
  const envPhrases = process.env.WAKE_WORD_PHRASES;
  if (envPhrases) {
    return envPhrases.split(',').map(p => p.trim().toLowerCase()).filter(Boolean);
  }
  return DEFAULT_WAKE_PHRASES;
}

function _getSensitivity() {
  const val = parseFloat(process.env.WAKE_WORD_SENSITIVITY || '0.4');
  return Math.min(Math.max(val, 0), 1);
}

function _buildFuse(phrases, threshold) {
  const items = phrases.map(p => ({ phrase: p }));
  return new Fuse(items, {
    keys: ['phrase'],
    threshold,
    includeScore: true,
    isCaseSensitive: false,
    minMatchCharLength: 3,
    ignoreLocation: true,
  });
}

function _ensureInitialized() {
  if (_initialized) return;
  const sensitivity = _getSensitivity();
  const wakePhrases = _getWakePhrases();

  _wakeFuse = _buildFuse(wakePhrases, sensitivity);
  _cancelFuse = _buildFuse(DEFAULT_CANCEL_PHRASES, 0.35);
  _statusFuse = _buildFuse(DEFAULT_STATUS_PHRASES, 0.4);
  _initialized = true;

  logger.info('[WakeWord] Initialized', {
    wakePhrases: wakePhrases.length,
    sensitivity,
  });
}

/**
 * Check whether the transcribed text contains a wake word.
 *
 * @param {string} transcript - Raw text from STT
 * @returns {{ detected: boolean, matchedPhrase: string|null, score: number, type: 'wake'|'cancel'|'status'|null }}
 */
function detect(transcript) {
  _ensureInitialized();

  if (!transcript || typeof transcript !== 'string') {
    return { detected: false, matchedPhrase: null, score: 0, type: null };
  }

  const text = transcript.toLowerCase().trim();

  const wakeResult = _checkFuse(_wakeFuse, text);
  if (wakeResult.detected) {
    logger.info('[WakeWord] Wake word detected', { transcript: text.substring(0, 60), matched: wakeResult.matchedPhrase });
    return { ...wakeResult, type: 'wake' };
  }

  const cancelResult = _checkFuse(_cancelFuse, text);
  if (cancelResult.detected) {
    logger.info('[WakeWord] Cancel phrase detected', { transcript: text.substring(0, 60), matched: cancelResult.matchedPhrase });
    return { ...cancelResult, type: 'cancel' };
  }

  const statusResult = _checkFuse(_statusFuse, text);
  if (statusResult.detected) {
    logger.info('[WakeWord] Status query detected', { transcript: text.substring(0, 60), matched: statusResult.matchedPhrase });
    return { ...statusResult, type: 'status' };
  }

  return { detected: false, matchedPhrase: null, score: 0, type: null };
}

function _checkFuse(fuse, text) {
  if (!fuse) return { detected: false, matchedPhrase: null, score: 0 };

  // ── Fast path: exact substring match first (Fuse scores poorly for short needle in long text) ──
  const phrases = fuse._docs ? fuse._docs.map(d => d.phrase) : [];
  for (const phrase of phrases) {
    // Short phrases (≤3 words) must match at word boundaries to avoid firing mid-sentence.
    // e.g. "wake up" should not match "I need to wake up my laptop"... well it should,
    // but "i need you" must not match "I need you to send an email".
    // Use word-boundary regex for phrases that are common English substrings.
    const wordCount = phrase.split(/\s+/).length;
    if (wordCount <= 3) {
      const escaped = phrase.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const boundary = new RegExp(`(?:^|\\s)${escaped}(?:\\s|$|[,!?.])`,'i');
      if (boundary.test(text)) {
        return { detected: true, matchedPhrase: phrase, score: 1.0 };
      }
    } else if (text.includes(phrase)) {
      return { detected: true, matchedPhrase: phrase, score: 1.0 };
    }
  }

  // ── Fuse path: word-by-word chunks for fuzzy/typo variants ──
  const words = text.split(/\s+/);
  for (let i = 0; i < words.length; i++) {
    const chunk = words.slice(Math.max(0, i - 1), i + 3).join(' ');
    const chunkResults = fuse.search(chunk);
    if (chunkResults.length > 0 && chunkResults[0].score <= _getSensitivity()) {
      return {
        detected: true,
        matchedPhrase: chunkResults[0].item.phrase,
        score: 1 - (chunkResults[0].score || 0),
      };
    }
  }

  return { detected: false, matchedPhrase: null, score: 0 };
}

/**
 * Strip the wake phrase from the transcript to get just the command.
 * e.g. "Hey ThinkDrop, what's the weather?" → "what's the weather?"
 *
 * @param {string} transcript - Full transcript
 * @param {string} matchedPhrase - The matched wake phrase
 * @returns {string} - Transcript with wake phrase removed
 */
function stripWakePhrase(transcript, matchedPhrase) {
  if (!matchedPhrase) return transcript;

  const escaped = matchedPhrase.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const cleaned = transcript
    .replace(new RegExp(escaped, 'gi'), '')
    .replace(/^[\s,!?.]+/, '')
    .trim();

  return cleaned || transcript;
}

/**
 * Reload wake phrases (e.g. after user updates settings).
 */
function reload() {
  _initialized = false;
  _wakeFuse = null;
  _cancelFuse = null;
  _statusFuse = null;
  _ensureInitialized();
}

module.exports = { detect, stripWakePhrase, reload };
