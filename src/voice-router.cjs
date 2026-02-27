/**
 * Voice Router — Fast Lane vs StateGraph Lane
 *
 * Every incoming voice utterance (already translated to English) is classified:
 *
 *   FAST LANE — direct WebSocket LLM hit, no MCPs
 *     - Cancel/pause/resume control signals (hard regex — unambiguous)
 *     - Chitchat, greetings, identity questions, status queries
 *       → classified by Xenova embedding model (voice-classifier.cjs)
 *
 *   STATEGRAPH LANE — full pipeline, same as text input
 *     - Web search, memory, screen intelligence, command automation
 *     - General knowledge questions (StateGraph adds web + memory context)
 *     - Default when classifier is uncertain
 *
 * Decision flow:
 *   1. Hard control signals (cancel/pause/resume) — 3 regexes, no model needed
 *   2. Journal state check (SG running → fast lane to avoid request stacking)
 *   3. Xenova embedding classifier → fast vs stategraph
 *
 * Output: { lane: 'fast' | 'stategraph', reason: string, signalType: string|null }
 */

'use strict';

const logger = require('./logger.cjs');
const journal = require('./voice-journal.cjs');
const classifier = require('./voice-classifier.cjs');

/**
 * Classify the utterance and return routing decision.
 * This function is now async because the classifier uses an embedding model.
 *
 * @param {string} englishText - Translated English transcript
 * @returns {Promise<{ lane: 'fast'|'stategraph', reason: string, signalType: string|null }>}
 */
async function route(englishText) {
  const text = englishText.trim();
  const state = journal.read();
  const sgStatus = state.stategraph?.status;

  // Stale-state TTL: treat 'done'/'error' older than 60s as 'idle'
  const lastUpdate = state.stategraph?.lastUpdate ? new Date(state.stategraph.lastUpdate).getTime() : 0;
  const isStale = (sgStatus === 'done' || sgStatus === 'error') && (Date.now() - lastUpdate > 60_000);
  const effectiveStatus = isStale ? 'idle' : sgStatus;
  const isRunning = effectiveStatus === 'running' || effectiveStatus === 'paused';

  // ── 1. Hard control signals — always fast lane, unambiguous ───────────────
  if (/\b(cancel (that|this|task|it)|abort (that|task|everything)|stop (that|everything|the task)|nevermind|never mind|forget it)\b/i.test(text)) {
    logger.info('[Router] → FAST LANE (cancel signal)', { text: text.substring(0, 60) });
    return { lane: 'fast', reason: 'cancel_signal', signalType: 'cancel' };
  }

  if (/\bpause\b/i.test(text) && isRunning) {
    logger.info('[Router] → FAST LANE (pause signal)', { text: text.substring(0, 60) });
    return { lane: 'fast', reason: 'pause_signal', signalType: 'pause' };
  }

  if (/\bresume\b/i.test(text) && effectiveStatus === 'paused') {
    logger.info('[Router] → FAST LANE (resume signal)', { text: text.substring(0, 60) });
    return { lane: 'fast', reason: 'resume_signal', signalType: 'resume' };
  }

  // ── 2. StateGraph running → intercept to fast lane (don't stack requests) ──
  if (isRunning) {
    logger.info('[Router] → FAST LANE (sg running — intercepted)', { text: text.substring(0, 60) });
    return { lane: 'fast', reason: 'sg_running_question', signalType: null };
  }

  // ── 2.5. Hard stategraph overrides — NARROW, unambiguous patterns only ───────
  // Rule: only add a hard override if the false-positive risk is near zero.
  // For anything ambiguous, let the classifier decide (see step 3 bias below).

  // Personal attribute lookup — "what's my name/email/age/etc."
  // Very low FP risk: "what's my X" almost always means fetch from memory.
  if (/\bwhat('?s| is) my (name|age|birthday|email|address|phone|job|company)\b/i.test(text)) {
    logger.info('[Router] → STATEGRAPH LANE (personal-memory override)', { text: text.substring(0, 60) });
    return { lane: 'stategraph', reason: 'personal_memory_override', signalType: null };
  }

  // Explicit skill listing — "list skills", "show me your skills", etc.
  // FP risk is low: these phrases are almost never casual chitchat.
  if (/\b(list|show me)\b.{0,20}\b(skills?|capabilities|commands|tools)\b/i.test(text)) {
    logger.info('[Router] → STATEGRAPH LANE (skill-list override)', { text: text.substring(0, 60) });
    return { lane: 'stategraph', reason: 'skill_list_override', signalType: null };
  }

  // Data-fetch — "get/fetch/check the weather/news/etc." with an explicit data noun
  if (/\b(get|fetch|find|look up|check|retrieve|search for)\b.{0,40}\b(weather|temperature|forecast|news|stock|price|score|results)\b/i.test(text) ||
      /\bwhat('?s| is) the (weather|temperature|forecast|news)\b/i.test(text)) {
    logger.info('[Router] → STATEGRAPH LANE (data-fetch override)', { text: text.substring(0, 60) });
    return { lane: 'stategraph', reason: 'data_fetch_override', signalType: null };
  }

  // Computer action override — "open X", "launch X", "go to X", "close X", "switch to X"
  // These are always command_automate. ML classifier often misroutes them as chitchat.
  if (/\b(open|launch|start|close|quit|switch to|go to|navigate to|pull up|bring up)\s+\S/i.test(text) &&
      !/\b(how (do|can|would)|what is|tell me|explain|why)\b/i.test(text)) {
    logger.info('[Router] → STATEGRAPH LANE (computer-action override)', { text: text.substring(0, 60) });
    return { lane: 'stategraph', reason: 'computer_action_override', signalType: null };
  }

  // ── 3. Embedding classifier → fast vs stategraph ──────────────────────────
  const result = await classifier.classify(text);
  const reason = result.lane === 'fast' ? 'classifier_fast' : 'classifier_stategraph';
  logger.info(`[Router] → ${result.lane.toUpperCase()} LANE (${reason})`, {
    text: text.substring(0, 60),
    fastScore: result.fastScore,
    sgScore: result.sgScore,
  });
  return { lane: result.lane, reason, signalType: null };
}

module.exports = { route };
