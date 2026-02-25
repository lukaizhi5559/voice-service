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

  // ── 2.5. Hard stategraph overrides — action-fetch / data-retrieval phrases ──
  // These patterns sound conversational but need real execution, not an LLM chitchat response.
  // Must come BEFORE the classifier to prevent "I need to get the weather" → fast lane.
  if (/\b(get|fetch|find|look up|check|retrieve|pull|search for|show me)\b.{0,40}\b(weather|temperature|forecast|news|stock|price|score|results|data|info|information)\b/i.test(text) ||
      /\bI need (to get|you to get|the latest|the current)\b/i.test(text) ||
      /\b(what('?s| is) the (weather|temperature|forecast|news|current))\b/i.test(text)) {
    logger.info('[Router] → STATEGRAPH LANE (action-fetch override)', { text: text.substring(0, 60) });
    return { lane: 'stategraph', reason: 'action_fetch_override', signalType: null };
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
