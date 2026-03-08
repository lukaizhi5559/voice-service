/**
 * Voice Router — Voice-First, Single Fast Lane
 *
 * ALL utterances go through the fast lane (direct LLM response with personality).
 * The LLM response is then analyzed: if it contains a StateGraph trigger phrase
 * (meaning the AI said "let me look into that", "checking now", etc.), the original
 * user prompt is forwarded to StateGraph in the background while the LLM's spoken
 * response plays. The SG result is spoken when ready.
 *
 * Control signals (cancel/pause/resume) are still detected for journal writes,
 * but all go through fast lane for the spoken response.
 *
 * Output: { lane: 'fast', reason: string, signalType: string|null }
 */

'use strict';

const logger = require('./logger.cjs');
const journal = require('./voice-journal.cjs');

/**
 * Detect control signals and return routing info.
 * Always returns lane: 'fast' — StateGraph triggering is driven by the
 * LLM response content, not by upfront routing.
 *
 * @param {string} englishText - Translated English transcript
 * @returns {{ lane: 'fast', reason: string, signalType: string|null }}
 */
function route(englishText) {
  const text = englishText.trim();
  const state = journal.read();
  const sgStatus = state.stategraph?.status;

  // Stale-state TTL: treat 'done'/'error' older than 60s as 'idle'
  const lastUpdate = state.stategraph?.lastUpdate ? new Date(state.stategraph.lastUpdate).getTime() : 0;
  const isStale = (sgStatus === 'done' || sgStatus === 'error') && (Date.now() - lastUpdate > 60_000);
  const effectiveStatus = isStale ? 'idle' : sgStatus;
  const isRunning = effectiveStatus === 'running' || effectiveStatus === 'paused';

  // ── Control signals — detect for journal writes, still fast lane ──────────
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

  // ── All other utterances → fast lane ──────────────────────────────────────
  // The LLM will respond with personality and, if needed, a trigger phrase
  // that escalates the task to StateGraph automatically.
  logger.info('[Router] → FAST LANE (voice-first)', { text: text.substring(0, 60) });
  return { lane: 'fast', reason: 'voice_first', signalType: null };
}

/**
 * Detect whether a USER'S UTTERANCE (English translation) signals an intent
 * that requires StateGraph execution — live data, computer actions, memory lookups.
 *
 * This is run BEFORE calling the LLM so actionable requests skip the LLM
 * round-trip entirely and get a canned holding phrase immediately.
 *
 * @param {string} englishUserText - English translation of the user's speech
 * @returns {{ shouldTrigger: boolean, triggerType: string|null }}
 */
function detectUserIntentTrigger(englishUserText) {
  if (!englishUserText || !englishUserText.trim()) return { shouldTrigger: false, triggerType: null };

  const text = englishUserText.toLowerCase().trim();

  // ── Live data lookups (weather, prices, scores, news, time, stocks) ────────
  if (/\b(what('s| is) (the )?(weather|temperature|forecast|price|cost|score|news|time|date|rate|stock|bitcoin|crypto|btc|eth|market)|weather (in|for|at)|forecast (for|in)|current (price|rate|weather|temperature|score)|latest (news|price|score|update)|how much (is|does|did)|how (many|far|long|fast|old|tall|big|large)|what time|what('s| is) today|who (won|is winning|leads|scored)|when (does|did|is|was)|look up|find out|search for|tell me (about|the|what|who|when|where|how)|what are the|show me the|get me the|pull up the)\b/.test(text)) {
    return { shouldTrigger: true, triggerType: 'action_lookup' };
  }

  // ── Computer / desktop actions ─────────────────────────────────────────────
  if (/\b(open (up )?|close |quit |launch |start |stop |minimize |maximize |hide |show |switch to |go to |navigate to |type |click |press |scroll |drag |resize |move |focus |paste |copy |cut |undo |redo |install |download |upload |delete |remove |rename |save |run |execute |create (a |an |new )?|make (a |an |new )?)\b/.test(text) &&
      /\b(app|application|window|tab|browser|chrome|safari|firefox|terminal|finder|slack|zoom|notion|vscode|code|editor|file|folder|document|spreadsheet|email|message|text|button|link|menu|settings|system|desktop|screen)\b/.test(text)) {
    return { shouldTrigger: true, triggerType: 'computer_action' };
  }

  // NOTE: Memory store/retrieve and research patterns are intentionally NOT here.
  // Those are handled by the LLM via trigger phrases in detectLLMTrigger.
  // Putting ambiguous patterns here causes false positives BEFORE the LLM can
  // correct them (e.g. "save that file" → wrongly triggers memory_store).
  // The LLM is the reliable gate for anything nuanced.

  return { shouldTrigger: false, triggerType: null };
}

/**
 * Detect whether an LLM RESPONSE contains a StateGraph trigger phrase.
 * These are the exact phrases the butler prompt instructs the LLM to use
 * when it needs real execution. Run AFTER the LLM call as a safety net.
 *
 * @param {string} llmResponse
 * @returns {{ shouldTrigger: boolean, triggerType: string|null }}
 */
function detectLLMTrigger(llmResponse) {
  if (!llmResponse || !llmResponse.trim()) return { shouldTrigger: false, triggerType: null };

  const text = llmResponse.toLowerCase().trim();

  // ── LLM lookup/research acknowledgment phrases ─────────────────────────────
  if (/\b(let me (check|look|find|search|pull|grab|get|look into|look that up|look into that)|checking (on|that|now|for you)|looking (that up|into that|into it)|searching (for|now)|i'll (check|look|find|search|pull|get|handle|do that)|routing that|passing that|handling that|taking care of that)\b/.test(text)) {
    return { shouldTrigger: true, triggerType: 'action_lookup' };
  }

  // ── LLM research/fetch acknowledgment ─────────────────────────────────────
  if (/\b(let me (research|investigate|analyze|dig into|pull up)|looking that up|checking the (latest|current|live)|fetching (that|the|live)|getting (that|the latest)|one moment (while i|let me)|just a (moment|second) (while i|let me))\b/.test(text)) {
    return { shouldTrigger: true, triggerType: 'research' };
  }

  // ── LLM computer-action routing phrases ───────────────────────────────────
  if (/\b(routing that to thinkdrop|passing that along|thinkdrop will handle|on its way to thinkdrop|consider it queued|forwarding that|sending that over)\b/.test(text)) {
    return { shouldTrigger: true, triggerType: 'computer_action' };
  }

  // ── LLM memory-store acknowledgment phrases ──────────────────────
  if (/\b(filing that away|storing that|saving (that|this) to (memory|your records)|committing (that|this) to memory|noted (and stored|for the record)|locking (that|this) in|i('ll| will) (remember|store|save|note|log|record) that|making (a )?note of (that|this))\b/.test(text)) {
    return { shouldTrigger: true, triggerType: 'memory_store' };
  }

  return { shouldTrigger: false, triggerType: null };
}

/**
 * Combined trigger detection — checks an arbitrary string against both
 * user-intent patterns and LLM acknowledgment patterns.
 *
 * @param {string} text
 * @returns {{ shouldTrigger: boolean, triggerType: string|null }}
 */
function detectStateGraphTrigger(text) {
  const a = detectUserIntentTrigger(text);
  if (a.shouldTrigger) return a;
  return detectLLMTrigger(text);
}

/**
 * Classify whether StateGraph should be escalated by weighing BOTH the
 * user's English intent AND the LLM answer together.
 *
 * Logic:
 *   - User intent is PRIMARY. If detectUserIntentTrigger fires on the
 *     user prompt → escalate (LLM gave a holding answer, user clearly wants action).
 *   - LLM acknowledgment is SECONDARY CONFIRMATION. If detectLLMTrigger fires
 *     on the LLM response → escalate (LLM itself signalled it needs real execution).
 *   - Pure chitchat / opinion: neither fires → no escalation.
 *
 * Why this beats pre-LLM only or post-LLM only:
 *   - Pre-LLM only: misses ambiguous requests where context matters.
 *   - Post-LLM only: fails for non-English users where LLM responds in their
 *     language (Chinese "让我查一下" won't match English regex).
 *   - Combined: user intent regex catches the English translation regardless of
 *     response language; LLM trigger catches English confirmations as a bonus.
 *
 * @param {string} userEnglishPrompt - English translation of user speech
 * @param {string} llmAnswer         - LLM response text
 * @returns {{ shouldTrigger: boolean, triggerType: string|null }}
 */
function classifySGTrigger(userEnglishPrompt, llmAnswer) {
  // Primary: user's intent (language-agnostic via English translation)
  const intentCheck = detectUserIntentTrigger(userEnglishPrompt);
  if (intentCheck.shouldTrigger) return intentCheck;

  // Secondary: LLM explicitly signalled it needs real execution
  const llmCheck = detectLLMTrigger(llmAnswer);
  if (llmCheck.shouldTrigger) return llmCheck;

  return { shouldTrigger: false, triggerType: null };
}

module.exports = { route, detectStateGraphTrigger, detectUserIntentTrigger, detectLLMTrigger, classifySGTrigger };
