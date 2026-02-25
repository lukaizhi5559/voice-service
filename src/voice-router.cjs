/**
 * Voice Router — Fast Lane vs StateGraph Lane
 *
 * Every incoming voice utterance (already translated to English) is classified:
 *
 *   FAST LANE — direct WebSocket LLM hit, no MCPs
 *     - Status queries ("how's it going?", "are you done?") → peek journal
 *     - Cancel/pause/resume signals → write to journal
 *     - Pure chitchat / greetings / simple math
 *
 *   STATEGRAPH LANE — full pipeline, same as text input
 *     - Anything that needs MCPs: web search, memory read/write,
 *       screen intelligence, conversation logging, command automation
 *     - "what do you know about X" → memory retrieval
 *     - "look up / search / find" → web search
 *     - "what's on my screen" → screen intelligence
 *     - "open / click / type / automate" → command execution
 *     - All general knowledge questions (StateGraph adds web context)
 *
 * The router makes this decision by:
 *   1. Checking voice-state.json for active StateGraph run
 *   2. Running lightweight intent peek (keyword heuristics, no LLM needed)
 *   3. Routing accordingly
 *
 * Output: { lane: 'fast' | 'stategraph', reason: string }
 */

'use strict';

const logger = require('./logger.cjs');
const journal = require('./voice-journal.cjs');

// Pure chitchat/status that needs NO tools — fast WebSocket LLM only
const FAST_ONLY_PATTERNS = [
  /^(hi|hello|hey|yo|howdy)([ !.,]|$)/i,
  /^(thanks?|thank you)([ !.,]|$)/i,
  /^good (morning|afternoon|evening|night)([ !.,]|$)/i,
  /^(yes|no|ok|okay|sure|got it|sounds good)([ !.,]|$)/i,
  /\bhow are you\b/i,
  /\bwhat time is it\b/i,
  /\bnever ?mind\b/i,
  // Presence / connection checks
  /\bcan you hear (me|this)\b/i,
  /\bare you there\b/i,
  /\bare you (listening|awake|online|working|alive)\b/i,
  /\bis (this|it) working\b/i,
  /\bdo you (hear|understand) me\b/i,
  /\btest(ing)?\b.*\bone.*(two|three)\b/i,
  /\bhello.*(anyone|there)\b/i,
];

// Status/journal queries — fast lane, read from journal
const STATUS_PATTERNS = [
  /\bstatus\b/i,
  /\bhow('?s| is) it going\b/i,
  /\bwhat are you doing\b/i,
  /\bare you done\b/i,
  /\bstill working\b/i,
  /\bprogress\b/i,
  /\bhow far along\b/i,
  /\bwhere are you at\b/i,
  /\bwhat('?s| is) happening\b/i,
];

// Everything that needs MCPs → StateGraph lane
const STATEGRAPH_PATTERNS = [
  // Automation / UI control
  /\bopen\b.*(app|browser|slack|chrome|safari|mail|email|terminal)/i,
  /\b(send|write|compose|draft|reply)\b.*(message|email|text|slack|tweet|post)/i,
  /\b(click|type|fill|submit|download|upload|install)\b/i,
  /\b(create|make|build|generate|code|edit)\b.*(file|doc|script|function)/i,
  /\b(schedule|remind|set.*(alarm|timer|reminder))\b/i,
  /\b(automate|run|execute)\b/i,
  // Web search
  /\b(search|look up|look it up|find|google|bing|browse|what('?s| is) the latest)\b/i,
  // Memory
  /\b(remember|save|store|log|recall|do you know|what do you know about)\b/i,
  // Screen intelligence
  /\b(what('?s| is) on (my )?screen|read (the )?screen|what do you see|screen)\b/i,
  // General knowledge / questions that benefit from web context
  /\b(who('?s| is)|what('?s| is)|where('?s| is)|when('?s| is)|why (is|are|did|does)|how (do|does|did|can|to))\b/i,
  /\b(tell me about|explain|describe|define|summarize)\b/i,
];

/**
 * Classify the utterance and return routing decision.
 *
 * @param {string} englishText - Translated English transcript
 * @returns {{ lane: 'fast'|'stategraph', reason: string, signalType: string|null }}
 */
function route(englishText) {
  const text = englishText.trim();
  const state = journal.read();
  const sgStatus = state.stategraph?.status;
  const isRunning = sgStatus === 'running' || sgStatus === 'paused';

  // Check for cancel/pause/resume signals first — always fast lane + write signal
  if (/\b(cancel|abort|stop( that| everything)?)\b/i.test(text)) {
    logger.info('[Router] → FAST LANE (cancel signal)', { text: text.substring(0, 60) });
    return { lane: 'fast', reason: 'cancel_signal', signalType: 'cancel' };
  }

  if (/\bpause\b/i.test(text) && isRunning) {
    logger.info('[Router] → FAST LANE (pause signal)', { text: text.substring(0, 60) });
    return { lane: 'fast', reason: 'pause_signal', signalType: 'pause' };
  }

  if (/\bresume\b/i.test(text) && sgStatus === 'paused') {
    logger.info('[Router] → FAST LANE (resume signal)', { text: text.substring(0, 60) });
    return { lane: 'fast', reason: 'resume_signal', signalType: 'resume' };
  }

  // Pure chitchat/greetings → fast lane only, no MCPs needed
  if (FAST_ONLY_PATTERNS.some(p => p.test(text))) {
    logger.info('[Router] → FAST LANE (chitchat)', { text: text.substring(0, 60) });
    return { lane: 'fast', reason: 'chitchat', signalType: null };
  }

  // Status queries → fast lane, peek journal
  if (STATUS_PATTERNS.some(p => p.test(text)) && !isRunning) {
    logger.info('[Router] → FAST LANE (status query)', { text: text.substring(0, 60) });
    return { lane: 'fast', reason: 'fast_intent', signalType: null };
  }

  // StateGraph is running + status query → fast lane (journal peek, don't queue on top)
  if (isRunning && STATUS_PATTERNS.some(p => p.test(text))) {
    logger.info('[Router] → FAST LANE (sg running — journal status)', { text: text.substring(0, 60) });
    return { lane: 'fast', reason: 'sg_running_question', signalType: null };
  }

  // StateGraph is running + something else → also fast lane (don't stack requests)
  if (isRunning) {
    logger.info('[Router] → FAST LANE (sg running — intercepted)', { text: text.substring(0, 60) });
    return { lane: 'fast', reason: 'sg_running_question', signalType: null };
  }

  // Anything that needs MCPs (web search, memory, screen, automation) → StateGraph
  if (STATEGRAPH_PATTERNS.some(p => p.test(text))) {
    logger.info('[Router] → STATEGRAPH LANE (needs MCPs)', { text: text.substring(0, 60) });
    return { lane: 'stategraph', reason: 'needs_mcps', signalType: null };
  }

  // Default: StateGraph — it has full context (memory, web, conversation history).
  // Fast LLM has no memory or search. Better to let StateGraph handle ambiguous queries.
  logger.info('[Router] → STATEGRAPH LANE (default — full context)', { text: text.substring(0, 60) });
  return { lane: 'stategraph', reason: 'default', signalType: null };
}

module.exports = { route };
