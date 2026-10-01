'use strict';

/**
 * filler-policy.cjs — decides WHICH filler (if any) to play for a voice event.
 *
 * Rules follow conversation-analysis findings:
 *  - Turn-gap mode is ~0-200ms → fillers fire immediately on `final`.
 *  - Backchannels belong at transition-relevance places (end of complete
 *    syntactic units + pauses); badly-timed ones actively disrupt speakers.
 *  - Generic acknowledgments ("mm-hmm") and reactive ones ("oh wow") are
 *    different classes — separate categories, separate triggers.
 *
 * Rate-limiting + streak state live here; the module is per-service-singleton.
 */

const path = require('path');
const logger = require('./logger.cjs');

// comms-graph's regex intent guesser — pure functions, safe to share.
let guessIntent = () => ({ guessedIntent: null });
try {
  guessIntent = require(path.join(__dirname, '..', '..', '..', 'comms-graph', 'src', 'intentGuesser.cjs')).guess;
} catch (err) {
  logger.warn('[FillerPolicy] intentGuesser unavailable', { error: err.message });
}

// ── Tunables ──────────────────────────────────────────────────────────────────
const MIN_FILLER_GAP_MS = 400;   // never fire two fillers closer than this
const MAX_ACK_STREAK = 3;        // after N consecutive acks, go quiet (humans do)
const CHAIN_GAP_MS = 1200;       // finals closer than this = rapid instruction chain
const REACTIVE_CHANCE = 0.35;    // don't emotionally react to everything
const BACKCHANNEL_MIN_INTERVAL_MS = 5000; // mid-monologue mm-hmm throttle
const BACKCHANNEL_ENABLED = process.env.VOICE_BACKCHANNEL !== '0'; // on by default

// ── Text classification ───────────────────────────────────────────────────────
// Trailing continuation signals — the user clearly has more to say.
const MID_CHAIN_END_RE = /(?:\b(?:and|but|so|then|also|plus|because|or)\s*$|[,;]\s*$)/i;
const MID_CHAIN_PHRASE_RE = /\b(and then|after that|and also|don't forget|make sure|oh and|one more thing|first|second|next|finally)\b/i;
const QUESTION_RE = /^(?:who|what|whats|what's|where|when|why|how|can|could|would|should|is|are|do|does|did|which)\b|\?\s*$/i;
const SLEEP_CMD_RE = /\b(go to sleep|sleep mode|that's all|that is all|be quiet|go quiet|stop listening|take a break|rest now)\b/i;
const WAKE_CMD_RE = /\b(wake up|hey\s+think\s*drop|think\s*drop|armis|are you (there|awake)|you there)\b/i;
const AFFIRM_RE = /^(yes|yeah|yep|yup|sure|ok|okay|do it|go ahead|send it|approve|sounds good|absolutely)\b/i;
const NEGATE_RE = /^(no|nope|nah|don't|do not|cancel|never mind|stop)\b/i;
const GREETING_RE = /^(hi|hey|hello|yo|good (morning|afternoon|evening)|hey there)\b/i;
const EMO_POS_RE = /\b(amazing|awesome|great|love it|nice|cool|funny|hilarious|perfect|beautiful|excellent|fantastic|brilliant)\b/i;
const EMO_NEG_RE = /\b(terrible|awful|hate|sucks|sucked|broken|stupid|worst|annoying|frustrat|disappointed|ugh)\b/i;

// ── Stateful tracker ──────────────────────────────────────────────────────────
const _state = {
  lastFillerAt: 0,
  ackStreak: 0,
  lastFinalAt: 0,
  lastBackchannelAt: 0,
};

function reset() {
  _state.lastFillerAt = 0;
  _state.ackStreak = 0;
  _state.lastFinalAt = 0;
  _state.lastBackchannelAt = 0;
}

function _allowed() {
  return Date.now() - _state.lastFillerAt >= MIN_FILLER_GAP_MS;
}
function _used(cat) {
  _state.lastFillerAt = Date.now();
  if (cat === 'ack' || cat === 'ack_continue') _state.ackStreak++;
  else _state.ackStreak = 0;
}

function isMidChain(text) {
  return MID_CHAIN_END_RE.test(text) || MID_CHAIN_PHRASE_RE.test(text);
}

/**
 * Decide for an incoming `final` transcript.
 * @returns {{filler:string|null, holdTurn:boolean, intent:string|null, cls:string}}
 *   filler — category to play (null = silence)
 *   holdTurn — true → user is mid-chain; aggregate, don't dispatch yet
 */
function decideFinal(text) {
  const now = Date.now();
  const gap = now - _state.lastFinalAt;
  _state.lastFinalAt = now;
  const t = (text || '').trim();
  const cls = t.match(GREETING_RE) ? 'greeting'
    : t.match(SLEEP_CMD_RE) ? 'sleep'
    : t.match(AFFIRM_RE) ? 'affirm'
    : t.match(NEGATE_RE) ? 'negative'
    : t.match(QUESTION_RE) ? 'question'
    : 'statement';

  // Wake phrases bypass filler logic — the sleep layer owns them.
  if (WAKE_CMD_RE.test(t)) return { filler: null, holdTurn: false, intent: null, cls: 'wake' };

  const midChain = isMidChain(t) || gap < CHAIN_GAP_MS && cls === 'statement';
  const { guessedIntent } = cls === 'affirm' || cls === 'negative' ? { guessedIntent: null } : guessIntent(t);

  let filler = null;
  if (_allowed()) {
    if (cls === 'sleep') filler = 'going_to_sleep';
    else if (cls === 'greeting') filler = 'greeting';
    else if (cls === 'affirm') filler = 'yes';
    else if (cls === 'negative') filler = 'no';
    else if (_state.ackStreak < MAX_ACK_STREAK) {
      if (midChain) filler = 'ack_continue';
      else if (guessedIntent) filler = `intent_${guessedIntent}`;
      else if (cls === 'question') filler = 'thinking';
      else if (EMO_POS_RE.test(t) && Math.random() < REACTIVE_CHANCE) filler = 'reactive_positive';
      else if (EMO_NEG_RE.test(t) && Math.random() < REACTIVE_CHANCE) filler = 'reactive_negative';
      else filler = 'ack';
    }
  }
  if (filler) _used(filler);
  return { filler, holdTurn: midChain, intent: guessedIntent, cls };
}

/**
 * Should a soft backchannel ("mm-hmm") fire during a user pause?
 * Called on worker `speech-end`/`sound-end` events while no final arrived —
 * a pause inside the user's turn is the classic backchannel slot.
 */
function decideBackchannel() {
  if (!BACKCHANNEL_ENABLED) return false;
  const now = Date.now();
  if (now - _state.lastBackchannelAt < BACKCHANNEL_MIN_INTERVAL_MS) return false;
  if (now - _state.lastFillerAt < MIN_FILLER_GAP_MS) return false;
  _state.lastBackchannelAt = now;
  _used('backchannel');
  return true;
}

module.exports = { decideFinal, decideBackchannel, isMidChain, reset, _state };
