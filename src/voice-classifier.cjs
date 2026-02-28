'use strict';

/**
 * Voice Classifier — Fast Lane vs StateGraph Lane
 *
 * Uses the same Xenova/all-MiniLM-L6-v2 embedding model as phi4-service
 * to classify utterances into two routing buckets:
 *
 *   'fast'        — chitchat, greetings, identity questions, presence checks,
 *                   status queries, control signals
 *   'stategraph'  — anything needing MCPs: web search, memory, screen
 *                   intelligence, command automation, general knowledge
 *
 * Classification is cosine similarity against pre-computed seed embeddings.
 * Model is loaded once at startup (~18s), then each classification is ~40ms.
 *
 * Falls back to 'stategraph' on any error (safe default — full context).
 */

const path = require('path');
const logger = require('./logger.cjs');

// Point Xenova cache at the shared phi4-service model directory so we don't
// download the model twice. If that path doesn't exist we fall back to the
// default cache inside the voice-service directory.
const PHI4_MODELS_DIR = path.join(
  __dirname,
  '../../thinkdrop-phi4-service/models'
);

let _pipeline = null;

async function getPipeline() {
  if (_pipeline) return _pipeline;
  // Dynamic import — @xenova/transformers is ESM internally
  const { pipeline, env } = await import('@xenova/transformers');
  try {
    const fs = require('fs');
    if (fs.existsSync(PHI4_MODELS_DIR)) {
      env.cacheDir = PHI4_MODELS_DIR;
    }
  } catch (_) {}
  env.allowLocalModels = true;
  _pipeline = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
    quantized: true,
  });
  return _pipeline;
}

// ── Seed examples per bucket ──────────────────────────────────────────────────

const FAST_SEEDS = [
  // Greetings
  'Hello', 'Hi there', 'Hey', 'Good morning', 'Good afternoon', 'Good evening',
  'Yo', 'Howdy', 'Hey how are you', 'Thanks a lot', 'Thank you', 'Cheers',
  // Presence / connection
  'Can you hear me', 'Are you there', 'Are you listening', 'Is this working',
  'Do you understand me', 'Are you awake', 'Are you online',
  // Identity / persona
  "What's your name", 'What is your name', 'Who are you', 'What are you',
  'Tell me about yourself', 'What can you do', 'What do you do',
  'Are you an AI', 'Are you a bot', 'Are you human', 'Do you have a name',
  'What should I call you', 'How old are you', 'What is your purpose',
  'What is your role', 'Do you think', 'Do you feel anything',
  // Status / progress
  'What is the status', 'How is it going', 'Are you done',
  'Still working on it', 'What are you doing', 'How far along are you',
  "What's happening", 'Any progress',
  // Control confirmations
  'Yes', 'No', 'Okay', 'Sure', 'Got it', 'Sounds good', 'Never mind',
  'Awesome', 'Great', 'Perfect', 'I see', 'I understand', 'Understood',
  // Simple math / time
  'What time is it', 'What day is it today',
  "What's two plus two", 'How many days in a week',
];

const STATEGRAPH_SEEDS = [
  // Web search / fetch / get data
  'Search the web for the latest news on AI', 'Look up the weather in New York',
  'Find the best restaurants near me', 'Google who won the Super Bowl',
  'What is the latest version of React', 'Browse to the Apple website',
  // Action-fetch — "I need to get X", "Fetch X", "Get me the latest X"
  'I need to get the latest weather', 'Get me the current weather in PA',
  'Fetch the weather for Philadelphia', 'Get the latest news', 'Fetch me the latest',
  'I need you to get the weather', 'Get me the weather right now',
  'Fetch the current temperature outside', 'What is the weather like in Pennsylvania',
  'Get me the weather in Massachusetts', 'I need the latest weather report',
  'Look up the current weather', 'Check the weather for me', 'Get weather data',
  // Memory — temporal history lookups (most likely to be misclassified as fast)
  'What was I doing two hours ago', 'What did I do today',
  'What was I working on this morning', 'What have I been doing for the last hour',
  'What was I doing an hour ago', 'What did I do yesterday',
  'What was I doing earlier', 'What have I been up to today',
  'What did I work on this afternoon', 'What was my last task',
  'What was I doing four hours ago', 'What am I supposed to be doing right now',
  'Remind me what I was doing before', 'What have I been focused on lately',
  'What was I doing at noon', 'What happened earlier today',
  // Memory — store/recall
  'Remember I have a meeting tomorrow at 3pm', 'Save my Wi-Fi password',
  'Do you know my name', 'What do you know about my schedule',
  'Recall what I told you last week', 'Store this note for later',
  // Screen intelligence
  'What do you see on my screen', "What's on my screen right now",
  'Read the text on screen', 'Analyze this window', 'What app is open',
  'What does the screen show', 'Describe what is visible',
  // Command automation
  'Open Chrome and go to Gmail', 'Click the submit button',
  'Type hello in the text box', 'Download this file',
  'Install the latest updates', 'Run the build script',
  'Create a new file called notes.txt', 'Schedule a meeting for Friday',
  // General knowledge / research
  'Who is the president of France', 'What is quantum computing',
  'Explain how machine learning works', 'Tell me about the history of Rome',
  'Why is the sky blue', 'How does photosynthesis work',
  'What is the capital of Australia', 'Summarize this article',
  'Define entropy in physics', 'What is the GDP of Japan',
  // Conversation / memory store
  'Send an email to John about the project', 'Write a message to the team',
  'Compose a tweet about productivity', 'Draft a reply to that email',
  // Personal memory retrieval
  "What's my name", 'What is my name', 'Who am I', 'What do you know about me',
  'Tell me about myself', 'What is my email', 'What is my job', 'What is my age',
  'Do you remember my name', 'What have I told you about me',
  // Skill / capability listing
  'List all the skills you have', 'Show me all your skills', 'List skills',
  'What skills do you have', 'What commands can you run', 'What can you automate',
  'Show me available tools', 'What tools are available', 'List capabilities',
  // Action-execution commands (typing, saving, creating files)
  'Type hello world', 'Type out hello world', 'Type this for me',
  'Save a file on my desktop', 'Create a new file', 'Write a text file',
  'Open Chrome', 'Click the button', 'Scroll down', 'Press enter',
  'Run this command', 'Execute the script', 'Launch the app',
  'Download this file', 'Install the package', 'Create a folder',
  // Short-form computer actions — most likely to be misclassified as fast/chitchat
  'Open YouTube', 'Open Spotify', 'Open Slack', 'Open Calendar', 'Open Finder',
  'Open my email', 'Open Gmail', 'Open my browser', 'Open a new tab',
  'Launch Chrome', 'Launch Safari', 'Launch Terminal', 'Launch VS Code',
  'Close Slack', 'Close Chrome', 'Close this window', 'Quit the app',
  'Switch to Chrome', 'Switch to Slack', 'Switch to Terminal',
  'Go to YouTube', 'Go to my calendar', 'Go to Gmail', 'Navigate to Google',
  'Pull up Spotify', 'Bring up Finder', 'Show me my calendar',
  'Scroll up', 'Scroll down a lot', 'Scroll to the top', 'Scroll to the bottom',
  'Zoom in', 'Zoom out', 'Make it bigger', 'Make the text larger',
  'Copy that', 'Paste it', 'Select all', 'Undo that', 'Press tab',
];

// ── Classifier state ──────────────────────────────────────────────────────────

let _initialized = false;
let _initializing = false;
let _initPromise = null;
let _fastEmbeddings = null;
let _stategraphEmbeddings = null;

async function embed(embedder, text) {
  const out = await embedder(text, { pooling: 'mean', normalize: true });
  return Array.from(out.data);
}

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na  += a[i] * a[i];
    nb  += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-10);
}

function meanScore(queryEmb, seedEmbs) {
  const scores = seedEmbs.map(s => cosine(queryEmb, s));
  // Use mean of top-5 to avoid outlier seeds dragging the score down
  scores.sort((a, b) => b - a);
  const top = scores.slice(0, 5);
  return top.reduce((s, v) => s + v, 0) / top.length;
}

async function _initialize() {
  const t = Date.now();
  logger.info('[VoiceClassifier] Loading embedding model...');
  const embedder = await getPipeline();

  logger.info('[VoiceClassifier] Computing seed embeddings...');
  _fastEmbeddings = await Promise.all(FAST_SEEDS.map(s => embed(embedder, s)));
  _stategraphEmbeddings = await Promise.all(STATEGRAPH_SEEDS.map(s => embed(embedder, s)));

  _initialized = true;
  logger.info(`[VoiceClassifier] Ready in ${Date.now() - t}ms`);
}

/**
 * Eagerly initialize in the background. Call at service startup.
 */
function warmUp() {
  if (_initialized || _initializing) return;
  _initializing = true;
  _initPromise = _initialize().catch(err => {
    logger.error('[VoiceClassifier] Warmup failed', { error: err.message });
    _initializing = false;
  });
}

/**
 * Classify utterance into 'fast' or 'stategraph'.
 *
 * @param {string} text - English utterance
 * @returns {Promise<{ lane: 'fast'|'stategraph', fastScore: number, sgScore: number }>}
 */
async function classify(text) {
  try {
    if (!_initialized) {
      if (_initPromise) {
        await _initPromise;
      } else {
        warmUp();
        await _initPromise;
      }
    }

    if (!_initialized) {
      logger.warn('[VoiceClassifier] Not ready — defaulting to stategraph');
      return { lane: 'stategraph', fastScore: 0, sgScore: 0 };
    }

    const embedder = await getPipeline();
    const queryEmb = await embed(embedder, text);

    const fastScore = meanScore(queryEmb, _fastEmbeddings);
    const sgScore   = meanScore(queryEmb, _stategraphEmbeddings);

    // Bias toward stategraph when scores are close — fast lane should only win
    // with a clear margin. Prevents action commands (score diff ~0.04) from
    // slipping into fast lane and getting an LLM description instead of execution.
    const FAST_BIAS = 0.10;
    const lane = (fastScore - sgScore) > FAST_BIAS ? 'fast' : 'stategraph';

    logger.info('[VoiceClassifier] Classification', {
      text: text.substring(0, 60),
      fastScore: fastScore.toFixed(3),
      sgScore: sgScore.toFixed(3),
      lane,
    });

    return { lane, fastScore, sgScore };
  } catch (err) {
    logger.error('[VoiceClassifier] classify() error', { error: err.message });
    return { lane: 'stategraph', fastScore: 0, sgScore: 0 };
  }
}

// ── Action-confirmation seeds — what a fast-lane LLM says when it PRETENDS to act ──
// These are responses that sound like the LLM is about to do something but won't.
// If the fast lane returns text matching these patterns, the original user prompt
// should be escalated to stategraph for actual execution.
const ACTION_CONFIRM_SEEDS = [
  // "Consider it done" variants
  'Consider it done', 'Consider it handled', 'Done', 'All done', 'It is done',
  'That is done', 'Done for you', 'Done right away',
  // "On it" / "Right away" variants
  'On it', "I'm on it", 'Right away', 'On it right away', 'Right on it',
  'Getting on it now', 'Already on it', 'Absolutely, on it',
  // "I will / I'll do that" variants
  "I'll do that", "I'll do that for you", "I'll handle that", "I'll take care of it",
  "I'll take care of that", "I'll get that done", "I'll get right on it",
  "I'll close that", "I'll open that", "I'll run that", "I'll execute that",
  "I will do that", "I will handle that", "I will take care of that",
  "I will close it", "I will open it", "I will run it",
  // "Doing it now" / "Closing/Opening now"
  'Doing it now', 'Closing now', 'Closing it now', 'Opening now', 'Opening it now',
  'Running now', 'Executing now', 'Launching now', 'Launching it now',
  'Closing Zoom now', 'Opening Chrome now', 'Switching now',
  // "Let me X" variants
  'Let me do that', 'Let me handle that', 'Let me take care of that',
  'Let me close that', 'Let me open that', 'Let me run that',
  'Let me get that done', 'Let me get right on it',
  // "Completing / Executing / Processing"
  'Completing that now', 'Executing that now', 'Processing your request',
  'Processing that request', 'Handling that now', 'Taking care of that now',
  // "Sure / Absolutely / Of course" + action
  'Sure, closing that', 'Sure, opening that', 'Absolutely, closing that',
  'Of course, right away', 'Certainly, on it', 'Sure thing, on it',
  // Stall + action intention
  'One moment', 'One moment please', 'Just a moment', 'Hold on',
  'Working on it', 'Performing that action', 'Carry out that request',
  // Tool-specific action verbs (in-progress)
  'Updating now', 'Searching now', 'Sending now', 'Scheduling now',
  'Downloading now', 'Uploading now', 'Installing now', 'Deleting now',
  'Creating now', 'Moving now', 'Copying now', 'Saving now',
  'Running the command', 'Executing the script', 'Launching the app',
  'Submitting now', 'Posting now', 'Filing now', 'Booking now',
  // Common LLM action-speak phrases
  "I've got you covered", 'Consider it vanished', 'As you wish',
  "I'll see to it", "I shall handle it", "Executing your command",
  "Your request is being processed", "Initiating that now",
  "That will be done", "Carrying that out now", "On it straight away",
  // Butler routing acknowledgments — exact phrases from fast-lane-butler.md prompt
  "Routing that to ThinkDrop now", "Passing that along to ThinkDrop",
  "ThinkDrop will handle that", "On its way to ThinkDrop",
  "Passing that along", "Consider it queued",
];

// Chitchat/informational response seeds — the LLM is ANSWERING, not acting
const ANSWER_SEEDS = [
  // Information delivery
  "Here's what I found", 'The answer is', 'According to', 'Based on',
  'I believe', 'I think', 'In my opinion', 'From what I know',
  // Conversational replies
  "I'm doing well", "I'm fine", "I'm great", "I'm just an AI",
  "I don't have feelings", "I can't do that", "I don't know",
  'That is a great question', 'Good question', "I'm not sure",
  "I don't have access", "I'm unable to", "I cannot",
  // Explanations
  'The reason is', 'This is because', 'To explain', 'In short',
  'Basically', 'The way it works is', 'Think of it as',
  // Greetings/small talk
  'Hello', 'Hi there', 'Hey', 'Good to hear from you', 'Nice to meet you',
  'How can I help you', 'What can I do for you', 'How are you doing',
  // Memory/context answers
  "You've been working on", "According to your history", "I remember you said",
  "Based on our conversation", "Your last task was", "Earlier you mentioned",
  // Polite refusals / capability statements — NOT actions, must not escalate
  "I'm sorry, I can't do that", "I don't have permission to do that",
  "I'm unable to perform that action", "That's outside my capabilities",
  "I cannot access that", "I don't have the ability to",
  "I can do that for you", "That is something I can help with",
  "Sure, I can help with that", "I'd be happy to help with that",
];

let _actionConfirmEmbeddings = null;
let _answerEmbeddings = null;

async function _ensureActionConfirmEmbeddings() {
  if (_actionConfirmEmbeddings) return;
  // Reuse the already-initialized pipeline
  if (!_initialized) {
    if (_initPromise) await _initPromise;
    else { warmUp(); await _initPromise; }
  }
  const embedder = await getPipeline();
  [_actionConfirmEmbeddings, _answerEmbeddings] = await Promise.all([
    Promise.all(ACTION_CONFIRM_SEEDS.map(s => embed(embedder, s))),
    Promise.all(ANSWER_SEEDS.map(s => embed(embedder, s))),
  ]);
}

/**
 * Classify an LLM response to determine if it's an action confirmation
 * (fast lane pretended to act but didn't) vs an actual answer.
 *
 * Returns { isActionConfirmation: boolean, actionScore: number, answerScore: number }
 * If isActionConfirmation is true, the original user prompt should be escalated
 * to stategraph for actual execution.
 *
 * @param {string} responseText - The LLM's response text from fast lane
 * @returns {Promise<{ isActionConfirmation: boolean, actionScore: number, answerScore: number }>}
 */
async function classifyLLMResponse(responseText) {
  try {
    await _ensureActionConfirmEmbeddings();
    const embedder = await getPipeline();

    // LLMs front-load their intent — "That sounds like a plan! I'll get right on it."
    // Classifying the full text lets conversational fluff dilute the action signal.
    // Extract just the first sentence (up to first '.', '!', or '?').
    const firstSentenceMatch = responseText.match(/^[^.!?]+[.!?]/);
    const classifyText = firstSentenceMatch
      ? firstSentenceMatch[0].trim()
      : responseText.substring(0, 120).trim();

    const queryEmb = await embed(embedder, classifyText);

    const actionScore = meanScore(queryEmb, _actionConfirmEmbeddings);
    const answerScore = meanScore(queryEmb, _answerEmbeddings);

    // Action confirmation wins if it leads by > 0.05
    const ACTION_BIAS = 0.05;
    const isActionConfirmation = (actionScore - answerScore) > ACTION_BIAS;

    logger.info('[VoiceClassifier] LLM response classification', {
      classifiedText: classifyText.substring(0, 80),
      actionScore: actionScore.toFixed(3),
      answerScore: answerScore.toFixed(3),
      isActionConfirmation,
    });

    return { isActionConfirmation, actionScore, answerScore };
  } catch (err) {
    logger.error('[VoiceClassifier] classifyLLMResponse() error', { error: err.message });
    return { isActionConfirmation: false, actionScore: 0, answerScore: 0 };
  }
}

module.exports = { warmUp, classify, classifyLLMResponse };
