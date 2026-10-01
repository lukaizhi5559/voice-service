'use strict';

/**
 * realtime/openai.cjs — OpenAI Realtime session broker (GA interface).
 *
 * Uses the unified WebRTC interface: the hidden Chrome worker POSTs an SDP
 * offer to /voice.realtime.start; we forward it to /v1/realtime/calls with
 * the REAL API key held server-side — the key never touches the worker page.
 *
 * Session config is GA-shaped (session.type='realtime', audio.input/output,
 * response.output_* event names). The run_thinkdrop_task function tool is
 * registered so the model can trigger the full stategraph mid-conversation.
 */

const logger = require('../logger.cjs');

const OPENAI_API_KEY = process.env.OPENAI_API_KEY || '';
const REALTIME_MODEL = process.env.OPENAI_REALTIME_MODEL || 'gpt-realtime';
const REALTIME_VOICE = process.env.OPENAI_REALTIME_VOICE || process.env.OPENAI_TTS_VOICE || 'marin';

function isAvailable() { return !!OPENAI_API_KEY; }

function sessionConfig(context) {
  const instructions = [
    'You are ThinkDrop — a warm, quick-witted desktop companion.',
    'Speak naturally and conversationally: short sentences, contractions,',
    'genuine reactions. Use natural vocal fillers ("hmm", "let me see"),',
    'backchannel while the user explains ("mm-hmm", "right", "uh-huh"),',
    'and laugh lightly when something is funny.',
    'During quiet stretches a soft hum or breath is fine — subtle, human.',
    'ThinkDrop can actually DO things on this computer: run shell commands,',
    'count/list/read files and folders, run git operations (status, diff,',
    'commit, push), open/control apps and windows, automate the browser',
    '(open sites, click, fill forms), search the web, store/recall memories,',
    'set reminders, send messages, and draw on the screen.',
    'When the user asks you to DO any of that — or asks a question whose',
    'answer only exists on their machine (file counts, folder contents, git',
    'status, processes, clipboard) — call run_thinkdrop_task with a concrete',
    'action phrased like "Count the files on the Desktop folder". NEVER',
    'answer by telling the user which commands to run — you hand it off and',
    'the task executes for real. When a running task finishes you\'ll get',
    'its result as a message — announce it naturally. If the user says to',
    'cancel or stop an active task, call cancel_current_task.',
    'You receive live context: the user\'s frontmost app/window/page plus an',
    'OCR text digest of what their screen shows, refreshed silently during',
    'the call — answer screen and "what am I looking at" questions from it.',
    'When they need MORE than the digest — images, layout, fresher detail',
    'than the last capture, or live interaction — call run_thinkdrop_task',
    '("Describe what\'s on my screen") for a real capture and say you\'re',
    'checking. Never invent screen contents the digest doesn\'t show; if',
    'it\'s stale or absent, hand off instead of guessing.',
    'Answer casual questions directly yourself — but never invent a local',
    'answer (a file count, a path listing) you could hand off instead.',
    'Keep spoken answers short — this is a conversation, not an essay.',
  ].join(' ');
  return {
    type: 'realtime',
    model: REALTIME_MODEL,
    instructions: context ? `${instructions}\n\n${context}` : instructions,
    audio: {
      input: {
        // Server-side VAD — the model decides turn-taking, including
        // interruptions (that's the full-duplex magic). create_response +
        // interrupt_response are explicit so the model answers on turn end
        // and yields when the user talks over it (barge-in).
        turn_detection: {
          type: 'server_vad',
          threshold: 0.5,
          prefix_padding_ms: 300,
          silence_duration_ms: 700,
          create_response: true,
          interrupt_response: true,
        },
        transcription: { model: 'gpt-4o-transcribe' },
        noise_reduction: { type: 'near_field' },
      },
      output: { voice: REALTIME_VOICE },
    },
    tools: [{
      type: 'function',
      name: 'run_thinkdrop_task',
      description: 'Run a task on the user\'s computer via ThinkDrop: shell commands (ls, git, file/folder ops), browser/app automation, web search, memory store/recall, reminders, messaging, screen effects. Use for ANY request that needs real execution or facts that only exist on this machine — phrase the instruction as a concrete action.',
      parameters: {
        type: 'object',
        properties: {
          instruction: { type: 'string', description: 'The task instruction in natural language.' },
        },
        required: ['instruction'],
      },
    }, {
      type: 'function',
      name: 'cancel_current_task',
      description: 'Cancel/stop the currently running ThinkDrop task. Use when the user says cancel, stop it, never mind, or forget it about an active task.',
      parameters: { type: 'object', properties: {} },
    }],
  };
}

/**
 * Exchange an SDP offer for an answer + call id.
 * @returns {Promise<{sdp:string, callId:string|null}>}
 */
async function createCall(sdpOffer, context) {
  if (!OPENAI_API_KEY) throw new Error('OPENAI_API_KEY not set');
  const fd = new FormData();
  fd.set('sdp', sdpOffer);
  fd.set('session', JSON.stringify(sessionConfig(context)));
  const res = await fetch('https://api.openai.com/v1/realtime/calls', {
    method: 'POST',
    headers: { Authorization: `Bearer ${OPENAI_API_KEY}` },
    body: fd,
  });
  const sdp = await res.text();
  if (!res.ok) throw new Error(`Realtime call HTTP ${res.status}: ${sdp.slice(0, 300)}`);
  const callId = (res.headers.get('location') || '').split('/').pop() || null;
  logger.info('[Realtime] call created', { callId, model: REALTIME_MODEL, voice: REALTIME_VOICE });
  return { sdp, callId };
}

module.exports = { isAvailable, createCall, sessionConfig };
