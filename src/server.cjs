/**
 * Voice Service MCP Server
 *
 * Isolated voice MCP service for ThinkDrop AI.
 * Handles STT, TTS, wake word detection, language translation,
 * StateGraph journal integration, and voice routing.
 *
 * HTTP endpoints:
 *   POST /voice.transcribe     — STT: audio buffer → text + language
 *   POST /voice.speak          — TTS: text → audio buffer (base64 mp3)
 *   POST /voice.process        — Full pipeline: audio → translate → route → respond → TTS
 *   POST /voice.signal         — Write a signal (cancel/pause/resume/inject) to voice journal
 *   GET  /voice.status         — Current voice + StateGraph status from journal
 *   POST /voice.setStatus      — Update voice activation status
 *   GET  /health               — Service health check
 */

'use strict';

require('dotenv').config();

const http = require('http');
const crypto = require('crypto');
const WebSocket = require('ws');
const logger = require('./logger.cjs');
const stt = require('./elevenlabs-stt.cjs');
const tts = require('./elevenlabs-tts.cjs');
const { toEnglish, fromEnglish, normalizeLanguage } = require('./language-translate.cjs');
const wakeWord = require('./wake-word.cjs');
const router = require('./voice-router.cjs');
const journal = require('./voice-journal.cjs');

const PORT = parseInt(process.env.PORT || '3006', 10);
const SERVICE_NAME = process.env.SERVICE_NAME || 'voice-service';

const THINKDROP_MAIN_PORT = parseInt(process.env.THINKDROP_MAIN_PORT || '3010', 10);
const STATEGRAPH_WS_URL = process.env.STATEGRAPH_WS_URL || 'ws://localhost:4000/ws/stream';
const STATEGRAPH_API_KEY = process.env.STATEGRAPH_API_KEY || '';

class VoiceServiceMCPServer {
  constructor() {
    this.serviceName = SERVICE_NAME;
    journal.setVoiceStatus('idle');
    logger.info('VoiceServiceMCPServer initialized', { serviceName: this.serviceName });
  }

  // ─── STT ─────────────────────────────────────────────────────────────────────

  /**
   * Transcribe audio to text.
   * @param {{ audioBase64: string, format?: string, languageHint?: string }} args
   */
  async transcribe(args) {
    const { audioBase64, format = 'wav', languageHint = null } = args || {};
    if (!audioBase64) {
      return { success: false, error: 'audioBase64 is required' };
    }

    try {
      journal.setVoiceStatus('processing');
      const result = await stt.transcribeBase64({ audioBase64, format, languageHint });
      journal.setVoiceStatus('idle', { detectedLanguage: result.language });

      return {
        success: true,
        text: result.text,
        language: result.language,
        confidence: result.confidence,
        isFinal: result.isFinal,
      };
    } catch (error) {
      journal.setVoiceStatus('idle');
      logger.error('[Server] STT error', { error: error.message });
      return { success: false, error: error.message };
    }
  }

  // ─── TTS ─────────────────────────────────────────────────────────────────────

  /**
   * Synthesize text to speech.
   * @param {{ text: string, language?: string, voiceId?: string, stream?: boolean }} args
   */
  async speak(args) {
    const { text, language = 'en', voiceId, stability, similarity, style } = args || {};
    if (!text) {
      return { success: false, error: 'text is required' };
    }

    try {
      journal.setVoiceStatus('speaking');
      const result = await tts.synthesize({ text, language, voiceId, stability, similarity, style });
      journal.setVoiceStatus('idle', { lastSpokenAt: new Date().toISOString() });

      return {
        success: true,
        audioBase64: result.audioBuffer.toString('base64'),
        format: result.format,
        durationEstimateMs: result.durationEstimateMs,
        language,
      };
    } catch (error) {
      journal.setVoiceStatus('idle');
      logger.error('[Server] TTS error', { error: error.message });
      return { success: false, error: error.message };
    }
  }

  // ─── Full Voice Pipeline ──────────────────────────────────────────────────────

  /**
   * Full pipeline: audio → STT → wake word check → translate → route → respond → TTS
   *
   * @param {{
   *   audioBase64: string,
   *   format?: string,
   *   languageHint?: string,
   *   skipWakeWordCheck?: boolean,
   *   pushToTalk?: boolean,
   *   sessionId?: string,
   * }} args
   */
  async process(args) {
    const {
      audioBase64,
      format = 'wav',
      languageHint = null,
      skipWakeWordCheck = false,
      pushToTalk = false,
      sessionId,
    } = args || {};

    if (!audioBase64) {
      return { success: false, error: 'audioBase64 is required' };
    }

    journal.setVoiceStatus('processing');

    try {
      // ── Step 1: STT ──
      const sttResult = await stt.transcribeBase64({ audioBase64, format, languageHint });
      logger.info('[Pipeline] STT complete', { text: sttResult.text.substring(0, 80), language: sttResult.language });

      if (!sttResult.text.trim()) {
        journal.setVoiceStatus('idle');
        return { success: true, skipped: true, reason: 'empty_transcript' };
      }

      // ── Noise filter: reject low-confidence or sound-effect-only transcripts ──
      const confidence = sttResult.confidence || 0;
      const langCode = (sttResult.language || '').toLowerCase().substring(0, 3);
      const isSoundEffect = /^\s*(\(.*?\)\s*)+\s*$/.test(sttResult.text); // pure "(sound effects)"
      const isTooShort = sttResult.text.trim().replace(/[^a-z]/gi, '').length < 3;
      // Wake word mode: require English + confidence ≥ 0.5 (foreign = background noise misdetection)
      const isWakeWordNoise = !pushToTalk && !skipWakeWordCheck &&
        (confidence < 0.5 || (langCode !== 'eng' && langCode !== 'en'));
      // PTT mode: only reject very low confidence
      const isPttNoise = pushToTalk && confidence < 0.2;
      if (isSoundEffect || isTooShort || isWakeWordNoise || isPttNoise) {
        logger.info('[Pipeline] Skipped (noise/low-confidence)', {
          text: sttResult.text.substring(0, 60),
          confidence,
          language: langCode,
          reason: isSoundEffect ? 'sound_effect' : isTooShort ? 'too_short' : isWakeWordNoise ? 'wake_word_noise' : 'ptt_low_confidence',
        });
        journal.setVoiceStatus('idle');
        return { success: true, skipped: true, reason: 'noise_filtered', transcript: sttResult.text };
      }

      // ── Step 2: Wake word check (skip if push-to-talk or explicitly bypassed) ──
      const detectedLanguage = normalizeLanguage(sttResult.language);
      journal.setVoiceStatus('processing', { detectedLanguage });

      if (!pushToTalk && !skipWakeWordCheck) {
        const wakeResult = wakeWord.detect(sttResult.text);
        if (!wakeResult.detected) {
          journal.setVoiceStatus('idle');
          return { success: true, skipped: true, reason: 'no_wake_word', transcript: sttResult.text };
        }

        if (wakeResult.type === 'cancel' || wakeResult.type === 'status') {
          sttResult.text = sttResult.text;
        } else {
          sttResult.text = wakeWord.stripWakePhrase(sttResult.text, wakeResult.matchedPhrase);
        }
      }

      // ── Step 3: Translate to English for StateGraph ──
      const { englishText, wasTranslated } = await toEnglish(sttResult);
      logger.info('[Pipeline] Translation', { wasTranslated, englishText: englishText.substring(0, 80) });

      // ── Step 4: Route ──
      const routeResult = router.route(englishText);

      // Write signal to journal if needed (cancel/pause/resume)
      if (routeResult.signalType) {
        const signalId = journal.writeSignal(routeResult.signalType, { message: englishText, sessionId });
        logger.info('[Pipeline] Signal written', { signalType: routeResult.signalType, signalId });
      }

      // ── Step 5: Generate response ──
      let responseEnglish;
      let responseMetadata = {};

      if (routeResult.lane === 'fast') {
        const fastResult = await this._fastLaneResponse(englishText, routeResult);
        responseEnglish = fastResult.text;
        responseMetadata = fastResult.metadata || {};
      } else {
        const sgResult = await this._stategraphLaneResponse(englishText, sessionId);
        responseEnglish = sgResult.text;
        responseMetadata = sgResult.metadata || {};
      }

      if (!responseEnglish) {
        journal.setVoiceStatus('idle');
        return { success: true, lane: routeResult.lane, transcript: sttResult.text, englishText, response: null };
      }

      // ── Step 6: Translate response back to user language ──
      const responseInUserLanguage = await fromEnglish(responseEnglish, detectedLanguage);

      // ── Step 7: TTS ──
      journal.setVoiceStatus('speaking');
      const ttsResult = await tts.synthesize({ text: responseInUserLanguage, language: detectedLanguage });
      journal.setVoiceStatus('idle', { lastSpokenAt: new Date().toISOString() });

      return {
        success: true,
        lane: routeResult.lane,
        transcript: sttResult.text,
        detectedLanguage,
        wasTranslated,
        englishText,
        responseEnglish,
        responseFinal: responseInUserLanguage,
        audioBase64: ttsResult.audioBuffer.toString('base64'),
        audioFormat: ttsResult.format,
        durationEstimateMs: ttsResult.durationEstimateMs,
        metadata: responseMetadata,
      };
    } catch (error) {
      journal.setVoiceStatus('idle');
      logger.error('[Pipeline] Error', { error: error.message });
      return { success: false, error: error.message };
    }
  }

  // ─── Fast Lane ────────────────────────────────────────────────────────────────

  async _fastLaneResponse(englishText, routeResult) {
    // Status queries — answer from journal without hitting LLM
    if (routeResult.reason === 'cancel_signal') {
      const sgState = journal.read().stategraph;
      if (sgState.status === 'running') {
        return { text: 'Cancelling the current task now.', metadata: { source: 'journal' } };
      }
      return { text: "There's nothing running to cancel.", metadata: { source: 'journal' } };
    }

    if (routeResult.reason === 'pause_signal') {
      return { text: 'Pausing the current task.', metadata: { source: 'journal' } };
    }

    if (routeResult.reason === 'resume_signal') {
      return { text: 'Resuming the task.', metadata: { source: 'journal' } };
    }

    if (routeResult.reason === 'fast_intent' || routeResult.reason === 'sg_running_question') {
      const statusSummary = journal.getStatusSummary();
      const isStatusQuery = /\b(status|how|progress|what|where|are you|still)\b/i.test(englishText);

      if (isStatusQuery) {
        return { text: statusSummary, metadata: { source: 'journal' } };
      }
    }

    // For other fast-lane queries, hit the LLM directly (no StateGraph)
    try {
      const answer = await this._directLLMQuery(englishText);
      return { text: answer, metadata: { source: 'fast_llm' } };
    } catch (error) {
      logger.error('[FastLane] LLM error', { error: error.message });
      return { text: "I'm sorry, I couldn't get an answer right now.", metadata: { source: 'error' } };
    }
  }

  // ─── StateGraph Lane ──────────────────────────────────────────────────────────

  async _stategraphLaneResponse(englishText, sessionId) {
    try {
      const result = await this._injectIntoStateGraph(englishText, sessionId);
      return { text: result.answer || result.text || '', metadata: { source: 'stategraph', intent: result.intent } };
    } catch (error) {
      logger.error('[StateGraphLane] Error', { error: error.message });
      return { text: "I encountered an error processing your request.", metadata: { source: 'error' } };
    }
  }

  // ─── Direct LLM (fast lane) ───────────────────────────────────────────────────

  _directLLMQuery(text) {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(STATEGRAPH_WS_URL, {
        headers: STATEGRAPH_API_KEY ? { Authorization: `Bearer ${STATEGRAPH_API_KEY}` } : {},
      });

      const requestId = `voice_fast_${Date.now()}`;
      let fullText = '';
      let settled = false;
      const timeout = setTimeout(() => {
        if (!settled) { settled = true; ws.close(); resolve(fullText || "I'm not sure about that."); }
      }, 15000);

      ws.on('open', () => {
        ws.send(JSON.stringify({
          id: requestId,
          type: 'llm_request',
          payload: {
            prompt: text,
            options: {
              maxTokens: 300,
              temperature: 0.3,
              taskType: 'conversation',
              responseLength: 'short',
            },
          },
          timestamp: Date.now(),
        }));
      });

      ws.on('message', (data) => {
        try {
          const msg = JSON.parse(data.toString());
          if (msg.type === 'llm_stream_chunk' && msg.payload?.text) {
            fullText += msg.payload.text;
          } else if (msg.type === 'llm_stream_end') {
            if (!settled) { settled = true; clearTimeout(timeout); ws.close(); resolve(fullText); }
          } else if (msg.type === 'error') {
            if (!settled) { settled = true; clearTimeout(timeout); ws.close(); reject(new Error(msg.payload?.message || 'LLM error')); }
          }
        } catch (_) {}
      });

      ws.on('error', (err) => { if (!settled) { settled = true; clearTimeout(timeout); reject(err); } });
      ws.on('close', () => { if (!settled) { settled = true; clearTimeout(timeout); resolve(fullText || ''); } });
    });
  }

  // ─── StateGraph Injection ─────────────────────────────────────────────────────

  async _injectIntoStateGraph(text, sessionId) {
    return new Promise((resolve, reject) => {
      const http_module = require('http');
      const body = JSON.stringify({ message: text, sessionId, source: 'voice' });

      const req = http_module.request({
        hostname: '127.0.0.1',
        port: THINKDROP_MAIN_PORT,
        path: '/voice/inject',
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) },
      }, (res) => {
        let data = '';
        res.on('data', chunk => { data += chunk; });
        res.on('end', () => {
          try {
            resolve(JSON.parse(data));
          } catch (_) {
            resolve({ answer: data });
          }
        });
      });

      req.on('error', reject);
      req.setTimeout(120000, () => req.destroy(new Error('StateGraph injection timeout')));
      req.write(body);
      req.end();
    });
  }

  // ─── Signal ───────────────────────────────────────────────────────────────────

  async writeSignal(args) {
    const { type, payload = {} } = args || {};
    if (!type) return { success: false, error: 'type is required (cancel|pause|resume|inject)' };

    const validTypes = ['cancel', 'pause', 'resume', 'inject'];
    if (!validTypes.includes(type)) {
      return { success: false, error: `Invalid signal type. Must be one of: ${validTypes.join(', ')}` };
    }

    const signalId = journal.writeSignal(type, payload);
    return { success: true, signalId, type };
  }

  // ─── Status ───────────────────────────────────────────────────────────────────

  async getStatus() {
    const state = journal.read();
    return {
      success: true,
      voice: state.voice,
      stategraph: state.stategraph,
      pendingSignals: (state.signals || []).filter(s => s.status === 'pending').length,
      summary: journal.getStatusSummary(),
    };
  }

  async setStatus(args) {
    const { status, extra = {} } = args || {};
    if (!status) return { success: false, error: 'status is required' };
    journal.setVoiceStatus(status, extra);
    return { success: true, status };
  }

  // ─── Health ───────────────────────────────────────────────────────────────────

  async healthCheck() {
    const elevenlabsAvailable = await stt.isAvailable().catch(() => false);
    return {
      success: true,
      service: this.serviceName,
      status: 'healthy',
      elevenlabs: elevenlabsAvailable,
      journalPath: journal.JOURNAL_PATH,
      skills: ['voice.transcribe', 'voice.speak', 'voice.process', 'voice.signal', 'voice.status'],
    };
  }

  // ─── HTTP Server ──────────────────────────────────────────────────────────────

  async start() {
    logger.info('Starting Voice Service MCP server');

    const server = http.createServer(async (req, res) => {
      res.setHeader('Content-Type', 'application/json');
      res.setHeader('Access-Control-Allow-Origin', '*');

      if (req.method === 'OPTIONS') {
        res.writeHead(204);
        res.end();
        return;
      }

      // Health
      if (req.url === '/health' || req.url === '/service.health') {
        res.writeHead(200);
        res.end(JSON.stringify(await this.healthCheck()));
        return;
      }

      // GET status
      if (req.method === 'GET' && req.url === '/voice.status') {
        res.writeHead(200);
        res.end(JSON.stringify(await this.getStatus()));
        return;
      }

      // POST routes
      if (req.method === 'POST') {
        let body = '';
        req.on('data', chunk => { body += chunk; });
        req.on('end', async () => {
          try {
            const parsed = body ? JSON.parse(body) : {};
            const payload = parsed.payload || parsed;

            let result;
            switch (req.url) {
              case '/voice.transcribe':
                result = await this.transcribe(payload);
                break;
              case '/voice.speak':
                result = await this.speak(payload);
                break;
              case '/voice.process':
                result = await this.process(payload);
                break;
              case '/voice.signal':
                result = await this.writeSignal(payload);
                break;
              case '/voice.status':
                result = await this.setStatus(payload);
                break;
              case '/voice.command':
                result = await this.executeVoiceCommand(payload);
                break;
              default:
                res.writeHead(404);
                res.end(JSON.stringify({ error: 'Not found', url: req.url }));
                return;
            }

            res.writeHead(200);
            res.end(JSON.stringify({ success: true, data: result }));
          } catch (err) {
            logger.error('Request error', { error: err.message, url: req.url });
            res.writeHead(200);
            res.end(JSON.stringify({ success: true, data: { success: false, error: err.message } }));
          }
        });
        return;
      }

      res.writeHead(404);
      res.end(JSON.stringify({ error: 'Not found' }));
    });

    server.listen(PORT, () => {
      logger.info(`Voice Service listening on http://localhost:${PORT}`);
    });

    // ── Graceful shutdown ──
    const shutdown = (signal) => {
      logger.info(`${signal} received — shutting down voice service`);
      journal.setVoiceStatus('idle');
      server.close(() => process.exit(0));
    };

    process.on('SIGINT',  () => shutdown('SIGINT'));
    process.on('SIGTERM', () => shutdown('SIGTERM'));

    logger.info('Voice Service ready', {
      port: PORT,
      journalPath: journal.JOURNAL_PATH,
      wakeWordEnabled: true,
    });
  }

  /**
   * General voice command endpoint — handles skill routing via payload.skill
   */
  async executeVoiceCommand(payload) {
    const { skill, args = {} } = payload || {};

    switch (skill) {
      case 'voice.transcribe':  return await this.transcribe(args);
      case 'voice.speak':       return await this.speak(args);
      case 'voice.process':     return await this.process(args);
      case 'voice.signal':      return await this.writeSignal(args);
      case 'voice.status':      return await this.getStatus();
      default:
        return { success: false, error: `Unknown voice skill: ${skill}` };
    }
  }
}

if (require.main === module) {
  const server = new VoiceServiceMCPServer();
  server.start().catch((error) => {
    logger.error('Failed to start voice service', { error: error.message });
    process.exit(1);
  });
}

module.exports = VoiceServiceMCPServer;
