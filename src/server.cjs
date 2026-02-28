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

const fs = require('fs');
const path = require('path');
const http = require('http');
const crypto = require('crypto');
const WebSocket = require('ws');
const logger = require('./logger.cjs');
const stt = require('./stt.cjs');
const tts = require('./inworld-tts.cjs');
const voiceProvider = require('./voice-provider.cjs');
const { toEnglish, fromEnglish, normalizeLanguage, LANGUAGE_NAMES } = require('./language-translate.cjs');
const wakeWord = require('./wake-word.cjs');
const router = require('./voice-router.cjs');
const { warmUp: classifierWarmUp, classifyLLMResponse } = require('./voice-classifier.cjs');
const journal = require('./voice-journal.cjs');

const PORT = parseInt(process.env.PORT || '3006', 10);
const SERVICE_NAME = process.env.SERVICE_NAME || 'voice-service';

const THINKDROP_MAIN_PORT = parseInt(process.env.THINKDROP_MAIN_PORT || '3010', 10);
const STATEGRAPH_WS_URL = process.env.STATEGRAPH_WS_URL || 'ws://localhost:4000/ws/stream';
const STATEGRAPH_API_KEY = process.env.STATEGRAPH_API_KEY || '';

// ── Fast Lane Butler Persona ──────────────────────────────────────────────────
// Loaded from prompts/fast-lane-butler.md — edit there to change tone/wording.
function _loadFastLanePrompt() {
  try {
    return fs.readFileSync(path.join(__dirname, 'prompts/fast-lane-butler.md'), 'utf8').trim();
  } catch (_) {
    return 'You are ThinkDrop\'s voice assistant. Be concise, helpful, and use no markdown.';
  }
}
const FAST_LANE_SYSTEM_PROMPT = _loadFastLanePrompt();

class VoiceServiceMCPServer {
  constructor() {
    this.serviceName = SERVICE_NAME;
    journal.setVoiceStatus('idle');
    classifierWarmUp();
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
    // ── Hume EVI path — replaces entire STT→LLM→TTS pipeline ─────────────────
    // EVI handles speech-to-speech internally. We intercept action intents via
    // the trigger_stategraph tool call and forward them to StateGraph.
    if (voiceProvider.isHumeEVI() && (args?.audioBase64) && !args?.skipSTT) {
      return this._processWithHumeEVI(args);
    }

    const {
      audioBase64,
      format = 'wav',
      languageHint = null,
      skipWakeWordCheck = false,
      pushToTalk = false,
      sessionId,
      // Web Speech API path: transcript already resolved in renderer — skip STT
      transcript: preTranscript = null,
      skipSTT = false,
      // PTT text-only mode: run STT only, skip LLM/inject, return transcript
      skipInject = false,
    } = args || {};

    // ── Pre-transcribed path (Web Speech API / native STT) ───────────────────
    if (skipSTT && preTranscript) {
      journal.setVoiceStatus('processing');
      logger.info('[Pipeline] process() called (skipSTT — transcript pre-supplied)', {
        transcript: preTranscript.substring(0, 80), pushToTalk, skipWakeWordCheck,
      });
      // Inject directly into the pipeline at step 2 with a synthetic sttResult
      const sttResult = { text: preTranscript, language: languageHint || 'en', confidence: 1.0, isFinal: true };
      try {
        return await this._runPipeline({ sttResult, pushToTalk, skipWakeWordCheck, sessionId, skipInject });
      } catch (err) {
        logger.error('[Pipeline] Error in skipSTT path', { error: err.message });
        journal.setVoiceStatus('idle');
        return { success: false, error: err.message };
      }
    }

    if (!audioBase64) {
      return { success: false, error: 'audioBase64 is required' };
    }

    // Minimum audio size guard — accidental taps / silence come in at ~0.5-1KB decoded.
    // PTT presses can legitimately be short (1-2 words), so threshold is 1500 decoded bytes.
    const decodedBytes = Math.floor(audioBase64.length * 3 / 4);
    if (decodedBytes < 1500) {
      logger.info('[Pipeline] Skipped — audio too short (likely silence or accidental press)', { decodedBytes });
      return { success: true, skipped: true, reason: 'audio_too_short' };
    }

    journal.setVoiceStatus('processing');
    logger.info('[Pipeline] process() called', { audioBase64Len: audioBase64.length, decodedBytes, format, pushToTalk, skipWakeWordCheck });

    try {
      // ── Step 1: STT ──
      // Always auto-detect language — Whisper handles multilingual input correctly.
      // The noise filter below rejects truly unexpected non-English in wake word mode.
      // Forcing 'en' here would cause Chinese/Spanish/etc. speech to be mangled.
      // STT_LANGUAGE_HINT env var forces a specific language (e.g. 'zh') when the user
      // speaks a language that Groq Whisper tends to hallucinate as English phonetics.
      const effectiveLanguageHint = languageHint || process.env.STT_LANGUAGE_HINT || null;
      const sttResult = await voiceProvider.transcribe({ audioBase64, format, languageHint: effectiveLanguageHint });
      logger.info('[Pipeline] STT complete', { text: sttResult.text.substring(0, 80), language: sttResult.language });

      if (!sttResult.text.trim()) {
        journal.setVoiceStatus('idle');
        return { success: true, skipped: true, reason: 'empty_transcript' };
      }

      return await this._runPipeline({ sttResult, pushToTalk, skipWakeWordCheck, sessionId, skipInject });
    } catch (err) {
      logger.error('[Pipeline] Error', { error: err.message });
      journal.setVoiceStatus('idle');
      return { success: false, error: err.message };
    }
  }

  async _runPipeline({ sttResult, pushToTalk, skipWakeWordCheck, sessionId, skipInject = false }) {
    try {
      // ── Noise filter ──
      // Strip parenthetical sound-effect labels first, keep actual spoken words
      const cleanedText = sttResult.text.replace(/\(.*?\)/g, '').trim();
      const confidence = sttResult.confidence || 0;
      const langCode = (sttResult.language || '').toLowerCase().substring(0, 3);
      const isPureSoundEffect = cleanedText.length < 2; // nothing left after stripping
      const isTooShort = cleanedText.replace(/[^a-zA-Z0-9]/g, '').length < 2;

      if (pushToTalk) {
        // PTT: user intentionally pressed — only reject pure sound effects (e.g. "[music]").
        // Do NOT filter on confidence (Groq returns 0 by default) or short transcripts.
        if (isPureSoundEffect) {
          logger.info('[Pipeline] PTT skipped (pure sound effect)', { text: sttResult.text.substring(0, 60) });
          journal.setVoiceStatus('idle');
          return { success: true, skipped: true, reason: 'noise_filtered', transcript: sttResult.text };
        }
      } else if (!pushToTalk) {
        // Wake word mode: filter out low-confidence noise only.
        // Do NOT filter on language — user may intentionally speak Chinese, Spanish, etc.
        // Exception: if transcript contains a wake phrase, always allow through.
        const WAKE_PHRASE_QUICK = /\b(thinkdrop|think\s*drop|hey\s*think|yo\s*think|ok\s*think|listen\s*up|wake\s*up|i\s*need\s*you|are\s*you\s*there)\b/i;
        const containsWakePhrase = WAKE_PHRASE_QUICK.test(sttResult.text);
        const isLowConfidenceNoise = !containsWakePhrase && confidence > 0 && confidence < 0.5;
        if (isLowConfidenceNoise) {
          logger.info('[Pipeline] Wake word skipped (low-confidence noise)', {
            text: sttResult.text.substring(0, 60), confidence, language: langCode,
          });
          journal.setVoiceStatus('idle');
          return { success: true, skipped: true, reason: 'noise_filtered', transcript: sttResult.text };
        }
      }

      // ── Step 2: Wake word check (skip if push-to-talk or explicitly bypassed) ──
      const detectedLanguage = normalizeLanguage(sttResult.language);
      journal.setVoiceStatus('processing', { detectedLanguage });
      // Persist session language as single source of truth — answer.js reads this directly
      if (detectedLanguage && detectedLanguage !== 'en') {
        journal.setSessionLanguage(detectedLanguage);
      }

      if (!pushToTalk && !skipWakeWordCheck) {
        const wakeResult = wakeWord.detect(sttResult.text);

        // If wake phrase detected, strip it and proceed
        if (wakeResult.detected) {
          if (wakeResult.type !== 'cancel' && wakeResult.type !== 'status') {
            sttResult.text = wakeWord.stripWakePhrase(sttResult.text, wakeResult.matchedPhrase);
          }
        } else {
          // No wake phrase — allow through ONLY if transcript starts with a clear question/command word.
          // Word count alone is not sufficient — background chatter can be 5+ words.
          const QUESTION_PREFIX = /^(what|where|when|who|why|how|can|could|will|would|should|is|are|do|does|did|tell|show|find|search|open|help|set|make|get|check|remind|play|create|build|run|send|write|schedule|look up|pull up|turn|type|scroll|press|click|focus|paste|copy|undo|tab|enter|install|list|go to|navigate|switch|close|minimize|maximize|use|using|i want|i need|i would|i('m| am)|i notice|now |let'?s|try|analyze|analyse|examine|delete|remove|save|show me|take|move|rename|read|launch|start|stop|enable|disable|download|upload|please|just|could you|can you|will you|would you|you('re| are| can| should| need| must| never| didn'?t| don'?t| haven'?t| weren'?t| aren'?t)|you just|you only|you still|that'?s|that is|no you|hey you|actually|wait|hold on|never mind|forget that|also|and then|then|after that|next|but|however|instead|again|redo|retry|fix|correct|update)\b/i;
          const isRealQuestion = QUESTION_PREFIX.test(cleanedText);

          if (!isRealQuestion) {
            logger.info('[Pipeline] Wake word skipped — no wake phrase and no clear question/command prefix', { text: sttResult.text.substring(0, 60) });
            journal.setVoiceStatus('idle');
            return { success: true, skipped: true, reason: 'no_wake_word', transcript: sttResult.text };
          }
          const wordCount = cleanedText.split(/\s+/).filter(Boolean).length;
          logger.info('[Pipeline] Wake word bypassed — clear question/command prefix detected', { text: sttResult.text.substring(0, 60), wordCount });
        }
      }

      // ── Step 3: Translate to English for StateGraph ──
      const { englishText, wasTranslated } = await toEnglish(sttResult);
      logger.info('[Pipeline] Translation', { wasTranslated, englishText: englishText.substring(0, 80) });

      // ── Step 4: Route ──
      const routeResult = await router.route(englishText);

      // Write signal to journal if needed (cancel/pause/resume)
      if (routeResult.signalType) {
        const signalId = journal.writeSignal(routeResult.signalType, { message: englishText, sessionId });
        logger.info('[Pipeline] Signal written', { signalType: routeResult.signalType, signalId });
      }

      // ── Step 5: Generate response ──
      let responseEnglish;
      let responseMetadata = {};

      // PTT text-only: skip all LLM/TTS, return transcript immediately after routing
      if (skipInject) {
        logger.info('[Pipeline] skipInject=true — returning transcript only', { transcript: sttResult.text.substring(0, 60) });
        journal.setVoiceStatus('idle');
        return {
          success: true,
          lane: routeResult.lane,
          transcript: sttResult.text,
          detectedLanguage,
          wasTranslated,
          englishText,
          skipped: false,
        };
      }

      let fullAnswerEnglish = '';
      if (routeResult.lane === 'fast') {
        const fastResult = await this._fastLaneResponse(englishText, routeResult, detectedLanguage);
        responseEnglish = fastResult.text;
        fullAnswerEnglish = fastResult.text;
        responseMetadata = fastResult.metadata || {};
      } else {
        const sgResult = await this._stategraphLaneResponse(englishText, sessionId, detectedLanguage);
        responseEnglish = sgResult.text;        // spoken summary (short)
        fullAnswerEnglish = sgResult.fullAnswer || sgResult.text; // full answer for Results window
        responseMetadata = { ...sgResult.metadata || {}, hadLiveStream: !!sgResult.hadLiveStream };
      }

      if (!responseEnglish) {
        journal.setVoiceStatus('idle');
        return { success: true, lane: routeResult.lane, transcript: sttResult.text, englishText, response: null };
      }

      // ── Step 6: TTS — both lanes synthesize audio ────────────────────────────
      // Hard cap at 50 words before translation to keep TTS audio short (< 10s)
      const MAX_SPOKEN_WORDS = 50;
      const spokenWords = responseEnglish.trim().split(/\s+/);
      const cappedEnglish = spokenWords.length > MAX_SPOKEN_WORDS
        ? spokenWords.slice(0, MAX_SPOKEN_WORDS).join(' ') + '.'
        : responseEnglish;
      if (spokenWords.length > MAX_SPOKEN_WORDS) {
        logger.info('[Pipeline] TTS text capped', { original: spokenWords.length, capped: MAX_SPOKEN_WORDS });
      }
      const responseInUserLanguage = await fromEnglish(cappedEnglish, detectedLanguage);

      // Translate fullAnswer + responseEnglish for ResultsWindow display when non-English.
      // Use sessionLanguage from journal as fallback when wasTranslated is false
      // (e.g. Groq returned language='zh' directly so no translation was needed, but
      // the fast lane LLM reply was generated in English and must be translated back).
      const displayLang = (detectedLanguage && detectedLanguage !== 'en')
        ? detectedLanguage
        : (journal.read().voice?.sessionLanguage || 'en');
      const needsDisplayTranslation = displayLang !== 'en';

      let fullAnswerDisplay = fullAnswerEnglish;
      let responseDisplay = responseEnglish;
      if (needsDisplayTranslation) {
        try {
          // Translate both independently — fast lane has responseEnglish === fullAnswerEnglish,
          // stategraph lane may have different strings. Always translate responseDisplay.
          const translateFull = fullAnswerEnglish
            ? fromEnglish(fullAnswerEnglish, displayLang).catch(() => fullAnswerEnglish)
            : Promise.resolve(fullAnswerEnglish);
          const translateResp = responseEnglish
            ? fromEnglish(responseEnglish, displayLang).catch(() => responseEnglish)
            : Promise.resolve(responseEnglish);
          [fullAnswerDisplay, responseDisplay] = await Promise.all([translateFull, translateResp]);
          logger.info('[Pipeline] Display text translated', { displayLang, lane: routeResult.lane });
        } catch (_) {}
      }

      journal.setVoiceStatus('speaking');
      const ttsResult = await voiceProvider.synthesize({ text: responseInUserLanguage, language: detectedLanguage });
      journal.setVoiceStatus('idle', { lastSpokenAt: new Date().toISOString() });

      return {
        success: true,
        lane: routeResult.lane,
        transcript: sttResult.text,
        detectedLanguage,
        wasTranslated,
        englishText,
        responseEnglish: responseDisplay,
        fullAnswer: fullAnswerDisplay,
        _hadLiveStream: !!(responseMetadata.hadLiveStream),
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

  async _fastLaneResponse(englishText, routeResult, responseLanguage = null) {
    // Status queries — answer from journal without hitting LLM
    if (routeResult.reason === 'cancel_signal') {
      const sgState = journal.read().stategraph;
      if (sgState.status === 'running') {
        return { text: 'Right away — cancelling the current task.', metadata: { source: 'journal' } };
      }
      return { text: "Nothing is running at the moment, sir. The slate is clean.", metadata: { source: 'journal' } };
    }

    if (routeResult.reason === 'pause_signal') {
      return { text: 'Holding position — task paused. Say resume when ready.', metadata: { source: 'journal' } };
    }

    if (routeResult.reason === 'resume_signal') {
      return { text: 'Back in motion. Resuming where we left off.', metadata: { source: 'journal' } };
    }

    if (routeResult.reason === 'fast_intent' || routeResult.reason === 'sg_running_question') {
      const statusSummary = journal.getStatusSummary();
      const isStatusQuery = /\b(status|how|progress|what|where|are you|still)\b/i.test(englishText);

      if (isStatusQuery) {
        return { text: statusSummary, metadata: { source: 'journal' } };
      }
    }

    // For other fast-lane queries, hit the LLM directly (no StateGraph)
    // Inject sessionLanguage from journal so the LLM replies in the user's language directly.
    const _fastLang = journal.read().voice?.sessionLanguage || responseLanguage || 'en';
    const _fastSystemPrompt = (_fastLang && _fastLang !== 'en')
      ? `${FAST_LANE_SYSTEM_PROMPT}\n\nIMPORTANT: The user is speaking ${LANGUAGE_NAMES[_fastLang] || _fastLang}. You MUST respond entirely in ${LANGUAGE_NAMES[_fastLang] || _fastLang}. Do not use English.`
      : FAST_LANE_SYSTEM_PROMPT;
    try {
      const answer = await this._directLLMQuery(englishText, _fastSystemPrompt);
      if (!answer || !answer.trim()) {
        return { text: "Consider it noted, though I've nothing to add at the moment.", metadata: { source: 'fast_llm_empty' } };
      }

      // ── Action escalation: classify LLM response ──────────────────────────
      // If the LLM replied with action-confirmation language ("I'll do that",
      // "Consider it done", "Closing now", etc.) the fast lane has no ability
      // to actually execute — escalate the original user prompt to stategraph
      // in the background so the real action runs while TTS plays.
      // The stategraph result (including its own TTS) is sent to the renderer
      // via the /voice/result overlay endpoint after execution completes.
      classifyLLMResponse(answer).then(async ({ isActionConfirmation }) => {
        if (!isActionConfirmation) return;
        logger.info('[FastLane] Action escalation — routing original prompt to stategraph', {
          prompt: englishText.substring(0, 80),
          response: answer.substring(0, 80),
        });
        try {
          const escalationLang = journal.read().voice?.sessionLanguage || 'en';
          const sgResult = await this._stategraphLaneResponse(englishText, null, escalationLang !== 'en' ? escalationLang : null);
          if (!sgResult.text) return;
          // Translate spoken summary back to user's language before TTS
          const ttsLang = (escalationLang && escalationLang !== 'en') ? escalationLang : 'en';
          const spokenText = (ttsLang !== 'en') ? await fromEnglish(sgResult.text, ttsLang).catch(() => sgResult.text) : sgResult.text;
          // Synthesize TTS for the stategraph spoken summary
          const ttsResult = await voiceProvider.synthesize({ text: spokenText, language: ttsLang }).catch(() => null);
          if (!ttsResult) return;
          // POST result back to main process so it forwards voice:response to renderer
          const http_module = require('http');
          const payload = JSON.stringify({
            text: sgResult.text,
            fullAnswer: sgResult.fullAnswer || sgResult.text,
            audioBase64: ttsResult.audioBuffer.toString('base64'),
            audioFormat: ttsResult.format || 'mp3',
            language: ttsLang,
            lane: 'stategraph',
            durationEstimateMs: ttsResult.durationEstimateMs || null,
          });
          const postReq = http_module.request({
            hostname: '127.0.0.1',
            port: THINKDROP_MAIN_PORT,
            path: '/voice/result',
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) },
          }, () => {});
          postReq.on('error', () => {});
          postReq.write(payload);
          postReq.end();
          logger.info('[FastLane] Escalation result sent to renderer', { text: sgResult.text.substring(0, 60) });
        } catch (err) {
          logger.error('[FastLane] Escalation stategraph error', { error: err.message });
        }
      }).catch(() => {});

      return { text: answer, metadata: { source: 'fast_llm' } };
    } catch (error) {
      logger.error('[FastLane] LLM error', { error: error.message });
      return { text: "My apologies — the circuits are occupied. Try again in a moment.", metadata: { source: 'error' } };
    }
  }

  // ─── StateGraph Lane ──────────────────────────────────────────────────────────

  async _stategraphLaneResponse(englishText, sessionId, responseLanguage = null) {
    try {
      // ── Guard: don't queue behind a running stategraph task ─────────────────
      // If the journal shows stategraph is actively running (e.g. guide.step is
      // waiting for user input, or waitForAuth is polling), injecting now will
      // queue behind it and timeout after 120s. Return a busy message instead.
      const sgState = journal.read().stategraph;
      const sgLastUpdate = sgState?.lastUpdate ? new Date(sgState.lastUpdate).getTime() : 0;
      const sgIsStale = (sgState?.status === 'running') && (Date.now() - sgLastUpdate > 300_000); // 5 min
      if (sgState?.status === 'running' && !sgIsStale) {
        logger.info('[StateGraphLane] Stategraph already running — returning busy response', { status: sgState.status });
        return { text: "I'm still working on the previous task. Please wait or say cancel to stop it.", metadata: { source: 'busy' } };
      }
      const result = await this._injectIntoStateGraph(englishText, sessionId, responseLanguage);
      const fullAnswer = result.answer || result.text || '';
      const intent = result.intent || 'unknown';

      // For command_automate lanes: answer is empty (streamed to Results window during execution).
      // Generate a short spoken confirmation instead.
      if (!fullAnswer || fullAnswer.trim().length === 0) {
        const confirmation = intent === 'command_automate'
          ? 'Done.'
          : "Got it.";
        logger.info('[StateGraphLane] Command lane — using confirmation', { intent, confirmation });
        return { text: confirmation, fullAnswer: '', hadLiveStream: !!result.hadLiveStream, metadata: { source: 'stategraph', intent } };
      }

      // Generate a short spoken summary (≤20 words) — full answer already streamed to Results window.
      const SUMMARY_WORD_LIMIT = 20;
      let spokenSummary = fullAnswer;
      if (fullAnswer.split(/\s+/).length > SUMMARY_WORD_LIMIT) {
        try {
          const raw = await this._directLLMQuery(
            `Reply in ONE sentence of at most 15 words for a voice assistant. No markdown. Just the sentence.\n\nContext: ${fullAnswer.substring(0, 800)}`
          );
          // Hard-cap regardless of what LLM returns
          const words = raw.trim().split(/\s+/);
          spokenSummary = words.length > SUMMARY_WORD_LIMIT
            ? words.slice(0, SUMMARY_WORD_LIMIT).join(' ') + '.'
            : raw.trim();
          logger.info('[StateGraphLane] Generated spoken summary', { fullWords: fullAnswer.split(/\s+/).length, summaryWords: spokenSummary.split(/\s+/).length });
        } catch (_) {
          spokenSummary = fullAnswer.split(/\s+/).slice(0, SUMMARY_WORD_LIMIT).join(' ') + '.';
        }
      }

      return {
        text: spokenSummary,
        fullAnswer,
        hadLiveStream: !!result.hadLiveStream,
        metadata: { source: 'stategraph', intent },
      };
    } catch (error) {
      logger.error('[StateGraphLane] Error', { error: error.message });
      return { text: "I encountered an error processing your request.", metadata: { source: 'error' } };
    }
  }

  // ─── Direct LLM (fast lane) ───────────────────────────────────────────────────

  _directLLMQuery(text, systemPrompt = null) {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(STATEGRAPH_WS_URL, {
        headers: STATEGRAPH_API_KEY ? { Authorization: `Bearer ${STATEGRAPH_API_KEY}` } : {},
      });

      const requestId = `voice_fast_${Date.now()}`;
      let fullText = '';
      let settled = false;
      const timeout = setTimeout(() => {
        if (!settled) { settled = true; ws.close(); resolve(fullText || 'Forgive the delay — no answer came in time.'); }
      }, 15000);

      ws.on('open', () => {
        ws.send(JSON.stringify({
          id: requestId,
          type: 'llm_request',
          payload: {
            prompt: text,
            provider: 'openai',
            options: {
              maxTokens: 150,
              temperature: 0.7,
              taskType: 'conversation',
              responseLength: 'short',
              stream: true,
            },
            context: {
              systemInstructions: systemPrompt || '',
            },
          },
          timestamp: Date.now(),
          metadata: { source: 'voice_fast_lane' },
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

  async _injectIntoStateGraph(text, sessionId, responseLanguage = null) {
    return new Promise((resolve, reject) => {
      const http_module = require('http');
      const body = JSON.stringify({ message: text, sessionId, source: 'voice', responseLanguage });

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
    const sttAvailable = await stt.isAvailable().catch(() => false);
    return {
      success: true,
      service: this.serviceName,
      status: 'healthy',
      stt: sttAvailable,
      tts: !!(process.env.CARTESIA_API_KEY),
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
              case '/voice.provider':
                result = await this.handleProviderSwitch(payload);
                break;
              default:
                res.writeHead(404);
                res.end(JSON.stringify({ error: 'Not found', url: req.url }));
                return;
            }

            const responseBody = JSON.stringify({ success: true, data: result });
            if (req.url === '/voice.process') {
              logger.info('[voice.process] Sending response', {
                hasAudio: !!(result && result.audioBase64),
                audioBytes: result?.audioBase64?.length || 0,
                lane: result?.lane,
                skipped: result?.skipped,
              });
            }
            res.writeHead(200);
            res.end(responseBody);
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
  // ─── Provider switching ─────────────────────────────────────────────────────

  async handleProviderSwitch(payload) {
    const { action, provider } = payload || {};
    if (action === 'list') {
      return { success: true, providers: voiceProvider.listProviders(), active: voiceProvider.getProviderName() };
    }
    if (action === 'set' && provider) {
      try {
        const result = voiceProvider.setProvider(provider);
        return { success: true, ...result };
      } catch (err) {
        return { success: false, error: err.message };
      }
    }
    return { success: true, active: voiceProvider.getProviderName(), providers: voiceProvider.listProviders() };
  }

  // ─── Hume EVI one-shot pipeline ───────────────────────────────────────────────

  async _processWithHumeEVI({ audioBase64, sessionId }) {
    journal.setVoiceStatus('processing');
    logger.info('[Pipeline:EVI] Processing audio via Hume EVI');

    try {
      const result = await voiceProvider.eviProcessAudio({
        audioBase64,
        onStateGraphTrigger: async (text) => {
          logger.info('[Pipeline:EVI] StateGraph trigger', { text: text.substring(0, 80) });
          try {
            const sgResult = await this._stategraphLaneResponse(text, sessionId);
            if (!sgResult.text) return;
            // Synthesize TTS for the stategraph result (EVI already spoke holding phrase)
            // For EVI mode we use the standard TTS for the result notification,
            // since EVI's session is already closed by then.
            const ttsResult = await tts.synthesize({ text: sgResult.text, language: 'en' }).catch(() => null);
            if (!ttsResult) return;
            const http_module = require('http');
            const payload = JSON.stringify({
              text: sgResult.text,
              fullAnswer: sgResult.fullAnswer || sgResult.text,
              audioBase64: ttsResult.audioBuffer.toString('base64'),
              audioFormat: ttsResult.format || 'mp3',
              language: 'en',
              lane: 'stategraph',
              durationEstimateMs: ttsResult.durationEstimateMs || null,
            });
            const postReq = http_module.request({
              hostname: '127.0.0.1', port: THINKDROP_MAIN_PORT,
              path: '/voice/result', method: 'POST',
              headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) },
            }, () => {});
            postReq.on('error', () => {});
            postReq.write(payload);
            postReq.end();
          } catch (err) {
            logger.error('[Pipeline:EVI] StateGraph error', { error: err.message });
          }
        },
      });

      journal.setVoiceStatus(result.lane === 'stategraph' ? 'idle' : 'speaking');

      if (!result.audioBase64) {
        journal.setVoiceStatus('idle');
        return { success: true, lane: result.lane || 'fast', transcript: result.transcript, skipped: false, provider: 'hume' };
      }

      journal.setVoiceStatus('idle', { lastSpokenAt: new Date().toISOString() });
      return {
        success: true,
        lane: result.lane || 'fast',
        transcript: result.transcript || '',
        detectedLanguage: 'en',
        wasTranslated: false,
        englishText: result.transcript || '',
        responseEnglish: result.responseText || '',
        fullAnswer: result.responseText || '',
        responseFinal: result.responseText || '',
        audioBase64: result.audioBase64,
        audioFormat: result.audioFormat || 'mp3',
        durationEstimateMs: result.durationEstimateMs || 0,
        provider: 'hume',
      };
    } catch (err) {
      logger.error('[Pipeline:EVI] Error — falling back to standard pipeline', { error: err.message });
      journal.setVoiceStatus('idle');
      // Graceful degradation: fall through to standard pipeline
      return this.process({ audioBase64, sessionId, skipWakeWordCheck: true, pushToTalk: true });
    }
  }

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
