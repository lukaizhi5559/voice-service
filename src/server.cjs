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
const voiceProvider = require('./voice-provider.cjs');
const { toEnglish, fromEnglish, normalizeLanguage, LANGUAGE_NAMES } = require('./language-translate.cjs');
const wakeWord = require('./wake-word.cjs');
const kwsDetector = require('./kws-detector.cjs');
const router = require('./voice-router.cjs');
const { warmUp: classifierWarmUp, classifyFreshTask: _classifyFreshTask } = require('./voice-classifier.cjs');
const { detectEmotion } = require('./audio-emotion.cjs');
const journal = require('./voice-journal.cjs');

const PORT = parseInt(process.env.PORT || '3006', 10);
const SERVICE_NAME = process.env.SERVICE_NAME || 'voice-service';

const THINKDROP_MAIN_PORT = parseInt(process.env.THINKDROP_MAIN_PORT || '3010', 10);
const STATEGRAPH_WS_URL = process.env.STATEGRAPH_WS_URL || 'ws://localhost:4000/ws/stream';
const STATEGRAPH_API_KEY = process.env.STATEGRAPH_API_KEY || '';

// ── Fast Lane Butler Persona ──────────────────────────────────────────────────
// Loaded from prompts/fast-lane-butler.md — static base, never changes at runtime.
// The live personality overlay (mood + traits) is fetched per-request from personality-service.
function _loadFastLanePrompt() {
  try {
    return fs.readFileSync(path.join(__dirname, 'prompts/fast-lane-butler.md'), 'utf8').trim();
  } catch (_) {
    return 'You are ThinkDrop\'s voice assistant. Be concise, helpful, and use no markdown.';
  }
}
const FAST_LANE_BASE_PROMPT = _loadFastLanePrompt();

// ── Personality overlay fetch ──────────────────────────────────────────────────
// Fetches the live THINKDROP LIVE STATE block from personality-service (port 3008).
// Falls back to empty string if personality-service is down — zero breakage.
const PERSONALITY_SERVICE_PORT = parseInt(process.env.PERSONALITY_SERVICE_PORT || '3012', 10);

function _fetchPersonalityOverlay() {
  return new Promise((resolve) => {
    const body = JSON.stringify({ version: 'mcp.v1', service: 'personality-service', action: 'personality.overlay', payload: {}, requestId: 'vs_' + Date.now() });
    const req = http.request({
      hostname: '127.0.0.1', port: PERSONALITY_SERVICE_PORT,
      path: '/personality.overlay', method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) },
      timeout: 500,
    }, (res) => {
      let raw = '';
      res.on('data', c => { raw += c; });
      res.on('end', () => {
        try {
          const parsed = JSON.parse(raw);
          resolve(parsed && parsed.data && parsed.data.overlay ? parsed.data.overlay : '');
        } catch (_) { resolve(''); }
      });
    });
    req.on('error', () => resolve(''));
    req.on('timeout', () => { req.destroy(); resolve(''); });
    req.write(body);
    req.end();
  });
}

// ── Mood context fetch (lightweight — for behavioral modifiers only) ───────────
function _fetchMoodContext() {
  return new Promise((resolve) => {
    const body = JSON.stringify({ version: 'mcp.v1', service: 'personality-service', action: 'personality.moodContext', payload: {}, requestId: 'vmc_' + Date.now() });
    const req = http.request({
      hostname: '127.0.0.1', port: PERSONALITY_SERVICE_PORT,
      path: '/personality.moodContext', method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) },
      timeout: 1500,
    }, (res) => {
      let raw = '';
      res.on('data', c => { raw += c; });
      res.on('end', () => {
        try {
          const parsed = JSON.parse(raw);
          resolve(parsed && parsed.data ? parsed.data : { mood_label: 'content', behavioral_guidance: '' });
        } catch (_) { resolve({ mood_label: 'content', behavioral_guidance: '' }); }
      });
    });
    req.on('error', () => resolve({ mood_label: 'content', behavioral_guidance: '' }));
    req.on('timeout', () => { req.destroy(); resolve({ mood_label: 'content', behavioral_guidance: '' }); });
    req.write(body);
    req.end();
  });
}

// ── Mood event emission — fire-and-forget ──────────────────────────────────────
function _fireMoodEvent(eventType, source, reason) {
  try {
    const body = JSON.stringify({
      version: 'mcp.v1', service: 'personality-service',
      action: 'personality.event',
      payload: { event_type: eventType, source: source || 'voice', reason: reason || '' },
      requestId: 'vme_' + Date.now(),
    });
    const req = http.request({
      hostname: '127.0.0.1', port: PERSONALITY_SERVICE_PORT,
      path: '/personality.event', method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) },
      timeout: 3000,
    }, () => {});
    req.on('error', () => {});
    req.on('timeout', () => { req.destroy(); });
    req.write(body);
    req.end();
  } catch (_) {}
}

/**
 * Strip Resemble AI / ElevenLabs emotion and paralinguistic tags from text.
 * Used when the active TTS provider is NOT Resemble (tags would be read aloud literally).
 * Also used for display text — tags are TTS-only markup.
 *
 * Strips: [happy] [sad] [angry] [excited] [empathetic] [laugh] [sigh] [gasp] [crying]
 * Also strips basic SSML: <break .../> <prosody ...>...</prosody> <emotion .../>
 *
 * @param {string} text
 * @returns {string}
 */
function _stripEmotionTags(text) {
  if (!text) return text;
  return text
    .replace(/\[(happy|sad|angry|excited|empathetic|laugh|laughter|sigh|gasp|crying|whispering|shouting)\]/gi, '')
    .replace(/<break[^>]*\/>/gi, '')
    .replace(/<prosody[^>]*>(.*?)<\/prosody>/gi, '$1')
    .replace(/<emotion[^>]*\/>/gi, '')
    .replace(/\s{2,}/g, ' ')
    .trim();
}

/**
 * Return a fast canned holding phrase for the given language and trigger type.
 * Used when pre-LLM intent detection fires — no LLM round-trip needed.
 *
 * @param {string} lang - BCP-47 language code (e.g. 'en', 'zh', 'es')
 * @param {string} triggerType - 'action_lookup' | 'research' | 'computer_action'
 * @returns {string}
 */
function _holdingPhrase(lang, triggerType) {
  const phrases = {
    zh: {
      action_lookup: '让我查一下，马上回来。',
      research:      '让我查一下，马上回来。',
      computer_action: '好的，我来处理这个。',
    },
    es: {
      action_lookup: 'Déjame buscar eso ahora mismo.',
      research:      'Déjame investigar eso.',
      computer_action: 'ThinkDrop se encargará de eso.',
    },
    fr: {
      action_lookup: 'Je vérifie ça tout de suite.',
      research:      'Je cherche ça maintenant.',
      computer_action: 'ThinkDrop va s\'en occuper.',
    },
    ja: {
      action_lookup: 'ただいま確認します。',
      research:      'ただいま調べます。',
      computer_action: 'ThinkDropが処理します。',
    },
    ko: {
      action_lookup: '지금 바로 확인해 드리겠습니다.',
      research:      '찾아보겠습니다.',
      computer_action: 'ThinkDrop이 처리하겠습니다.',
    },
  };

  const langPhrases = phrases[lang] || {
    action_lookup:  'Let me check on that for you.',
    research:       'Let me look that up.',
    computer_action: 'Routing that to ThinkDrop now.',
  };

  return langPhrases[triggerType] || langPhrases.action_lookup;
}

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
      const result = await voiceProvider.synthesize({ text, language, voiceId, stability, similarity, style });
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
        return await this._runPipeline({ sttResult, audioBase64: null, pushToTalk, skipWakeWordCheck, sessionId, skipInject });
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

    // ── KWS gate: audio-domain keyword spotting (sherpa-onnx) ─────────────────
    // Run BEFORE Groq STT in wake-word mode. If sherpa-onnx can't find the wake
    // word in the raw audio, skip Groq entirely — no transcription cost, and we
    // avoid Whisper hallucinating "Armis" as "harvest" / "I promise" / "permiss".
    //
    // Three outcomes:
    //   1. KWS available + keyword NOT detected → bail out immediately
    //   2. KWS available + keyword detected     → set skipWakeWordCheck=true
    //      (audio already confirmed wake word; skip redundant text-based check)
    //   3. KWS unavailable (models not downloaded yet) → fall through silently
    //      to existing text-based wake-word detection in wake-word.cjs
    let _skipWakeWordCheck = skipWakeWordCheck;
    if (!pushToTalk && !skipWakeWordCheck) {
      const kwsResult = await kwsDetector.detectKeyword(audioBase64, format);
      if (kwsResult.available) {
        if (!kwsResult.detected && !kwsResult.decodeError) {
          // Audio-domain: no wake word present — skip Groq API call entirely
          logger.info('[Pipeline] KWS: no wake word in audio — skipping STT', {
            durationMs: kwsResult.durationMs,
          });
          return { success: true, skipped: true, reason: 'no_wake_word_audio' };
        } else if (kwsResult.detected) {
          // Audio confirmed wake word — bypass redundant text-based check downstream
          logger.info('[Pipeline] KWS: wake word confirmed in audio', {
            keyword: kwsResult.keyword,
            durationMs: kwsResult.durationMs,
          });
          _skipWakeWordCheck = true;
        }
        // kwsResult.decodeError → fall through to text-based detection
      }
    }

    journal.setVoiceStatus('processing');
    logger.info('[Pipeline] process() called', { audioBase64Len: audioBase64.length, decodedBytes, format, pushToTalk, skipWakeWordCheck: _skipWakeWordCheck });

    try {
      // ── Step 1: STT ──
      // Language hint strategy — prevent both hallucination AND language locking:
      //   1. Explicit hint from caller or STT_LANGUAGE_HINT env always wins.
      //   2. journalSessionLang is only used as a hint for NON-LATIN-SCRIPT languages
      //      (zh, ja, ko, ar, ru, hi) — Groq Whisper hallucinates these as English phonetics
      //      without a hint (e.g. "我想上Google" → "I want to go to Google" in English).
      //   3. For Latin-script languages (en, es, fr, de, etc.) we NEVER pass a hint from
      //      journal — Groq auto-detects Latin scripts reliably and passing 'zh' here would
      //      lock English speech into the Chinese decoder (the language-locking bug).
      //   4. If the user switches from Chinese to English, the next turn produces language='en'
      //      from Groq auto-detect, setSessionLanguage('en') is called, and the lock releases.
      const NON_LATIN_LANGS = new Set(['zh', 'ja', 'ko', 'ar', 'ru', 'hi', 'th', 'vi', 'uk']);
      const journalVoice = journal.read().voice || {};
      const journalSessionLang = journalVoice.sessionLanguage;
      // Expire language lock after 60s of inactivity
      const lastActivity = journalVoice.lastLanguageActivityAt ? new Date(journalVoice.lastLanguageActivityAt).getTime() : 0;
      const langIsExpired = lastActivity > 0 && (Date.now() - lastActivity) > 60 * 1000;
      // Wake-word mode: NEVER pass language hint — always auto-detect.
      // With a 'zh' hint active, Whisper encodes 'Armis'/'ThinkDrop' as Chinese characters
      // and the wake-word regex can't find the Latin-script token.
      // PTT mode: apply hint so Chinese/Japanese/etc. mid-conversation commands
      // aren't hallucinated as English phonetics.
      const isWakeWordMode = !pushToTalk && !_skipWakeWordCheck;
      const journalHint = (!isWakeWordMode && journalSessionLang && NON_LATIN_LANGS.has(journalSessionLang) && !langIsExpired) ? journalSessionLang : null;
      const effectiveLanguageHint = languageHint || process.env.STT_LANGUAGE_HINT || journalHint || null;
      logger.info('[Pipeline] STT language hint', { effectiveLanguageHint, journalSessionLang, callerHint: languageHint || null });
      const sttResult = await voiceProvider.transcribe({ audioBase64, format, languageHint: effectiveLanguageHint });
      logger.info('[Pipeline] STT complete', { text: sttResult.text.substring(0, 80), language: sttResult.language });

      if (!sttResult.text.trim()) {
        journal.setVoiceStatus('idle');
        return { success: true, skipped: true, reason: 'empty_transcript' };
      }

      return await this._runPipeline({ sttResult, audioBase64, pushToTalk, skipWakeWordCheck: _skipWakeWordCheck, sessionId, skipInject });
    } catch (err) {
      logger.error('[Pipeline] Error', { error: err.message });
      journal.setVoiceStatus('idle');
      return { success: false, error: err.message };
    }
  }

  async _runPipeline({ sttResult, audioBase64 = null, pushToTalk, skipWakeWordCheck, sessionId, skipInject = false }) {
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
        const WAKE_PHRASE_QUICK = /\b(thinkdrop|think\s*drop|hey\s*think|yo\s*think|ok\s*think|armis|hey\s*armis|ok\s*armis|yo\s*armis)\b/i;
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
      // NOTE: detectedLanguage here is a preliminary value from raw STT.
      // The authoritative language (with CJK script-override) comes from toEnglish() below.
      const rawDetectedLanguage = normalizeLanguage(sttResult.language);
      journal.setVoiceStatus('processing', { detectedLanguage: rawDetectedLanguage });

      if (!pushToTalk && !skipWakeWordCheck) {
        // ── Active session window bypass ───────────────────────────────────────
        // If the user said the wake word recently (within 15s), skip the wake-word
        // check entirely — they are already in an active conversation.
        const inActiveSession = journal.isWakeSessionActive();
        // CJK-dominant utterances cannot physically contain a Latin wake word.
        // If the user has been active in the last 5 minutes, treat it as in-session
        // and refresh the 30s window so the conversation stays open.
        // We check "mostly CJK" (>50% non-ASCII) rather than "purely CJK" so that
        // mixed utterances like "你没有帮我去Amazon" still bypass the wake check.
        const nonAsciiCount = (cleanedText.match(/[^\x00-\x7F]/g) || []).length;
        const isMostlyCJK = nonAsciiCount > 0 && nonAsciiCount / cleanedText.replace(/\s/g,'').length > 0.5;
        const cjkRecentBypass = isMostlyCJK && journal.isRecentlyActive();
        if (inActiveSession || cjkRecentBypass) {
          if (cjkRecentBypass && !inActiveSession) {
            logger.info('[Pipeline] CJK-recent-session bypass — refreshing wake window', { text: sttResult.text.substring(0, 60) });
            journal.setWakeSession();
          } else {
            logger.info('[Pipeline] Active session window — wake-word check bypassed', { text: sttResult.text.substring(0, 60) });
          }
        } else {
          // ── Minimum word gate ────────────────────────────────────────────────
          // Coughs, throat-clears, single-syllable sounds, and short mumbles that
          // slip past VAD are typically ≤1 word after transcription.
          // Exception: if the single word IS the wake word, let it through so the
          // user can say "Armis" or "ThinkDrop" alone to activate (returns too_short
          // after stripping the wake phrase, so the user gets a prompt to continue).
          const wordCount = cleanedText.split(/\s+/).filter(Boolean).length;
          const hasNonLatinScript = /[\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF\u0600-\u06FF\u0400-\u04FF\u0900-\u097F]/.test(cleanedText);
          const WAKE_PHRASE_QUICK = /\b(thinkdrop|think\s*drop|tinkdrop|tinkdrab|thinktrop|thinkidrop|hey\s*think|yo\s*think|ok\s*think|armis|armus|armys|armies|armas|ormis|hermes|hermis|harris|jarvis|artemis|hey\s*arm[iuy]s|ok\s*arm[iuy]s|yo\s*arm[iuy]s)\b/i;
          const cleanedNoPunct = cleanedText.replace(/[^\w\s\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF\u0600-\u06FF\u0400-\u04FF\u0900-\u097F]/g, ' ').trim();
          const isSingleWordWakeWord = wordCount < 2 && !hasNonLatinScript && WAKE_PHRASE_QUICK.test(cleanedNoPunct);

          if (wordCount < 2 && !hasNonLatinScript && !isSingleWordWakeWord) {
            logger.info('[Pipeline] Wake word skipped — transcript too short (likely noise)', { text: sttResult.text.substring(0, 60), wordCount });
            journal.setVoiceStatus('idle');
            return { success: true, skipped: true, reason: 'too_short', transcript: sttResult.text };
          }

          const wakeResult = wakeWord.detect(sttResult.text);

          if (wakeResult.detected) {
            // Wake word confirmed — open the 15-second active session window
            journal.setWakeSession();
            if (wakeResult.type !== 'cancel' && wakeResult.type !== 'status') {
              sttResult.text = wakeWord.stripWakePhrase(sttResult.text, wakeResult.matchedPhrase);
            }
            // If nothing remains after stripping (user said only the wake word),
            // return too_short so the UI prompts them to continue speaking.
            const remainingText = sttResult.text.trim();
            if (!remainingText) {
              logger.info('[Pipeline] Wake word only — session opened, prompting user to continue', { matched: wakeResult.matchedPhrase });
              journal.setVoiceStatus('idle');
              return { success: true, skipped: true, reason: 'too_short', wakeWordOnly: true, transcript: cleanedText };
            }
          } else {
            // Strict wake-word mode: REQUIRE the wake word — no fallback bypass.
            logger.info('[Pipeline] Wake word not detected — ignoring (strict mode)', { text: sttResult.text.substring(0, 60) });
            journal.setVoiceStatus('idle');
            return { success: true, skipped: true, reason: 'no_wake_word', transcript: sttResult.text };
          }
        }
      } else if (pushToTalk) {
        // PTT always counts as an active interaction — refresh the session window
        // so switching from PTT back to wake-word mode feels seamless.
        journal.setWakeSession();
      }

      // ── Step 3 + 4a: Translate to English AND detect emotion in parallel ──────
      // Translation is a network call (~300-800ms). Emotion detection is CPU-only
      // but uses the transcript — run them concurrently using the raw STT text for
      // emotion detection (close enough) and await both together.
      const [translationResult, emotionResult] = await Promise.all([
        toEnglish(sttResult),
        Promise.resolve(detectEmotion(audioBase64 || '', { transcript: sttResult.text })),
      ]);
      const { englishText, wasTranslated, detectedLanguage } = translationResult;
      logger.info('[Pipeline] Translation', { wasTranslated, detectedLanguage, englishText: englishText.substring(0, 80) });
      // Persist the authoritative language (script-override applied) — answer.js reads this directly.
      if (detectedLanguage) journal.setSessionLanguage(detectedLanguage);
      if (emotionResult.emotion !== 'neutral') {
        logger.info('[Pipeline] Emotion detected', { emotion: emotionResult.emotion, intensity: emotionResult.intensity });
      }

      // ── Step 4b: Voice fingerprint match (fire-and-forget) ───────────────────
      // Attempt to match the speaker against stored fingerprints.
      // If no match, auto-enroll as Primary User if this is the first voice seen.
      // Result is attached to emotionResult.speakerMatch for prompt injection.
      let speakerMatch = null;
      if (audioBase64) {
        try {
          const vfp = require('./voice-fingerprint.cjs');
          const sp  = emotionResult.speakerProfile;

          // Extract features from audio for fingerprint ops
          const features = vfp.extractFeaturesFromBase64(audioBase64, 16000);
          if (features) {
            // Try to match first
            speakerMatch = await vfp.matchSpeaker(
              // matchSpeaker needs raw samples — re-extract inline
              (() => {
                const buf = Buffer.from(audioBase64, 'base64');
                const isWav = buf.length >= 4 && buf.slice(0, 4).toString('ascii') === 'RIFF';
                const offset = isWav ? 44 : 0;
                const s = [];
                for (let _i = offset; _i < buf.length - 1; _i += 2) s.push(buf.readInt16LE(_i) / 32768.0);
                return s;
              })(),
              16000
            );

            // Voice pattern flags for this turn
            const voicePattern = {
              angry:   emotionResult.emotion === 'angry',
              loud:    sp && sp.volume === 'loud',
              whisper: sp && sp.volume === 'whisper',
            };

            if (speakerMatch && speakerMatch.match_level === 'confirmed') {
              // Known speaker — update their fingerprint with new sample
              vfp.enrollOrUpdate(
                null, 16000,
                speakerMatch.speaker_id,
                speakerMatch.speaker_name,
                sp ? sp.gender : speakerMatch.gender,
                sp ? sp.age_group : speakerMatch.age_group,
                voicePattern
              ).catch(() => {});
              logger.info('[Pipeline] Speaker identified', {
                name: speakerMatch.speaker_name,
                confidence: speakerMatch.confidence.toFixed(3),
              });
            } else if (!speakerMatch) {
              // Unknown — auto-enroll as Primary User (can be renamed later)
              const spGender   = sp ? sp.gender   : 'unknown';
              const spAgeGroup = sp ? sp.age_group : 'adult';
              vfp.enrollOrUpdate(
                null, 16000, 'spk_primary',
                'Primary User', spGender, spAgeGroup, voicePattern
              ).catch(() => {});
              logger.debug('[Pipeline] No fingerprint match — enrolled as Primary User');
            }
          }
        } catch (_fpErr) {
          logger.debug('[Pipeline] Fingerprint step skipped', { error: _fpErr.message });
        }
      }
      // Attach speakerMatch to emotionResult so prompt builder can use it
      if (speakerMatch) emotionResult.speakerMatch = speakerMatch;

      // ── Step 5: Route (always fast lane) ─────────────────────────────────────
      const routeResult = router.route(englishText);

      // Write signal to journal if needed (cancel/pause/resume)
      if (routeResult.signalType) {
        const signalId = journal.writeSignal(routeResult.signalType, { message: englishText, sessionId });
        logger.info('[Pipeline] Signal written', { signalType: routeResult.signalType, signalId });
      }

      // ── Step 6: Generate response (always fast lane) ──────────────────────────
      let responseEnglish;
      let responseMetadata = {};

      // PTT text-only: skip all LLM/TTS, return transcript immediately after routing
      if (skipInject) {
        logger.info('[Pipeline] skipInject=true — returning transcript only', { transcript: sttResult.text.substring(0, 60) });
        journal.setVoiceStatus('idle');
        return {
          success: true,
          lane: 'fast',
          transcript: sttResult.text,
          detectedLanguage,
          wasTranslated,
          englishText,
          skipped: false,
        };
      }

      let fullAnswerEnglish = '';
      // Voice-first: always fast lane. SG escalation driven by LLM trigger phrase detection.
      // _fastLaneResponse also pre-starts TTS translation (ttsPromise) in parallel so we
      // can await it concurrently with display translation below — hiding ~300-500ms latency.
      const fastResult = await this._fastLaneResponse(englishText, routeResult, detectedLanguage, emotionResult, sttResult.text);
      responseEnglish = fastResult.text;
      fullAnswerEnglish = fastResult.text;
      responseMetadata = fastResult.metadata || {};

      if (!responseEnglish) {
        journal.setVoiceStatus('idle');
        return { success: true, lane: 'fast', transcript: sttResult.text, englishText, response: null };
      }

      // ── Step 7: TTS ────────────────────────────────────────────────────────
      // Hard cap at 50 words before translation to keep TTS audio short (< 10s).
      // Emotion tags ([happy], [excited], etc.) are preserved for Resemble AI
      // and stripped for all other providers (Cartesia, Inworld, macOS).
      const MAX_SPOKEN_WORDS = 50;
      const spokenWords = responseEnglish.trim().split(/\s+/);
      const cappedEnglish = spokenWords.length > MAX_SPOKEN_WORDS
        ? spokenWords.slice(0, MAX_SPOKEN_WORDS).join(' ') + '.'
        : responseEnglish;
      if (spokenWords.length > MAX_SPOKEN_WORDS) {
        logger.info('[Pipeline] TTS text capped', { original: spokenWords.length, capped: MAX_SPOKEN_WORDS });
      }

      // Strip emotion tags for non-Resemble providers (they'd be read aloud literally)
      const isResembleActive = voiceProvider.getProviderName() === 'resemble';
      const ttsEnglish = isResembleActive
        ? cappedEnglish
        : _stripEmotionTags(cappedEnglish);

      // Use pre-started ttsPromise from _fastLaneResponse if available (English already handled).
      // For non-English: ttsPromise was kicked off in parallel with this code — just await it.
      // For English: fastResult.ttsPromise resolves immediately with the original text.
      const responseInUserLanguage = fastResult.ttsPromise
        ? await fastResult.ttsPromise
        : await fromEnglish(ttsEnglish, detectedLanguage);

      // Translate fullAnswer + responseEnglish for ResultsWindow display when non-English.
      // Use sessionLanguage from journal as fallback when wasTranslated is false.
      // Always strip emotion tags from display text (they're TTS-only markup).
      const displayLang = (detectedLanguage && detectedLanguage !== 'en')
        ? detectedLanguage
        : (journal.read().voice?.sessionLanguage || 'en');
      const needsDisplayTranslation = displayLang !== 'en';

      const responseEnglishClean = _stripEmotionTags(responseEnglish);
      const fullAnswerEnglishClean = _stripEmotionTags(fullAnswerEnglish);

      let fullAnswerDisplay = fullAnswerEnglishClean;
      let responseDisplay = responseEnglishClean;
      // Only call fromEnglish for display if the LLM responded in English AND the user speaks
      // a different language. When detectedLanguage is non-English, the LLM already responded
      // in the target language directly — no translation needed, avoids an extra LLM call.
      const llmRespondedInEnglish = !detectedLanguage || detectedLanguage === 'en';
      if (needsDisplayTranslation && llmRespondedInEnglish) {
        try {
          const translateFull = fullAnswerEnglishClean
            ? fromEnglish(fullAnswerEnglishClean, displayLang).catch(() => fullAnswerEnglishClean)
            : Promise.resolve(fullAnswerEnglishClean);
          const translateResp = responseEnglishClean
            ? fromEnglish(responseEnglishClean, displayLang).catch(() => responseEnglishClean)
            : Promise.resolve(responseEnglishClean);
          [fullAnswerDisplay, responseDisplay] = await Promise.all([translateFull, translateResp]);
          logger.info('[Pipeline] Display text translated', { displayLang });
        } catch (_) {}
      }

      // Build TTS options — pass exaggeration for Resemble (maps emotion intensity to vocal expressiveness)
      const isResembleForTTS = voiceProvider.getProviderName() === 'resemble';
      const ttsOpts = { text: responseInUserLanguage, language: detectedLanguage };
      if (isResembleForTTS && emotionResult && emotionResult.intensity) {
        const { _intensityToExaggeration } = require('./providers/resemble.cjs');
        ttsOpts.exaggeration = _intensityToExaggeration(emotionResult.intensity);
      }

      journal.setVoiceStatus('speaking');
      const ttsResult = await voiceProvider.synthesize(ttsOpts);
      journal.setVoiceStatus('idle', { lastSpokenAt: new Date().toISOString() });

      // ── Mood event emission (fire-and-forget — does not affect response) ───────
      try {
        const textSignal = require('./text-signal.cjs');
        const textEvt = textSignal.analyze(englishText, []);
        if (textEvt.event_type) {
          _fireMoodEvent(textEvt.event_type, 'voice', textEvt.reason);
        } else if (emotionResult && emotionResult.emotion === 'angry') {
          _fireMoodEvent('user_raised_voice', 'voice', 'Audio emotion detected: angry');
        } else if (emotionResult && (emotionResult.emotion === 'sad' || emotionResult.emotion === 'empathetic')) {
          _fireMoodEvent('user_frustrated', 'voice', 'Audio emotion detected: ' + emotionResult.emotion);
        } else if (emotionResult && emotionResult.emotion === 'happy') {
          _fireMoodEvent('positive_feedback', 'voice', 'Audio emotion detected: happy');
        }

        // ── Name introduction detection ─────────────────────────────────────────
        // If the user says "my name is Luka" or "call me Emma", rename their fingerprint.
        // Uses the matched speaker_id from the fingerprint step above (speakerMatch).
        const detectedName = textSignal.extractName(englishText);
        if (detectedName) {
          const sid = speakerMatch ? speakerMatch.speaker_id : 'spk_primary';
          const renameBody = JSON.stringify({
            version: 'mcp.v1', service: 'user-memory', action: 'fingerprint.rename',
            payload: { speaker_id: sid, speaker_name: detectedName },
            requestId: 'vfp_rename_' + Date.now(),
          });
          const rnReq = http.request({
            hostname: '127.0.0.1', port: parseInt(process.env.MEMORY_SERVICE_PORT || '3001', 10),
            path: '/fingerprint.rename', method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(renameBody) },
            timeout: 4000,
          }, () => {});
          rnReq.on('error', () => {});
          rnReq.on('timeout', () => { rnReq.destroy(); });
          rnReq.write(renameBody);
          rnReq.end();
          logger.info('[Pipeline] Speaker renamed from voice intro', { name: detectedName, speaker_id: sid });
        }
      } catch (_moodErr) {}

      // ── Speaker profile persistence (fire-and-forget) ────────────────────────
      // Write detected gender/age to personality_traits so heartbeat + synthesis
      // agent can read the user's profile across sessions.
      // Only writes when confidence is high (not 'unknown') to avoid overwriting
      // a good detection with a bad one.
      try {
        const sp = emotionResult && emotionResult.speakerProfile;
        if (sp && sp.gender !== 'unknown' && sp.gender) {
          const profileValue = sp.gender + '_' + sp.ageGroup; // e.g. 'male_adult', 'female_adult', 'child_child'
          const body = JSON.stringify({
            version: 'mcp.v1', service: 'personality-service', action: 'personality.upsertTrait',
            payload: { trait_key: 'speaker_profile', trait_value: profileValue, source: 'learned', weight: 1.0 },
            requestId: 'vs_sp_' + Date.now(),
          });
          const spReq = http.request({
            hostname: '127.0.0.1', port: PERSONALITY_SERVICE_PORT,
            path: '/personality.upsertTrait', method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) },
            timeout: 3000,
          }, () => {});
          spReq.on('error', () => {});
          spReq.on('timeout', () => { spReq.destroy(); });
          spReq.write(body);
          spReq.end();
        }
      } catch (_spErr) {}

      const pipelineResult = {
        success: true,
        lane: 'fast',
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
        triggered: !!(responseMetadata.triggered),  // true when stategraph escalation fired in background
        metadata: responseMetadata,
      };
      // Refresh the wake session on every successful response so multi-turn
      // conversations stay alive without requiring the wake word again.
      journal.setWakeSession();
      return pipelineResult;
    } catch (error) {
      journal.setVoiceStatus('idle');
      logger.error('[Pipeline] Error', { error: error.message });
      return { success: false, error: error.message };
    }
  }

  // ─── Fast Lane (voice-first — ALL utterances go here) ────────────────────────

  /**
   * @param {string} englishText
   * @param {{ reason: string, signalType: string|null }} routeResult
   * @param {string|null} responseLanguage
   * @param {{ emotion: string, intensity: string, tags: string[], context: string }} emotionResult
   */
  async _fastLaneResponse(englishText, routeResult, responseLanguage = null, emotionResult = null, originalText = null) {
    // Kick off personality overlay fetch immediately — runs in parallel with signal checks
    // and prompt assembly so the ~200ms network round-trip is mostly hidden.
    const overlayPromise = _fetchPersonalityOverlay();

    // ── Control signal short-circuits (no LLM needed) ─────────────────────────
    if (routeResult.reason === 'sg_noise_suppressed') {
      return { text: null, metadata: { source: 'sg_noise_suppressed' } };
    }

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

    // ── Build system prompt ────────────────────────────────────────────────────
    const _fastLang = responseLanguage || journal.read().voice?.sessionLanguage || 'en';
    const isResembleActive = voiceProvider.getProviderName() === 'resemble';
    logger.info('[FastLane] Language', { _fastLang, responseLanguage, sessionLanguage: journal.read().voice?.sessionLanguage });

    // Await the personality overlay — by now the request is already in-flight
    const personalityOverlay = await overlayPromise;
    let _fastSystemPrompt = FAST_LANE_BASE_PROMPT;
    if (personalityOverlay) {
      _fastSystemPrompt += '\n\n' + personalityOverlay;
    }

    // ── Speaker profile injection ──────────────────────────────────────────────
    // Detected from audio ZCR / RMS + fingerprint DB match.
    const sp  = emotionResult && emotionResult.speakerProfile;
    const sm  = emotionResult && emotionResult.speakerMatch;  // fingerprint match result
    if (sp && sp.gender !== 'unknown') {
      const volNote = sp.volume === 'loud'    ? 'They are speaking loudly — acknowledge their energy without escalating.'
                    : sp.volume === 'whisper' ? 'They are whispering — respond softly and gently.'
                    : sp.volume === 'quiet'   ? 'They are speaking quietly — be warm and measured.'
                    : '';
      const ageNote = sp.ageGroup === 'child'
        ? 'This is a child. Use simple, warm, encouraging language. No complex vocabulary. Be patient and kind.'
        : '';
      const addressNote = sp.address
        ? `Address them as "${sp.address}" when appropriate (not every sentence — use sparingly and naturally).`
        : '';
      // If we have a confirmed fingerprint match with a real name, add it
      const nameNote = (sm && sm.speaker_name && sm.speaker_name !== 'Primary User' && sm.match_level === 'confirmed')
        ? `\nIdentified speaker: ${sm.speaker_name} (confidence: ${(sm.confidence * 100).toFixed(0)}%). You may use their name naturally in conversation.`
        : sm && sm.match_level === 'confirmed'
        ? `\nSpeaker recognized as Primary User (${(sm.confidence * 100).toFixed(0)}% match).`
        : '';
      // Voice pattern history (if known speaker with a history)
      const historyNote = sm && (sm.angry_count > 5 || sm.whisper_count > 10)
        ? `\nVoice history: angry turns=${sm.angry_count}, whisper turns=${sm.whisper_count}, loud turns=${sm.loud_count}. Adapt tone accordingly.`
        : '';
      _fastSystemPrompt += `\n\n═══ SPEAKER PROFILE ═══\nDetected: ${sp.gender.toUpperCase()}${sp.ageGroup === 'child' ? ' (CHILD)' : ''} | Volume: ${sp.volume.toUpperCase()}${sp.f0Estimate > 0 ? ` | F0 ≈${Math.round(sp.f0Estimate)}Hz` : ''}\n${addressNote}${ageNote ? '\n' + ageNote : ''}${volNote ? '\n' + volNote : ''}${nameNote}${historyNote}\n═══════════════════`;
    }

    // Only inject emotion tag instructions when Resemble is active — other
    // providers (Cartesia, Inworld, macOS) read the brackets literally.
    if (isResembleActive && emotionResult && emotionResult.emotion !== 'neutral' && emotionResult.context) {
      _fastSystemPrompt += `\n\n═══ USER EMOTIONAL STATE ═══\nDetected: ${emotionResult.emotion.toUpperCase()} (intensity: ${emotionResult.intensity})\n${emotionResult.context}\nSuggested tag: ${emotionResult.tags.join(', ') || 'none'}\n═══════════════════════════`;
    } else if (!isResembleActive) {
      _fastSystemPrompt += `\n\nDo NOT use any emotion tags like [happy] or [excited] in your response. Plain text only.`;
    }

    if (_fastLang && _fastLang !== 'en') {
      _fastSystemPrompt += `\n\nIMPORTANT: The user is speaking ${LANGUAGE_NAMES[_fastLang] || _fastLang}. You MUST respond entirely in ${LANGUAGE_NAMES[_fastLang] || _fastLang}. Do not use English.`;
    }

    // ── LLM call (early-resolve) + combined trigger classification ────────────
    // When the user speaks non-English, pass the ORIGINAL text directly to the LLM
    // (which already knows to respond in the target language from _fastLang injection).
    // This skips the translate-in LLM call, cutting latency from ~4s to ~1s.
    const isNonEnglish = _fastLang && _fastLang !== 'en';
    const llmInputText = isNonEnglish && originalText ? originalText : englishText;
    const { classifySGTrigger } = require('./voice-router.cjs');

    try {
      const { firstSentence, fullText } = await this._directLLMQueryEarly(llmInputText, _fastSystemPrompt);
      const spokenAnswer = firstSentence || fullText;
      if (!spokenAnswer || !spokenAnswer.trim()) {
        return { text: "Consider it noted, though I've nothing to add at the moment.", metadata: { source: 'fast_llm_empty' } };
      }

      // ── Combined trigger classification (user intent + full LLM response) ───
      const { shouldTrigger, triggerType } = classifySGTrigger(englishText, fullText);

      if (shouldTrigger) {
        logger.info('[FastLane] SG trigger classified — escalating in background', {
          prompt: englishText.substring(0, 80),
          triggerType,
          response: fullText.substring(0, 80),
        });
        this._escalateToStateGraph(englishText, triggerType).catch(err => {
          logger.error('[FastLane] SG escalation error', { error: err.message });
        });
      }

      // ── TTS: when non-English, LLM already responded in target language — use directly ───
      // Skip fromEnglish() translate-back call entirely for non-English (saves another ~800ms).
      // For English, ttsPromise resolves immediately with the original text.
      const ttsPromise = isNonEnglish
        ? Promise.resolve(spokenAnswer)   // LLM answered in Chinese/Spanish/etc. directly
        : responseLanguage && responseLanguage !== 'en'
          ? fromEnglish(spokenAnswer, responseLanguage).catch(() => spokenAnswer)
          : Promise.resolve(spokenAnswer);

      return { text: spokenAnswer, ttsPromise, metadata: { source: 'fast_llm', triggered: shouldTrigger, triggerType, fullText } };
    } catch (error) {
      logger.error('[FastLane] LLM error', { error: error.message });
      return { text: "My apologies — the circuits are occupied. Try again in a moment.", metadata: { source: 'error' } };
    }
  }

  /**
   * Background StateGraph escalation — fires after fast lane returns its holding phrase.
   * Result is POSTed to /voice/result on the main process overlay port so the
   * renderer receives it as a second voice:response after the TTS completes.
   *
   * @param {string} englishText - Original user prompt
   * @param {string} triggerType - 'action_lookup' | 'research' | 'computer_action'
   */
  async _escalateToStateGraph(englishText, triggerType) {
    const escalationLang = journal.read().voice?.sessionLanguage || 'en';
    const sgResult = await this._stategraphLaneResponse(englishText, null, escalationLang !== 'en' ? escalationLang : null, true);
    if (!sgResult.text) return;

    // Translate spoken summary back to user's language before TTS
    const ttsLang = (escalationLang && escalationLang !== 'en') ? escalationLang : 'en';
    const spokenText = (ttsLang !== 'en')
      ? await fromEnglish(sgResult.text, ttsLang).catch(() => sgResult.text)
      : sgResult.text;

    // Synthesize TTS for the stategraph result
    const ttsResult = await voiceProvider.synthesize({ text: spokenText, language: ttsLang }).catch(() => null);
    if (!ttsResult) return;

    // POST result back to main process → forwards voice:response to renderer
    const http_module = require('http');
    const payload = JSON.stringify({
      text: sgResult.text,
      fullAnswer: sgResult.fullAnswer || sgResult.text,
      audioBase64: ttsResult.audioBuffer.toString('base64'),
      audioFormat: ttsResult.format || 'mp3',
      language: ttsLang,
      lane: 'stategraph',
      triggerType,
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
    logger.info('[FastLane] SG escalation result sent to renderer', {
      text: sgResult.text.substring(0, 60),
      triggerType,
    });
  }

  // ─── StateGraph Lane ──────────────────────────────────────────────────────────

  async _stategraphLaneResponse(englishText, sessionId, responseLanguage = null, voiceOnly = false) {
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
      const result = await this._injectIntoStateGraph(englishText, sessionId, responseLanguage, voiceOnly);
      const fullAnswer = result.answer || result.text || '';
      const intent = result.intent || 'unknown';
      const skillResults = result.skillResults || [];

      // ── Extract the best spoken answer from skill results ────────────────────
      // Priority: (1) short stdout from a successful step (e.g. `date` output = "08:54:18"),
      // (2) fullAnswer if it looks like real content, (3) generic confirmation.
      // This prevents the LLM summarizer from being called with useless strings like
      // "All 1 step completed successfully." and hallucinating "I can't determine the time."
      const usableStdout = skillResults
        .filter(r => r.ok && r.stdout && typeof r.stdout === 'string' && r.stdout.trim().length > 0)
        .map(r => r.stdout.trim())
        .join(' ');

      // Detect generic/useless answer strings from executeCommand that shouldn't be spoken verbatim.
      // These are internal status messages, not meaningful answers for voice.
      const isGenericAnswer = !fullAnswer || /^(all \d+ steps? completed|completed \d+\/\d+|done\.?$|got it\.?$)/i.test(fullAnswer.trim())
        || /^All \d+ steps? completed successfully/i.test(fullAnswer.trim());

      // If the only content is a generic completion message, use actual step stdout instead
      let contentForVoice = fullAnswer;
      let usedRawStdout = false;
      if (isGenericAnswer && usableStdout) {
        contentForVoice = usableStdout;
        usedRawStdout = true;
        logger.info('[StateGraphLane] Using skill stdout as spoken answer instead of generic completion message', { stdout: usableStdout.substring(0, 80) });
      }

      // If still nothing useful, return a short confirmation
      if (!contentForVoice || contentForVoice.trim().length === 0) {
        const confirmation = intent === 'command_automate' ? 'Done.' : 'Got it.';
        logger.info('[StateGraphLane] Command lane — using confirmation', { intent, confirmation });
        return { text: confirmation, fullAnswer: '', hadLiveStream: !!result.hadLiveStream, metadata: { source: 'stategraph', intent } };
      }

      // Generate a short spoken summary (≤20 words) — full answer already streamed to Results window.
      // Also naturalise raw stdout (e.g. "Sun Mar 8 09:06:32 EDT 2026" → "It's 9:06 AM.") so it
      // sounds like a human response rather than a terminal printout.
      const SUMMARY_WORD_LIMIT = 20;
      let spokenSummary = contentForVoice;
      const needsNaturalisation = usedRawStdout || contentForVoice.split(/\s+/).length > SUMMARY_WORD_LIMIT;
      if (needsNaturalisation) {
        try {
          const raw = await this._directLLMQuery(
            `You are a voice assistant. Convert the following raw data into ONE natural spoken sentence (≤15 words). Sound like a human, not a terminal. No markdown.\n\nUser asked: "${englishText}"\nRaw data: ${contentForVoice.substring(0, 400)}`
          );
          // Hard-cap regardless of what LLM returns
          const words = raw.trim().split(/\s+/);
          spokenSummary = words.length > SUMMARY_WORD_LIMIT
            ? words.slice(0, SUMMARY_WORD_LIMIT).join(' ') + '.'
            : raw.trim();
          logger.info('[StateGraphLane] Generated spoken summary', { fullWords: contentForVoice.split(/\s+/).length, summaryWords: spokenSummary.split(/\s+/).length, naturalised: usedRawStdout });
        } catch (_) {
          spokenSummary = contentForVoice.split(/\s+/).slice(0, SUMMARY_WORD_LIMIT).join(' ') + '.';
        }
      }

      return {
        text: spokenSummary,
        fullAnswer: fullAnswer || contentForVoice,
        hadLiveStream: !!result.hadLiveStream,
        metadata: { source: 'stategraph', intent },
      };
    } catch (error) {
      logger.error('[StateGraphLane] Error', { error: error.message });
      return { text: "I encountered an error processing your request.", metadata: { source: 'error' } };
    }
  }

  // ─── Direct LLM (fast lane) ───────────────────────────────────────────────────
  // Uses llm-providers.cjs: openai → claude → gemini → grok → mistral → deepseek
  // Completely decoupled from port 4000 StateGraph WebSocket.
  // WS fallback only fires when ALL HTTPS providers fail (no keys / all rate-limited).

  /**
   * Ask the provider chain, return full text string.
   */
  async _directLLMQuery(text, systemPrompt = null) {
    const { ask, buildMessages } = require('./llm-providers.cjs');
    const messages = buildMessages(text, systemPrompt);
    const { text: result, provider } = await ask(messages, { maxTokens: 100, temperature: 0.7 });
    if (result) return result;
    // All HTTPS providers failed — fall back to WS
    logger.warn('[FastLane] All HTTPS providers exhausted, falling back to WS');
    return this._directLLMQueryViaWS(text, systemPrompt);
  }

  /**
   * Ask the provider chain, resolve early at first sentence boundary.
   * Uses askEarly() which streams OpenAI and resolves at first '.?!' (~300-500ms).
   * Returns { firstSentence, fullText } so TTS can start immediately.
   */
  async _directLLMQueryEarly(text, systemPrompt = null) {
    const { askEarly, buildMessages } = require('./llm-providers.cjs');
    const messages = buildMessages(text, systemPrompt);
    const { firstSentence, fullText, provider } = await askEarly(messages, { maxTokens: 150, temperature: 0.7 });
    if (fullText) {
      return { firstSentence, fullText };
    }
    // All HTTPS providers failed — fall back to WS
    logger.warn('[FastLane] All HTTPS providers exhausted, falling back to WS (early)');
    return this._directLLMQueryEarlyViaWS(text, systemPrompt);
  }

  // ─── WS fallback (used only when ALL HTTPS providers fail) ───────────────────

  _directLLMQueryViaWS(text, systemPrompt = null) {
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
          id: requestId, type: 'llm_request',
          payload: { prompt: text, provider: 'openai', options: { maxTokens: 100, temperature: 0.7, model: 'gpt-4o-mini', stream: true }, context: { systemInstructions: systemPrompt || '' } },
          timestamp: Date.now(), metadata: { source: 'voice_fast_lane' },
        }));
      });
      ws.on('message', (data) => {
        try {
          const msg = JSON.parse(data.toString());
          if (msg.type === 'llm_stream_chunk' && msg.payload?.text) { fullText += msg.payload.text; }
          else if (msg.type === 'llm_stream_end') { if (!settled) { settled = true; clearTimeout(timeout); ws.close(); resolve(fullText); } }
          else if (msg.type === 'error') { if (!settled) { settled = true; clearTimeout(timeout); ws.close(); reject(new Error(msg.payload?.message || 'LLM error')); } }
        } catch (_) {}
      });
      ws.on('error', (err) => { if (!settled) { settled = true; clearTimeout(timeout); reject(err); } });
      ws.on('close', () => { if (!settled) { settled = true; clearTimeout(timeout); resolve(fullText || ''); } });
    });
  }

  _directLLMQueryEarlyViaWS(text, systemPrompt = null) {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(STATEGRAPH_WS_URL, {
        headers: STATEGRAPH_API_KEY ? { Authorization: `Bearer ${STATEGRAPH_API_KEY}` } : {},
      });
      const requestId = `voice_fast_early_${Date.now()}`;
      let fullText = '';
      let firstSentence = '';
      let earlyResolved = false;
      let settled = false;
      const checkEarlyResolve = () => {
        if (earlyResolved) return;
        const match = fullText.match(/^[^.?!]*[.?!]/);
        if (match && match[0].trim().length > 4) { firstSentence = match[0].trim(); earlyResolved = true; }
      };
      const timeout = setTimeout(() => {
        if (!settled) { settled = true; ws.close(); const fs = firstSentence || fullText || 'Forgive the delay — no answer came in time.'; resolve({ firstSentence: fs, fullText: fullText || fs }); }
      }, 15000);
      ws.on('open', () => {
        ws.send(JSON.stringify({
          id: requestId, type: 'llm_request',
          payload: { prompt: text, provider: 'openai', options: { maxTokens: 100, temperature: 0.7, model: 'gpt-4o-mini', stream: true }, context: { systemInstructions: systemPrompt || '' } },
          timestamp: Date.now(), metadata: { source: 'voice_fast_lane' },
        }));
      });
      ws.on('message', (data) => {
        try {
          const msg = JSON.parse(data.toString());
          if (msg.type === 'llm_stream_chunk' && msg.payload?.text) { fullText += msg.payload.text; checkEarlyResolve(); }
          else if (msg.type === 'llm_stream_end') { if (!settled) { settled = true; clearTimeout(timeout); ws.close(); const fs = firstSentence || fullText.trim(); resolve({ firstSentence: fs, fullText: fullText.trim() }); } }
          else if (msg.type === 'error') { if (!settled) { settled = true; clearTimeout(timeout); ws.close(); reject(new Error(msg.payload?.message || 'LLM error')); } }
        } catch (_) {}
      });
      ws.on('error', (err) => { if (!settled) { settled = true; clearTimeout(timeout); reject(err); } });
      ws.on('close', () => { if (!settled) { settled = true; clearTimeout(timeout); const fs = firstSentence || fullText || ''; resolve({ firstSentence: fs, fullText: fullText || fs }); } });
    });
  }

  // ─── StateGraph Injection ─────────────────────────────────────────────────────

  async _injectIntoStateGraph(text, sessionId, responseLanguage = null, voiceOnly = false) {
    return new Promise((resolve, reject) => {
      const http_module = require('http');
      const body = JSON.stringify({ message: text, sessionId, source: 'voice', responseLanguage, voiceOnly });

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

    // Reset sessionLanguage on startup — journal persists language across restarts
    // which causes the previous session's language (e.g. 'zh') to lock all subsequent
    // audio through the wrong Whisper decoder. Always start fresh in English.
    journal.setSessionLanguage('en');

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
              case '/voice.classify':
                result = await this.classifyFreshTask(payload);
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
            const ttsResult = await voiceProvider.synthesize({ text: sgResult.text, language: 'en' }).catch(() => null);
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

  /**
   * Classify whether a new user prompt is a fresh task vs a reply to a paused question.
   * Delegates to voice-classifier.cjs classifyFreshTask (embedding cosine similarity).
   * payload: { prompt: string, pausedQuestion?: string }
   */
  async classifyFreshTask({ prompt, pausedQuestion } = {}) {
    return _classifyFreshTask(prompt, pausedQuestion || null);
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
