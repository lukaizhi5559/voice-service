'use strict';

/**
 * companion-driver.cjs — Hidden Chrome voice-worker launcher.
 *
 * Owns the Playwright lifecycle for the ThinkDrop voice session:
 *   - Launches REAL Google Chrome (`channel: 'chrome'`) in headed mode,
 *     sized 1×1 and parked offscreen — identical to the hidden-headed
 *     pattern used by command-service's browser-engine.cjs.
 *   - webkitSpeechRecognition only works in real Chrome (not Chromium/CfT)
 *     and NOT in headless mode, so headed + invisible is the only option.
 *   - Mic permission is auto-granted via --use-fake-ui-for-media-stream and
 *     context.grantPermissions(['microphone']); the persistent profile
 *     remembers the grant across runs.
 *
 * The driver only manages the browser/page lifecycle. All voice traffic
 * (transcripts, state, TTS audio) flows over the /voice.ws WebSocket owned
 * by server.cjs — the page connects back to us.
 */

const path = require('path');
const os = require('os');
const { chromium } = require('playwright');
const logger = require('./logger.cjs');

const PROFILE_DIR = path.join(os.homedir(), '.thinkdrop', 'browser-profiles', 'voice-companion');

// Same hidden-window args as mcp-services/command-service/src/skills/browser-engine.cjs
const HIDDEN_ARGS = [
  '--window-size=1,1',
  '--window-position=-32000,-32000',
  '--window-workspace=-32000',
  '--use-fake-ui-for-media-stream', // auto-accept mic prompt (real audio, NOT fake-device)
  '--autoplay-policy=no-user-gesture-required', // AudioContext without a user gesture
  '--disable-blink-features=AutomationControlled',
  '--no-first-run',
  '--no-default-browser-check',
  '--disable-component-update',
];

class CompanionDriver {
  /**
   * @param {{ workerUrl: () => string, onDisconnect: (reason: string) => void }} opts
   */
  constructor(opts) {
    this.workerUrl = opts.workerUrl;
    this.onDisconnect = opts.onDisconnect || (() => {});
    this.browser = null;
    this.context = null;
    this.page = null;
    this.starting = false;
  }

  get running() {
    return !!(this.browser && this.browser.isConnected() && this.page && !this.page.isClosed());
  }

  /**
   * Launch (or reuse) the hidden Chrome and navigate to the worker page.
   * Resolves once the page has loaded; the WS 'ready' handshake is
   * confirmed separately by server.cjs.
   */
  async start() {
    if (this.running) {
      // Page may have navigated away — bring it back to the worker.
      if (!this.page.url().startsWith(this.workerUrl())) {
        await this.page.goto(this.workerUrl(), { waitUntil: 'domcontentloaded' });
      }
      return { ok: true, reused: true };
    }
    if (this.starting) throw new Error('launch already in progress');
    this.starting = true;
    try {
      this.context = await chromium.launchPersistentContext(PROFILE_DIR, {
        channel: 'chrome', // REQUIRED: real Google Chrome for SpeechRecognition
        headless: false,   // REQUIRED: headless cannot grant mic permission
        viewport: { width: 1, height: 1 },
        permissions: ['microphone'],
        ignoreDefaultArgs: ['--enable-automation'],
        args: HIDDEN_ARGS,
      });
      this.browser = this.context.browser();
      this.browser.on('disconnected', () => {
        logger.warn('[VoiceDriver] Chrome disconnected');
        this._cleanup();
        this.onDisconnect('browser-disconnected');
      });
      this.page = this.context.pages()[0] || (await this.context.newPage());
      this.page.on('crash', () => {
        logger.warn('[VoiceDriver] Worker page crashed');
        this.onDisconnect('page-crash');
      });
      this.page.on('close', () => {
        logger.warn('[VoiceDriver] Worker page closed');
        this.onDisconnect('page-closed');
      });
      await this.page.goto(this.workerUrl(), { waitUntil: 'domcontentloaded' });
      logger.info('[VoiceDriver] Hidden Chrome worker launched', { url: this.workerUrl() });
      return { ok: true, reused: false };
    } finally {
      this.starting = false;
    }
  }

  /** Suspend the session — keep the browser resident for instant re-entry. */
  async stop() {
    // STT suspension is driven over WS by server.cjs; nothing to do at the
    // browser level. Kept for symmetry / future use.
  }

  /** Fully tear down the browser. */
  async close() {
    try { if (this.context) await this.context.close(); } catch (_) {}
    this._cleanup();
  }

  status() {
    return {
      running: this.running,
      starting: this.starting,
      pageUrl: this.running ? this.page.url() : null,
      profileDir: PROFILE_DIR,
    };
  }

  _cleanup() {
    this.browser = null;
    this.context = null;
    this.page = null;
  }
}

module.exports = { CompanionDriver, PROFILE_DIR };
