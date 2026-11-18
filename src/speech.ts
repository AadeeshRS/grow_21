import {Platform} from 'react-native';
import * as Speech from 'expo-speech';

/**
 * Spoken instructions and feedback. Every child-facing instruction is spoken (and can be replayed),
 * because many children with Down syndrome are stronger visual than auditory learners and may not
 * read yet: speech supports the picture, it never replaces it.
 *
 * Voice quality: on the web, when the host configures `ttsUrl` (the learner service's
 * /api/v1/tts, a neural Piper voice), that audio is played. Otherwise, or if it fails, the device
 * voice is used, preferring the most natural one installed ("Natural"/"Neural"/"Online" voices).
 * Speech failures are swallowed: a missing voice must never block play.
 */
let language = 'en-IN';
let enabled = true;
let ttsUrl: string | undefined;
let deviceVoice: string | undefined;
let current: {pause(): void; src: string} | null = null;
/** Guards against the same line being spoken twice. React StrictMode runs effects
 *  twice in development, which otherwise makes every instruction stutter. */
let lastSpoken = {text: '', at: 0};
const REPEAT_GUARD_MS = 1200;

export function configureSpeech(options: {language?: string; enabled?: boolean; ttsUrl?: string}) {
  if (options.language) language = options.language;
  if (options.enabled !== undefined) enabled = options.enabled;
  if (options.ttsUrl !== undefined) ttsUrl = options.ttsUrl || undefined;
  void pickDeviceVoice();
}

/** Ranks installed voices: natural/neural first, then known good voices, then the right language. */
export function rankVoice(v: {name?: string; identifier?: string; language?: string; quality?: string}, lang: string): number {
  const name = `${v.name ?? ''} ${v.identifier ?? ''}`.toLowerCase();
  let score = 0;
  if (/natural|neural|online|premium|enhanced/.test(name) || v.quality === 'Enhanced') score += 100;
  if (/google|samantha|aria|jenny|libby|sonia|neerja|ava|zira|karen|moira|tessa|serena/.test(name)) score += 40;
  if (/espeak|robot|whisper|zarvox|bad news|bells|novelty/.test(name)) score -= 200;
  const l = (v.language ?? '').toLowerCase().replace('_', '-');
  if (l === lang.toLowerCase()) score += 30;
  else if (l.startsWith('en')) score += 20;
  else score -= 500;
  return score;
}

async function pickDeviceVoice() {
  try {
    const voices = await Speech.getAvailableVoicesAsync();
    const best = [...voices].sort((a, b) => rankVoice(b, language) - rankVoice(a, language))[0];
    deviceVoice = best && rankVoice(best, language) > -100 ? best.identifier : undefined;
  } catch {
    deviceVoice = undefined;
  }
}

function speakDevice(text: string, onDone?: () => void) {
  try {
    Speech.stop();
    Speech.speak(text, {language, voice: deviceVoice, rate: 0.9, pitch: 1.0, onDone, onStopped: onDone, onError: () => onDone?.()});
  } catch {
    onDone?.();
  }
}

export function say(text: string, onDone?: () => void) {
  if (!enabled) { onDone?.(); return; }
  const now = Date.now();
  if (text === lastSpoken.text && now - lastSpoken.at < REPEAT_GUARD_MS) { onDone?.(); return; }
  lastSpoken = {text, at: now};
  stopSpeaking();
  const AudioCtor = (globalThis as {Audio?: new (src: string) => HTMLAudioElement}).Audio;
  if (Platform.OS === 'web' && ttsUrl && AudioCtor) {
    const audio = new AudioCtor(`${ttsUrl}?text=${encodeURIComponent(text)}`);
    let settled = false;
    const done = () => { if (!settled) { settled = true; onDone?.(); } };
    const fallback = () => {
      if (settled || current !== audio) return;
      try { audio.pause(); } catch { /* already stopped */ }
      current = null;
      speakDevice(text, done);
    };
    audio.onended = done;
    audio.onerror = fallback;
    current = audio;
    audio.play().catch(fallback);
    return;
  }
  speakDevice(text, onDone);
}

/**
 * Speaks a line, then runs `after` once it has actually finished.
 *
 * Screen changes used to be on a fixed timer, so a line longer than the timer
 * was cut off halfway. `minMs` keeps a short line on screen long enough to read,
 * `maxMs` stops a stuck voice from blocking play.
 */
export function sayThen(text: string, after: () => void, minMs = 900, maxMs?: number) {
  // Without a cap a silent or stuck voice would freeze the screen, but a fixed
  // cap cuts long lines off, which is the bug this replaced. So the cap is read
  // from the line itself: roughly how long it takes to say, plus a little slack.
  const words = text.trim().split(/\s+/).length;
  const cap = maxMs ?? Math.max(minMs, 700 + words * 420);
  const start = Date.now();
  let fired = false;
  const run = () => { if (!fired) { fired = true; after(); } };
  const finish = () => setTimeout(run, Math.max(0, minMs - (Date.now() - start)));
  say(text, finish);
  setTimeout(run, cap);
}

export function stopSpeaking() {
  lastSpoken = {text: '', at: 0};
  if (current) { current.pause(); current = null; }
  try { Speech.stop(); } catch { /* no speech engine */ }
}

const PRAISE = ['Great job!', 'You did it!', 'Yay! Well done!', 'Super!', 'Wonderful!'];
export const praise = (n: number) => PRAISE[n % PRAISE.length];
/** Retry language is the same every time: consistent, calm, never "wrong". */
export const RETRY = "Let's try again.";
export const HINT = "Let's try together. Here is a little help.";
export const NICE_TRY = 'Nice trying! Let’s play something new.';
