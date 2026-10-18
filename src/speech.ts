import * as Speech from 'expo-speech';

/**
 * Spoken instructions and feedback. Every child-facing instruction is spoken (and can be replayed),
 * because many children with Down syndrome are stronger visual than auditory learners and may not
 * read yet: speech supports the picture, it never replaces it.
 * Speech failures are swallowed: a missing voice must never block play.
 */
let language = 'en-IN';
let enabled = true;

export function configureSpeech(options: {language?: string; enabled?: boolean}) {
  if (options.language) language = options.language;
  if (options.enabled !== undefined) enabled = options.enabled;
}

export function say(text: string, onDone?: () => void) {
  if (!enabled) { onDone?.(); return; }
  try {
    Speech.stop();
    // Slightly slower than default speech helps auditory processing.
    Speech.speak(text, {language, rate: 0.85, pitch: 1.05, onDone, onStopped: onDone, onError: () => onDone?.()});
  } catch {
    onDone?.();
  }
}

export function stopSpeaking() {
  try { Speech.stop(); } catch { /* no speech engine */ }
}

const PRAISE = ['Great job!', 'You did it!', 'Yay! Well done!', 'Super!', 'Wonderful!'];
export const praise = (n: number) => PRAISE[n % PRAISE.length];
/** Retry language is the same every time: consistent, calm, never "wrong". */
export const RETRY = "Let's try again.";
export const HINT = "Let's try together. Here is a little help.";
export const NICE_TRY = 'Nice trying! Let’s play something new.';
