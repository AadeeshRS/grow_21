import {Platform} from 'react-native';

/**
 * Voice-activity detection for Echo Safari. The baseline accepts ANY detected vocalisation
 * (no transcription, no pronunciation scoring). Only implemented for the web host (Web Audio).
 * Elsewhere `createVoiceDetector` returns null and the game falls back to an adult observer.
 */
export interface VoiceDetector {
  /** Calls onVoice once when sustained vocal energy is detected; returns a stop function. */
  listen(onVoice: (info: {peakRms: number; noiseFloor: number}) => void): () => void;
  close(): void;
}

/** Pure decision rule (unit-tested): voice = `minFrames` consecutive frames above max(minRms, k × noise floor). */
export function isVoice(recentRms: number[], noiseFloor: number, {minRms = 0.02, factor = 3, minFrames = 3} = {}) {
  if (recentRms.length < minFrames) return false;
  const threshold = Math.max(minRms, noiseFloor * factor);
  return recentRms.slice(-minFrames).every(v => v > threshold);
}

/** ~600 ms of room noise before we start listening for a voice. */
export const CALIBRATION_FRAMES = 12;
/** A measured floor above this is almost certainly the app's own audio or a knock,
 *  not the room, and would push the threshold out of a child's reach. */
export const MAX_NOISE_FLOOR = 0.03;

/**
 * Room-noise level from the calibration frames.
 *
 * The median, not the maximum: taking the maximum meant a single cough, chair
 * creak or the tail of the app's own voice set the floor high, and since the
 * threshold is three times the floor, the child could then shout and never be
 * heard. That was why Echo Safari so often missed a real answer.
 */
export function noiseFloorFrom(samples: number[]): number {
  if (!samples.length) return 0.005;
  const sorted = [...samples].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];
  return Math.min(MAX_NOISE_FLOOR, Math.max(0.005, median));
}

export const rms = (samples: ArrayLike<number>) => {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  return Math.sqrt(sum / Math.max(1, samples.length));
};

export async function createVoiceDetector(): Promise<VoiceDetector | null> {
  if (Platform.OS !== 'web') return null;
  const nav = (globalThis as {navigator?: Navigator}).navigator;
  const AudioCtx = (globalThis as {AudioContext?: typeof AudioContext}).AudioContext;
  if (!nav?.mediaDevices?.getUserMedia || !AudioCtx) return null;
  let stream: MediaStream;
  try {
    stream = await nav.mediaDevices.getUserMedia({audio: {echoCancellation: true, noiseSuppression: true}});
  } catch {
    return null; // permission denied or no microphone
  }
  const ctx = new AudioCtx();
  void ctx.resume(); // contexts may start suspended until a user gesture (the Play button provides one)
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 1024;
  ctx.createMediaStreamSource(stream).connect(analyser);
  const buf = new Float32Array(analyser.fftSize);
  return {
    listen(onVoice) {
      const history: number[] = [];
      const calibration: number[] = [];
      let floor = 0.005;
      let frames = 0;
      let peak = 0;
      const id = setInterval(() => {
        analyser.getFloatTimeDomainData(buf);
        const v = rms(buf);
        frames++;
        if (frames <= CALIBRATION_FRAMES) { calibration.push(v); return; }
        if (frames === CALIBRATION_FRAMES + 1) floor = noiseFloorFrom(calibration);
        peak = Math.max(peak, v);        // only count peaks after the room has been measured
        history.push(v);
        if (history.length > 10) history.shift();
        if (isVoice(history, floor)) { clearInterval(id); onVoice({peakRms: peak, noiseFloor: floor}); }
      }, 50);
      return () => clearInterval(id);
    },
    close() {
      stream.getTracks().forEach(t => t.stop());
      void ctx.close();
    },
  };
}
