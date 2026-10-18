import type {GameConfig} from '../types';

/** What a board reports when one attempt ends. Detail fields go into the attempt's telemetry. */
export interface AttemptResult {
  success: boolean;
  motor?: Record<string, unknown>;
  speech?: Record<string, unknown>;
  task?: Record<string, unknown>;
}

export interface BoardProps {
  /** Difficulty-mapped and scaffolded parameters for this game (games.ts). */
  config: GameConfig;
  /** Current scaffold level 0–3 (more help at higher levels). */
  level: number;
  /** Content seed, stable for the whole task so retries show the same items. */
  seed: number;
  width: number;
  height: number;
  /**
   * Screen scale ≥ 1 applied to dp-based difficulty sizes on large screens, so relative difficulty
   * is kept while using the space (sizes still shrink as difficulty rises).
   */
  scale: number;
  reducedMotion: boolean;
  /** Call on the child's first touch of the attempt (response latency). */
  onInteract(): void;
  onAttemptEnd(result: AttemptResult): void;
}

/** Child-facing instruction: a short written cue plus the sentence that is spoken. */
export interface Instruction {
  text: string;
  spoken: string;
}
