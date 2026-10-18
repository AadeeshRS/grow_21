import type {GameId} from './types';

/**
 * Visual language for the Grow21 module.
 * Calm, high-contrast, low-clutter: a light sky background, white play surfaces, one accent per
 * game domain, and no red "error" colour anywhere in child-facing UI (errors are never punished).
 */
export const colors = {
  sky: '#EAF4FF',
  skyDeep: '#D4E8FF',
  paper: '#FFFFFF',
  ink: '#1D2B4F',
  inkSoft: '#4B5B7E',
  inkFaint: '#8C9AB8',
  line: '#D6E3F3',
  success: '#22A565',
  successSoft: '#DDF6E8',
  sun: '#FFC531',
  sunSoft: '#FFF3CC',
  hint: '#FF9F1C',
  motor: '#16A99A',
  motorSoft: '#D6F5F0',
  verbal: '#F28C28',
  verbalSoft: '#FFE9D2',
  cognitive: '#7A5CF0',
  cognitiveSoft: '#ECE6FF',
  shadow: '#1D2B4F',
} as const;

/** Nunito is loaded by the host (see demo/App.tsx); Nunito's rounded forms read well for early readers. */
export const fonts = {
  body: 'Nunito_700Bold',
  strong: 'Nunito_800ExtraBold',
  display: 'Nunito_900Black',
} as const;

export const radius = {sm: 14, md: 22, lg: 32, pill: 999} as const;

/** Minimum child touch target (dp). Children miss small targets far more often than adults. */
export const MIN_TARGET = 72;

export type Domain = 'motor' | 'verbal' | 'cognitive';

export const GAME_DOMAIN: Record<GameId, Domain> = {
  trace: 'motor', bubble: 'motor', block: 'motor',
  echo: 'verbal', sentence: 'verbal', audio: 'verbal',
  sort: 'cognitive', flip: 'cognitive', pattern: 'cognitive',
};

export const GAME_ICON: Record<GameId, string> = {
  trace: '🖍️', bubble: '🫧', block: '🧱', echo: '🦁', sentence: '💬', audio: '👂', sort: '🧺', flip: '🃏', pattern: '🚂',
};

export const domainColors = (d: Domain) =>
  d === 'motor' ? {main: colors.motor, soft: colors.motorSoft}
  : d === 'verbal' ? {main: colors.verbal, soft: colors.verbalSoft}
  : {main: colors.cognitive, soft: colors.cognitiveSoft};

export const shadow = (elevation = 6) => ({
  shadowColor: colors.shadow,
  shadowOpacity: 0.12,
  shadowRadius: elevation * 1.6,
  shadowOffset: {width: 0, height: Math.round(elevation / 2)},
  elevation,
});
