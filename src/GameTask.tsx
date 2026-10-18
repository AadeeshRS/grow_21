import React, {useEffect, useMemo, useRef, useState} from 'react';
import {AccessibilityInfo, StyleSheet, Text, useWindowDimensions, View} from 'react-native';
import {BOARDS, instructionFor, seedFromString, type AttemptResult} from './boards';
import {DEFAULTS} from './config';
import {applyScaffold, GAME_NAMES} from './games';
import {ScaffoldController} from './ScaffoldController';
import {HINT, NICE_TRY, praise, RETRY, say, stopSpeaking} from './speech';
import {colors, domainColors, fonts, GAME_DOMAIN, GAME_ICON, radius, shadow} from './theme';
import type {AttemptRecord, GameConfig, GameId, TelemetryEvent} from './types';
import {Celebration, Mascot, Pulse, SpeakerButton} from './ui/components';

const now = () => performance.now();
const uid = () => `${Date.now()}-${Math.random().toString(36).slice(2)}`;

export function useReducedMotion() {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    let alive = true;
    AccessibilityInfo.isReduceMotionEnabled?.().then(v => { if (alive) setReduced(!!v); }).catch(() => {});
    const sub = AccessibilityInfo.addEventListener?.('reduceMotionChanged', v => setReduced(!!v));
    return () => { alive = false; sub?.remove?.(); };
  }, []);
  return reduced;
}

type Banner = null | {kind: 'success' | 'retry' | 'hint' | 'done'; text: string};

export interface GameTaskProps {
  gameId: GameId;
  skillId: TelemetryEvent['skillId'];
  taskId?: string;
  taskSeed?: number;
  difficulty: number;
  config: GameConfig;
  startLevel?: number;
  childId: string;
  sessionId: string;
  index: number;
  onComplete: (e: TelemetryEvent) => void;
}

/**
 * Runs one task: speaks the instruction, renders the game's board, turns each board attempt into an
 * AttemptRecord, escalates scaffolding through ScaffoldController, and emits one TelemetryEvent.
 */
export function GameTask({gameId, skillId, taskId, taskSeed, difficulty, config, startLevel = 0, childId, sessionId, index, onComplete}: GameTaskProps) {
  const controller = useRef(new ScaffoldController(startLevel));
  const presented = useRef(now());
  const attemptStart = useRef(now());
  const attemptFirst = useRef<number | undefined>(undefined);
  const taskFirst = useRef<number | undefined>(undefined);
  const lastActivity = useRef(now());
  const nudges = useRef(0);
  const finished = useRef(false);
  const [level, setLevel] = useState(controller.current.level);
  const [attempt, setAttempt] = useState(0);
  const [banner, setBanner] = useState<Banner>(null);
  const [size, setSize] = useState<{width: number; height: number} | null>(null);
  const [nudge, setNudge] = useState(false);
  const reducedMotion = useReducedMotion();
  const window = useWindowDimensions();
  const compact = window.width < 600 || window.height < 500;
  const seed = useMemo(() => taskSeed ?? seedFromString(taskId ?? `${gameId}-${index}`), [taskSeed, taskId, gameId, index]);
  const display = useMemo(() => applyScaffold(gameId, config, level), [gameId, config, level]);
  const instruction = useMemo(() => instructionFor(gameId, display, seed), [gameId, display, seed]);
  const spoken = display.repeatInstruction || display.instructionRepeats ? `${instruction.spoken} ${instruction.spoken}` : instruction.spoken;
  const domain = domainColors(GAME_DOMAIN[gameId]);
  const Board = BOARDS[gameId];

  // Speak the instruction at the start of every attempt (echo speaks its own model sounds).
  useEffect(() => {
    if (gameId !== 'echo' || attempt === 0) say(spoken);
    attemptStart.current = now();
    attemptFirst.current = undefined;
    lastActivity.current = now();
    nudges.current = 0;
  }, [attempt]);
  useEffect(() => () => stopSpeaking(), []);

  // Inactivity: repeat the instruction and pulse the speaker, at most twice per attempt. Never a failure.
  useEffect(() => {
    const id = setInterval(() => {
      if (banner || finished.current) return;
      if (nudges.current < 2 && now() - lastActivity.current > DEFAULTS.scaffold.inactivityMs) {
        lastActivity.current = now();
        nudges.current++;
        setNudge(true);
        if (gameId !== 'echo') say(spoken);
        setTimeout(() => setNudge(false), 2600);
      }
    }, 1000);
    return () => clearInterval(id);
  }, [banner, spoken, gameId]);

  const interact = () => {
    const t = now();
    lastActivity.current = t;
    if (attemptFirst.current === undefined) attemptFirst.current = t;
    if (taskFirst.current === undefined) taskFirst.current = t;
  };

  const emit = (outcome: 'success' | 'failure') => {
    if (finished.current) return;
    finished.current = true;
    const t = now();
    const c = controller.current;
    const first = taskFirst.current ?? t;
    onComplete({
      schemaVersion: 1, eventId: uid(), childId, sessionId, gameId, skillId, taskId: taskId ?? uid(), taskIndexInSession: index,
      isBaseline: false, timestampIso: new Date().toISOString(), tPresentedMs: presented.current, tFirstResponseMs: first, tCompletedMs: t,
      outcome, latencyMs: first - presented.current, durationMs: t - presented.current, retries: c.attempts.length - 1, difficulty,
      scaffoldStartLevel: startLevel, scaffoldFinalLevel: c.level, assisted: c.level > startLevel, scaffoldExhausted: c.exhausted,
      attempts: c.attempts, gameConfig: display, task: {gameName: GAME_NAMES[gameId]},
    });
  };

  const onAttemptEnd = (result: AttemptResult) => {
    if (finished.current || banner) return;
    const t = now();
    const record: AttemptRecord = {
      attemptIndex: controller.current.attempts.length, scaffoldLevel: controller.current.level,
      outcome: result.success ? 'success' : 'failure', latencyMs: (attemptFirst.current ?? t) - attemptStart.current, durationMs: t - attemptStart.current,
      ...(result.motor ? {motor: result.motor} : {}), ...(result.speech ? {speech: result.speech} : {}), ...(result.task ? {task: result.task} : {}),
    };
    const action = controller.current.onAttemptEnd(record);
    if (result.success) {
      const text = praise(index);
      setBanner({kind: 'success', text});
      say(text);
      setTimeout(() => emit('success'), 1500);
    } else if (action.action === 'escalate') {
      setBanner({kind: 'hint', text: HINT});
      say(HINT);
      setTimeout(() => { setLevel(action.newLevel); setAttempt(a => a + 1); setBanner(null); }, 1700);
    } else if (action.action === 'continue') {
      setBanner({kind: 'retry', text: RETRY});
      say(RETRY);
      setTimeout(() => { setAttempt(a => a + 1); setBanner(null); }, 1300);
    } else {
      setBanner({kind: 'done', text: NICE_TRY});
      say(NICE_TRY);
      setTimeout(() => emit('failure'), 1900);
    }
  };

  return (
    <View style={styles.root}>
      <View style={styles.header}>
        {compact ? null : <View style={[styles.badge, {backgroundColor: domain.soft, borderColor: domain.main}]}><Text style={styles.badgeIcon}>{GAME_ICON[gameId]}</Text></View>}
        <View style={[styles.instruction, shadow(4), compact ? {minHeight: 52, paddingVertical: 6} : null]}>
          <Text style={[styles.instructionText, compact ? {fontSize: 21} : null]} numberOfLines={2} accessibilityRole="header">{instruction.text}</Text>
          {level > startLevel ? <Text style={styles.helpChip}>💡 Help is on</Text> : null}
        </View>
        <Pulse active={nudge} reducedMotion={reducedMotion}><SpeakerButton size={compact ? 52 : 64} onPress={() => { lastActivity.current = now(); say(gameId === 'echo' ? instruction.spoken : spoken); }} /></Pulse>
      </View>
      <View testID="game-stage" style={[styles.stage, shadow(6), {borderColor: domain.soft}]}
        onLayout={e => { const {width, height} = e.nativeEvent.layout; if (!size || Math.abs(size.width - width) > 1 || Math.abs(size.height - height) > 1) setSize({width, height}); }}>
        {size && size.width > 0 && size.height > 0 ? (
          <Board key={`${attempt}:${Math.round(size.width)}x${Math.round(size.height)}`} config={display} level={level} seed={seed} width={size.width - 24} height={size.height - 24}
            scale={Math.max(1, Math.min(1.6, Math.min(size.width, size.height) / 520))}
            reducedMotion={reducedMotion} onInteract={interact} onAttemptEnd={onAttemptEnd} />
        ) : null}
        {banner?.kind === 'success' ? <Celebration text={banner.text} reducedMotion={reducedMotion} /> : null}
        {banner && banner.kind !== 'success' ? (
          <View style={styles.overlay}>
            <View style={[styles.note, shadow(10)]}>
              <Mascot size={110} mood={banner.kind === 'hint' ? 'cheer' : 'happy'} />
              <Text style={styles.noteText}>{banner.text}</Text>
            </View>
          </View>
        ) : null}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: {flex: 1, gap: 12},
  header: {flexDirection: 'row', alignItems: 'center', gap: 14},
  badge: {width: 64, height: 64, borderRadius: 20, alignItems: 'center', justifyContent: 'center', borderWidth: 3},
  badgeIcon: {fontSize: 34},
  instruction: {flex: 1, minHeight: 64, backgroundColor: colors.paper, borderRadius: radius.md, paddingHorizontal: 20, paddingVertical: 10, flexDirection: 'row', alignItems: 'center', gap: 12},
  instructionText: {flex: 1, fontFamily: fonts.display, fontSize: 26, color: colors.ink},
  helpChip: {fontFamily: fonts.strong, fontSize: 15, color: colors.ink, backgroundColor: colors.sunSoft, paddingHorizontal: 12, paddingVertical: 6, borderRadius: radius.pill, overflow: 'hidden'},
  stage: {flex: 1, backgroundColor: colors.paper, borderRadius: radius.lg, padding: 12, borderWidth: 4, overflow: 'hidden'},
  overlay: {...StyleSheet.absoluteFillObject, backgroundColor: 'rgba(234,244,255,0.75)', alignItems: 'center', justifyContent: 'center', zIndex: 20},
  note: {backgroundColor: colors.paper, borderRadius: radius.lg, paddingHorizontal: 36, paddingVertical: 22, alignItems: 'center', maxWidth: 480},
  noteText: {fontFamily: fonts.display, fontSize: 28, color: colors.ink, textAlign: 'center', marginTop: 6},
});
