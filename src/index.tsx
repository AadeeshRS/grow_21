import React, {useEffect, useMemo, useState} from 'react';
import {StyleSheet, Text, useWindowDimensions, View} from 'react-native';
import {GameTask} from './GameTask';
import {GAME_NAMES} from './games';
import {MockEngine} from './MockEngine';
import {MockHostBridge} from './MockHostBridge';
import {SessionRunner} from './SessionRunner';
import {configureSpeech} from './speech';
import {colors, fonts, radius, shadow} from './theme';
import type {EngineClient, HostBridge, TaskDirective, TelemetryEvent} from './types';
import {HoldButton, Mascot} from './ui/components';

export interface Grow21ModuleProps {
  childId: string;
  sessionId: string;
  locale: string;
  mode: 'baseline' | 'play';
  onExit: (reason: 'completed' | 'break' | 'userExit' | 'error') => void;
  engine?: EngineClient;
  host?: HostBridge;
  /** Optional: called after each task's telemetry is handed to the engine (e.g. for a host progress display). */
  onTaskComplete?: (event: TelemetryEvent) => void;
}

/** Integration root. Real apps pass bridge-backed clients; mock dependencies are development-only. */
export function Grow21GameModule({childId, sessionId, locale, mode, onExit, engine, host, onTaskComplete}: Grow21ModuleProps) {
  const hostBridge = useMemo(() => host ?? new MockHostBridge(), [host]);
  const runner = useMemo(() => new SessionRunner(engine ?? new MockEngine(), hostBridge, childId, sessionId, mode), [engine, hostBridge, childId, sessionId, mode]);
  const [directive, setDirective] = useState<TaskDirective>();
  const [taskSeq, setTaskSeq] = useState(0);
  const [stars, setStars] = useState(0);
  const [error, setError] = useState(false);
  const {width, height} = useWindowDimensions();
  const compact = width < 600;
  // Landscape phones: move stars and the lock into side rails so the play area keeps its height.
  const rails = height < 500 && width > height;

  useEffect(() => { configureSpeech({language: locale}); }, [locale]);

  const load = async () => {
    try {
      const next = await runner.next();
      if (next.endSession) return onExit('completed');
      if (next.break) {
        await hostBridge.notifyBreak(next.break);
        return onExit('break');
      }
      setDirective(next);
      setTaskSeq(n => n + 1);
    } catch {
      setError(true);
      onExit('error');
    }
  };
  useEffect(() => { void load(); }, []);

  const task = directive?.task;
  const starBadge = (
    <View style={[styles.stars, shadow(3)]} accessibilityLabel={`${stars} stars`}>
      <Text style={styles.starIcon}>⭐</Text>
      <Text testID="star-count" style={styles.starCount}>{stars}</Text>
    </View>
  );
  const lock = <HoldButton testID="grown-ups-exit" onHold={() => onExit('userExit')} label="Grown-ups: hold to leave" />;
  return (
    <View style={styles.root}>
      <View style={[styles.column, rails ? styles.railLayout : null, {padding: rails ? 8 : compact ? 12 : 22}]}>
        {rails ? <View style={styles.rail}>{starBadge}</View> : (
          <View style={styles.topBar}>
            {starBadge}
            <Text style={styles.gameName} numberOfLines={1}>{task ? GAME_NAMES[task.gameId] : ''}</Text>
            {lock}
          </View>
        )}
        <View style={styles.body}>
          {error ? (
            <View style={styles.center}>
              <Mascot size={150} mood="rest" />
              <Text style={styles.message}>Let's ask a grown-up for help.</Text>
              <Text style={styles.subtle}>The game could not reach the learning service.</Text>
            </View>
          ) : !task ? (
            <View style={styles.center} accessibilityLabel="Getting the next activity…">
              <Mascot size={150} mood="happy" />
              <Text style={styles.message}>Get ready…</Text>
            </View>
          ) : (
            <GameTask key={taskSeq} gameId={task.gameId} skillId={task.skillId} taskId={task.taskId} taskSeed={task.taskSeed}
              difficulty={Math.max(0, Math.min(1, task.D))} config={runner.config(task)} startLevel={task.startScaffoldLevel}
              childId={childId} sessionId={sessionId} index={runner.tasksCompleted}
              onComplete={async (e: TelemetryEvent) => {
                setDirective(undefined);
                if (e.outcome === 'success') setStars(s => s + 1);
                await runner.submit(e);
                onTaskComplete?.(e);
                void load();
              }} />
          )}
        </View>
        {rails ? <View style={styles.rail}>{lock}</View> : null}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: {flex: 1, backgroundColor: colors.sky, alignItems: 'center'},
  column: {flex: 1, width: '100%', maxWidth: 1240, gap: 14},
  topBar: {flexDirection: 'row', alignItems: 'center', gap: 14, minHeight: 56},
  railLayout: {flexDirection: 'row', gap: 10},
  rail: {alignItems: 'center', paddingTop: 2},
  stars: {flexDirection: 'row', alignItems: 'center', gap: 6, backgroundColor: colors.paper, borderRadius: radius.pill, paddingHorizontal: 16, height: 52},
  starIcon: {fontSize: 26},
  starCount: {fontFamily: fonts.display, fontSize: 26, color: colors.ink, minWidth: 22},
  gameName: {flex: 1, fontFamily: fonts.strong, fontSize: 20, color: colors.inkSoft},
  body: {flex: 1},
  center: {flex: 1, alignItems: 'center', justifyContent: 'center', gap: 14, padding: 24},
  message: {fontFamily: fonts.display, fontSize: 32, color: colors.ink, textAlign: 'center'},
  subtle: {fontFamily: fonts.body, fontSize: 17, color: colors.inkSoft, textAlign: 'center'},
});

export * from './types'; export * from './games'; export * from './ScaffoldController'; export * from './MockEngine'; export * from './MockHostBridge'; export * from './api/HttpEngineClient';
export {Mascot, BigButton, HoldButton} from './ui/components';
export {colors as theme, fonts} from './theme';
export {say} from './speech';
