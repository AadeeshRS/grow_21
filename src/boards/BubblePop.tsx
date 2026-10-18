import React, {useEffect, useRef, useState} from 'react';
import {Pressable, StyleSheet, Text, View} from 'react-native';
import {DEFAULTS} from '../config';
import {colors} from '../theme';
import type {GameConfig} from '../types';
import {makeRng, num, type Rng} from './random';
import type {BoardProps} from './types';

export interface Bubble { id: number; x: number; y: number; hue: string }
const HUES = ['#7CC8FF', '#9BE7C4', '#FFB3D1', '#C9B6FF', '#FFD98A'];

export interface BubbleSettings { size: number; speed: number; concurrent: number; separation: number; target: number; hitPadding: number }

export function bubbleSettings(config: GameConfig, width: number, height: number, scale = 1): BubbleSettings {
  const size = Math.max(72, Math.min(num(config.bubbleSizeDp, 110) * scale, Math.min(width, height) * 0.34));
  return {
    size,
    speed: num(config.speedDpPerSec, 40) * scale,
    concurrent: Math.max(1, Math.min(5, Math.round(num(config.concurrentBubbles, 1)))),
    separation: num(config.minSeparationDp, 120) * scale,
    target: Math.max(1, Math.round(num(config.popTarget, 5))),
    hitPadding: num(config.hitPaddingDp, 8) * scale,
  };
}

function spawn(rng: Rng, id: number, s: BubbleSettings, width: number, height: number, others: Bubble[], startY?: number): Bubble {
  let x = s.size / 2 + rng() * Math.max(1, width - s.size);
  for (let tries = 0; tries < 12 && others.some(o => Math.abs(o.x - x) < Math.min(s.separation, width / s.concurrent)); tries++) {
    x = s.size / 2 + rng() * Math.max(1, width - s.size);
  }
  return {id, x, y: startY ?? height + s.size / 2 + rng() * s.size, hue: HUES[id % HUES.length]};
}

export function BubblePopBoard({config, seed, width, height, scale, reducedMotion, onInteract, onAttemptEnd}: BoardProps) {
  const s = bubbleSettings(config, width, height, scale);
  const rng = useRef(makeRng(seed)).current;
  const nextId = useRef(0);
  const [bubbles, setBubbles] = useState<Bubble[]>(() => {
    const list: Bubble[] = [];
    // First bubbles start inside the board so there is something to pop straight away.
    for (let i = 0; i < s.concurrent; i++) list.push(spawn(rng, nextId.current++, s, width, height, list, height * (0.55 + 0.3 * (i / Math.max(1, s.concurrent)))));
    return list;
  });
  const [popped, setPopped] = useState(0);
  const poppedRef = useRef(0);
  const [pops, setPops] = useState<Array<{id: number; x: number; y: number}>>([]);
  const stats = useRef({misses: 0, consecutiveMisses: 0, maxConsecutiveMisses: 0, missDistanceSum: 0, escaped: 0});
  const done = useRef(false);
  const speed = reducedMotion ? s.speed * 0.6 : s.speed;

  useEffect(() => {
    let last = performance.now();
    let raf = 0;
    const step = (t: number) => {
      const dt = Math.min(0.05, (t - last) / 1000);
      last = t;
      setBubbles(list => list.map(b => {
        const y = b.y - speed * dt;
        if (y < -s.size / 2) { stats.current.escaped++; return spawn(rng, nextId.current++, s, width, height, list); }
        return {...b, y};
      }));
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [speed, width, height]);

  const report = (success: boolean, count: number) => {
    done.current = true;
    const st = stats.current;
    setTimeout(() => onAttemptEnd({success, motor: {popped: count, misses: st.misses, maxConsecutiveMisses: st.maxConsecutiveMisses, meanMissDistanceDp: st.misses ? Math.round(st.missDistanceSum / st.misses) : null, escaped: st.escaped, bubbleSizeDp: Math.round(s.size)}}), success ? 450 : 700);
  };

  const pop = (b: Bubble) => {
    if (done.current) return;
    onInteract();
    stats.current.consecutiveMisses = 0;
    setPops(p => [...p.slice(-4), {id: b.id, x: b.x, y: b.y}]);
    setTimeout(() => setPops(p => p.filter(x => x.id !== b.id)), 650);
    // The replacement rises from just inside the bottom edge so the sky is never empty.
    setBubbles(list => list.filter(x => x.id !== b.id).concat(spawn(rng, nextId.current++, s, width, height, list, height - s.size * 0.55)));
    const count = ++poppedRef.current;
    setPopped(count);
    if (count >= s.target) report(true, count);
  };

  const miss = (x: number, y: number) => {
    if (done.current) return;
    onInteract();
    const st = stats.current;
    st.misses++;
    st.consecutiveMisses++;
    st.maxConsecutiveMisses = Math.max(st.maxConsecutiveMisses, st.consecutiveMisses);
    st.missDistanceSum += Math.min(...bubbles.map(b => Math.hypot(b.x - x, b.y - y)), 9999);
    if (st.consecutiveMisses >= DEFAULTS.attemptFailure.consecutiveMissLimit) report(false, poppedRef.current);
  };

  const hintId = config.hint ? bubbles.filter(b => b.y < height - s.size / 2).sort((a, b) => a.y - b.y)[0]?.id : undefined;
  return (
    <View style={[styles.sky, {width, height}]}>
      <Pressable testID="bubble-sky" accessibilityLabel="sky" style={StyleSheet.absoluteFill} onPress={e => miss(e.nativeEvent.locationX, e.nativeEvent.locationY)} />
      <View pointerEvents="none" style={styles.counter}>
        {Array.from({length: s.target}, (_, i) => <View key={i} style={[styles.pip, i < popped ? styles.pipOn : null]} />)}
      </View>
      {pops.map(p => (
        <View key={`pop-${p.id}`} pointerEvents="none" style={[styles.pop, {left: p.x - 40, top: p.y - 40}]}><Text style={{fontSize: 52}}>✨</Text></View>
      ))}
      {bubbles.map(b => (
        <Pressable key={b.id} testID={`bubble-${b.id}`} accessibilityRole="button" accessibilityLabel="bubble" onPress={() => pop(b)} hitSlop={s.hitPadding}
          style={[styles.bubble, {left: b.x - s.size / 2, top: b.y - s.size / 2, width: s.size, height: s.size, borderRadius: s.size / 2, backgroundColor: b.hue + '80', borderColor: b.hue},
            hintId === b.id ? styles.hint : null]}>
          <View style={[styles.shine, {width: s.size * 0.26, height: s.size * 0.18, borderRadius: s.size, top: s.size * 0.16, left: s.size * 0.2}]} />
          {hintId === b.id ? <Text style={styles.hand}>👆</Text> : null}
        </Pressable>
      ))}
    </View>
  );
}

export const bubbleInstruction = (config: GameConfig) => {
  const n = Math.max(1, Math.round(num(config.popTarget, 5)));
  return {text: `Pop ${n} bubbles`, spoken: `Tap the bubbles to pop them! Pop ${n} bubbles.`};
};

const styles = StyleSheet.create({
  sky: {overflow: 'hidden', borderRadius: 20, backgroundColor: '#EAF6FF'},
  counter: {position: 'absolute', top: 12, alignSelf: 'center', flexDirection: 'row', gap: 12, zIndex: 3, backgroundColor: 'rgba(255,255,255,0.85)', borderRadius: 999, paddingHorizontal: 14, paddingVertical: 8},
  pip: {width: 32, height: 32, borderRadius: 16, borderWidth: 4, borderColor: colors.motor, backgroundColor: colors.paper},
  pipOn: {backgroundColor: colors.motor},
  bubble: {position: 'absolute', borderWidth: 4, alignItems: 'center', justifyContent: 'center'},
  shine: {position: 'absolute', backgroundColor: 'rgba(255,255,255,0.85)', transform: [{rotate: '-30deg'}]},
  hint: {borderColor: colors.hint, borderWidth: 6},
  hand: {fontSize: 34, position: 'absolute', bottom: -30},
  pop: {position: 'absolute', width: 80, height: 80, alignItems: 'center', justifyContent: 'center'},
});
