import React, {useEffect, useRef, useState} from 'react';
import {Animated, Pressable, StyleSheet, Text, View} from 'react-native';
import {DEFAULTS} from '../config';
import {say} from '../speech';
import {colors, fonts, radius, shadow} from '../theme';
import type {GameConfig} from '../types';
import {NATIVE_DRIVER, Pulse} from '../ui/components';
import {LOOKALIKE_GROUPS, type Item} from './content';
import {makeRng, num, pick, sample, shuffle} from './random';
import type {BoardProps} from './types';

export interface FlipRound { cards: Item[]; pairs: number }

export function buildFlipRound(config: GameConfig, seed: number): FlipRound {
  const rng = makeRng(seed);
  const cardCount = num(config.cardCount, 4) >= 6 ? 6 : 4;
  const pairs = cardCount / 2;
  const similar = num(config.visualSimilarity) >= 0.3;
  const groups = shuffle(rng, LOOKALIKE_GROUPS);
  const faces = similar ? sample(rng, groups[0], pairs) : groups.slice(0, pairs).map(g => pick(rng, g));
  return {cards: shuffle(rng, [...faces, ...faces]), pairs};
}

/** Grid that gives the largest cards for the available space. */
function grid(n: number, width: number, height: number) {
  let best = {cols: 2, size: 0};
  for (const cols of [2, 3, 4, 6]) {
    if (n % cols) continue;
    const rows = n / cols;
    const size = Math.min((width - 24 * (cols + 1)) / cols, (height - 24 * (rows + 1)) / rows);
    if (size > best.size) best = {cols, size};
  }
  return {cols: best.cols, size: Math.min(200, best.size)};
}

function Card({item, faceUp, matched, hint, size, onPress, testID, reducedMotion}: {item: Item; faceUp: boolean; matched: boolean; hint: boolean; size: number; onPress: () => void; testID: string; reducedMotion: boolean}) {
  const turn = useRef(new Animated.Value(1)).current;
  const first = useRef(true);
  useEffect(() => {
    if (first.current) { first.current = false; return; }
    turn.setValue(reducedMotion ? 1 : 0.05);
    Animated.spring(turn, {toValue: 1, speed: 24, bounciness: 6, useNativeDriver: NATIVE_DRIVER}).start();
  }, [faceUp, turn, reducedMotion]);
  return (
    <Pulse active={hint} reducedMotion={reducedMotion}>
      <Pressable testID={testID} accessibilityRole="button" accessibilityLabel={faceUp ? item.word : 'hidden card'} onPress={onPress} disabled={faceUp}>
        <Animated.View style={[styles.card, shadow(matched ? 2 : 6), {width: size, height: size, backgroundColor: faceUp ? colors.paper : colors.cognitive, transform: [{scaleX: turn}]}, matched ? styles.matched : null, hint ? styles.hint : null]}>
          {faceUp
            ? <Text style={{fontSize: size * 0.5}} allowFontScaling={false}>{item.emoji}</Text>
            : <Text style={[styles.back, {fontSize: size * 0.42}]}>?</Text>}
        </Animated.View>
      </Pressable>
    </Pulse>
  );
}

export function FlipMatchBoard({config, seed, width, height, reducedMotion, onInteract, onAttemptEnd}: BoardProps) {
  const round = useRef(buildFlipRound(config, seed)).current;
  const exposure = num(config.exposureMs, 3000);
  const [previewing, setPreviewing] = useState(true);
  const [open, setOpen] = useState<number[]>([]);
  const [matched, setMatched] = useState<number[]>([]);
  const mismatches = useRef(0);
  const flips = useRef(0);
  const done = useRef(false);
  const demoPair = config.demo ? round.cards.map((c, i) => i).filter(i => round.cards[i] === round.cards[0]) : [];

  useEffect(() => {
    const t = setTimeout(() => { setPreviewing(false); say('Now find the pairs!'); }, exposure);
    return () => clearTimeout(t);
  }, [exposure]);

  const tap = (i: number) => {
    if (previewing || done.current || open.includes(i) || matched.includes(i) || open.length >= 2) return;
    onInteract();
    flips.current++;
    const next = [...open, i];
    setOpen(next);
    if (next.length < 2) return;
    const [a, b] = next;
    const task = () => ({mismatches: mismatches.current, flips: flips.current, pairs: round.pairs, previewMs: exposure});
    if (round.cards[a] === round.cards[b]) {
      const all = [...matched, a, b];
      setTimeout(() => {
        setMatched(all);
        setOpen([]);
        if (all.length === round.cards.length) { done.current = true; onAttemptEnd({success: true, task: task()}); }
      }, 350);
    } else {
      mismatches.current++;
      const fail = mismatches.current >= DEFAULTS.attemptFailure.incorrectFlipLimit;
      if (fail) done.current = true;
      setTimeout(() => {
        setOpen([]);
        if (fail) onAttemptEnd({success: false, task: task()});
      }, 900);
    }
  };

  const {cols, size} = grid(round.cards.length, width, height - 70);
  return (
    <View style={[styles.root, {width, height}]}>
      <View style={[styles.phase, previewing ? styles.phaseLook : styles.phaseFind]}>
        <Text style={styles.phaseText}>{previewing ? '👀 Look carefully!' : '👆 Find the pairs'}</Text>
      </View>
      <View style={[styles.grid, {width: cols * (size + 24)}]}>
        {round.cards.map((item, i) => (
          <View key={i} style={{margin: 12}}>
            <Card testID={`flip-card-${i}`} item={item} size={size} reducedMotion={reducedMotion}
              faceUp={previewing || open.includes(i) || matched.includes(i)} matched={matched.includes(i)}
              hint={previewing && demoPair.includes(i)} onPress={() => tap(i)} />
          </View>
        ))}
      </View>
    </View>
  );
}

export const flipInstruction = () => ({text: 'Find the pairs', spoken: 'Look at the pictures. Then find the pairs that match.'});

const styles = StyleSheet.create({
  root: {alignItems: 'center', justifyContent: 'center', gap: 8},
  phase: {borderRadius: radius.pill, paddingHorizontal: 22, paddingVertical: 8},
  phaseLook: {backgroundColor: colors.sunSoft},
  phaseFind: {backgroundColor: colors.cognitiveSoft},
  phaseText: {fontFamily: fonts.display, fontSize: 24, color: colors.ink},
  grid: {flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'center'},
  card: {borderRadius: radius.md, alignItems: 'center', justifyContent: 'center'},
  back: {color: '#fff', fontFamily: fonts.display},
  matched: {opacity: 0.85, borderWidth: 4, borderColor: colors.success, borderRadius: radius.md},
  hint: {borderWidth: 5, borderColor: colors.hint, borderRadius: radius.md},
});
