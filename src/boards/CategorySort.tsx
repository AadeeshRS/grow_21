import React, {useRef, useState} from 'react';
import {Animated, Pressable, StyleSheet, Text, View} from 'react-native';
import {DEFAULTS} from '../config';
import {say} from '../speech';
import {colors, fonts, radius, shadow} from '../theme';
import type {GameConfig} from '../types';
import {Pulse, useWiggle} from '../ui/components';
import {useDrag} from '../ui/drag';
import {SORT_PAIRS, type Item, type SortPair} from './content';
import {clampInt, makeRng, num, pick, sample, shuffle} from './random';
import type {BoardProps, Instruction} from './types';

export interface SortRound { pair: SortPair; items: Array<{item: Item; side: 0 | 1}>; instruction: Instruction }

export function buildSortRound(config: GameConfig, seed: number): SortRound {
  const rng = makeRng(seed);
  const sim = num(config.categorySimilarity);
  const level = num(config.attributeComplexity, 1) >= 2 ? 2 : sim < 0.2 ? 0 : sim < 0.45 ? 1 : 2;
  const pair = pick(rng, SORT_PAIRS.filter(p => p.similarity === level));
  const n = clampInt(num(config.itemCount, 3), 3, 8);
  const left = Math.ceil(n / 2);
  const items = shuffle(rng, [
    ...sample(rng, pair.left.items, left).map(item => ({item, side: 0 as const})),
    ...sample(rng, pair.right.items, n - left).map(item => ({item, side: 1 as const})),
  ]);
  return {pair, items, instruction: {text: `${pair.left.name} or ${pair.right.name}?`, spoken: `Put each picture in the right basket. ${pair.left.name}, or ${pair.right.name}?`}};
}

export interface SortLayout { item: {x: number; y: number; size: number}; bins: Array<{x: number; y: number; w: number; h: number}> }

/** Board geometry (shared with tests so drags can be aimed). */
export function sortLayout(width: number, height: number): SortLayout {
  const binW = Math.min(width * 0.42, 340);
  const binH = Math.min(height * 0.42, 250, binW * 1.05);
  const size = Math.min(height * 0.3, 170, width * 0.34);
  const gap = (width - 2 * binW) / 3;
  const binY = height - binH - 8;
  return {
    // The picture sits in the middle of the free space above the baskets.
    item: {x: width / 2 - size / 2, y: Math.max(8, (binY - size) / 2), size},
    bins: [{x: gap, y: binY, w: binW, h: binH}, {x: gap * 2 + binW, y: binY, w: binW, h: binH}],
  };
}

export function CategorySortBoard({config, seed, width, height, scale, reducedMotion, onInteract, onAttemptEnd}: BoardProps) {
  const round = useRef(buildSortRound(config, seed)).current;
  const layout = sortLayout(width, height);
  const [index, setIndex] = useState(0);
  const [sorted, setSorted] = useState<Array<{item: Item; side: 0 | 1}>>([]);
  const [drag, setDrag] = useState({dx: 0, dy: 0});
  const [flash, setFlash] = useState<0 | 1 | null>(null);
  const stats = useRef({classificationErrors: 0, placementMisses: 0, drags: 0, taps: 0});
  const done = useRef(false);
  const {style: wiggleStyle, wiggle} = useWiggle();
  const tolerance = num(config.placementToleranceDp, 40) * (config.snapToBin ? 2 : 1) * scale;
  const current = round.items[index];

  const place = (side: 0 | 1 | null) => {
    setDrag({dx: 0, dy: 0});
    if (done.current || !current) return;
    if (side === null) { stats.current.placementMisses++; return; }
    if (side === current.side) {
      const next = [...sorted, current];
      setSorted(next);
      setFlash(current.side);
      setTimeout(() => setFlash(f => (f === current.side ? null : f)), 600);
      say(current.item.word);
      if (index + 1 >= round.items.length) {
        done.current = true;
        setTimeout(() => onAttemptEnd({success: true, task: {...stats.current, items: round.items.length, pair: `${round.pair.left.name}/${round.pair.right.name}`}}), 500);
      }
      setIndex(index + 1);
    } else {
      stats.current.classificationErrors++;
      wiggle();
      if (stats.current.classificationErrors >= DEFAULTS.attemptFailure.dropFailLimit) {
        done.current = true;
        setTimeout(() => onAttemptEnd({success: false, task: {...stats.current, items: round.items.length, pair: `${round.pair.left.name}/${round.pair.right.name}`}}), 700);
      }
    }
  };

  const binAt = (cx: number, cy: number): 0 | 1 | null => {
    const hit = layout.bins.findIndex(b => cx >= b.x - tolerance && cx <= b.x + b.w + tolerance && cy >= b.y - tolerance && cy <= b.y + b.h + tolerance);
    return hit === -1 ? null : (hit as 0 | 1);
  };

  const handlers = useDrag({
    onStart: () => onInteract(),
    onMove: d => setDrag(d),
    onEnd: d => {
      if (Math.hypot(d.dx, d.dy) < 12) { setDrag({dx: 0, dy: 0}); return; } // a tap on the picture is not a drop
      stats.current.drags++;
      const s = layout.item.size;
      place(binAt(layout.item.x + s / 2 + d.dx, layout.item.y + s / 2 + d.dy));
    },
  });

  const tapBin = (side: 0 | 1) => {
    if (!current) return;
    onInteract();
    stats.current.taps++;
    place(side);
  };

  return (
    <View style={{width, height}}>
      {layout.bins.map((b, i) => {
        const cat = i === 0 ? round.pair.left : round.pair.right;
        const inBin = sorted.filter(s => s.side === i);
        const example = config.example ? (i === 0 ? round.pair.left.items : round.pair.right.items).find(it => !round.items.some(r => r.item === it)) : undefined;
        return (
          <Pulse key={i} active={!!config.highlightBin && current?.side === i} reducedMotion={reducedMotion} style={[styles.binWrap, {left: b.x, top: b.y, width: b.w, height: b.h}]}>
            <Pressable testID={`sort-bin-${i}`} accessibilityRole="button" accessibilityLabel={cat.name} onPress={() => tapBin(i as 0 | 1)}
              style={[styles.bin, shadow(6), config.highlightBin && current?.side === i ? styles.binHint : null, flash === i ? styles.binYes : null]}>
              <View style={styles.binHeader}>
                <Text style={{fontSize: Math.min(config.clearLabels ? 44 : 34, b.w / 5)}}>{cat.icon}</Text>
                <Text numberOfLines={1} adjustsFontSizeToFit style={[styles.binName, {fontSize: Math.min(config.clearLabels ? 30 : 24, b.w / 7)}]}>{cat.name}</Text>
              </View>
              <View style={styles.binItems}>
                {example ? <Text style={[styles.binItem, {opacity: 0.45}]}>{example.emoji}</Text> : null}
                {inBin.map((s, k) => <Text key={k} style={styles.binItem}>{s.item.emoji}</Text>)}
              </View>
            </Pressable>
          </Pulse>
        );
      })}
      {current ? (
        <Animated.View key={index} {...handlers} testID="sort-item" accessibilityLabel={current.item.word}
          style={[styles.item, shadow(drag.dx || drag.dy ? 14 : 8), {left: layout.item.x, top: layout.item.y, width: layout.item.size, height: layout.item.size},
            {transform: [...wiggleStyle.transform, {translateX: drag.dx}, {translateY: drag.dy}]}]}>
          <Text style={{fontSize: layout.item.size * 0.55}} allowFontScaling={false}>{current.item.emoji}</Text>
          <Text style={styles.itemWord}>{current.item.word}</Text>
        </Animated.View>
      ) : null}
      {/* The pile: one card outline per picture still waiting, so progress is visible without numbers. */}
      {Array.from({length: Math.min(4, Math.max(0, round.items.length - index - 1))}, (_, k) => (
        <View key={`pile-${k}`} pointerEvents="none" style={[styles.pileCard, {left: layout.item.x, top: layout.item.y, width: layout.item.size, height: layout.item.size, transform: [{rotate: `${(k % 2 ? -1 : 1) * (4 + k * 3)}deg`}]}]} />
      ))}
    </View>
  );
}

export const sortInstruction = (config: GameConfig, seed: number) => buildSortRound(config, seed).instruction;

const styles = StyleSheet.create({
  binWrap: {position: 'absolute'},
  bin: {flex: 1, borderRadius: radius.lg, backgroundColor: colors.paper, borderWidth: 4, borderColor: colors.cognitiveSoft, padding: 12},
  binHint: {borderColor: colors.hint, borderWidth: 6},
  binHeader: {flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6},
  binName: {fontFamily: fonts.display, color: colors.ink},
  binItems: {flex: 1, flexDirection: 'row', flexWrap: 'wrap', alignContent: 'center', justifyContent: 'center', gap: 4},
  binItem: {fontSize: 46},
  item: {position: 'absolute', zIndex: 5, backgroundColor: colors.paper, borderRadius: radius.lg, alignItems: 'center', justifyContent: 'center', borderWidth: 4, borderColor: colors.cognitive},
  itemWord: {fontFamily: fonts.strong, color: colors.ink, fontSize: 18, marginTop: -4},
  pileCard: {position: 'absolute', zIndex: 1, borderRadius: radius.lg, backgroundColor: colors.paper, borderWidth: 3, borderColor: colors.cognitiveSoft},
  binYes: {borderColor: colors.success, backgroundColor: colors.successSoft},
});

