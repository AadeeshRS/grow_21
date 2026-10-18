import React, {useRef, useState} from 'react';
import {Pressable, StyleSheet, View} from 'react-native';
import Svg, {Circle, Path, Rect} from 'react-native-svg';
import {DEFAULTS} from '../config';
import {colors} from '../theme';
import type {GameConfig} from '../types';
import {Pulse} from '../ui/components';
import {useDrag} from '../ui/drag';
import {BLOCK_COLORS, BLOCK_SHAPES, type BlockShape} from './content';
import {clampInt, makeRng, num, sample, shuffle} from './random';
import type {BoardProps} from './types';

export interface BlockSpec { shape: BlockShape; color: string }
export interface Slot { x: number; y: number; layer: number }
export interface BlockLayout { size: number; targets: Slot[]; tray: Array<{x: number; y: number}>; blocks: BlockSpec[]; trayOrder: number[] }

/** Geometry for the configured structure (exported so tests can aim drags). */
export function buildBlockLayout(config: GameConfig, seed: number, width: number, height: number, scale = 1): BlockLayout {
  const rng = makeRng(seed);
  const n = clampInt(num(config.blockCount, 1), 1, 5);
  const structure = String(config.structure ?? 'single');
  const topH = height * 0.6;
  const layers = structure === 'stack' ? n : structure === 'pattern' && n > 1 ? 2 : 1;
  const across = structure === 'stack' ? 1 : structure === 'pattern' ? Math.ceil(n / 2) : n;
  // Never below the 64 dp target size unless the board itself cannot fit it.
  const fit = Math.min((width - 40) / (Math.max(across, n) + 1), (topH - 16) / (layers + 0.15));
  const size = Math.min(fit, Math.max(64, num(config.blockSizeDp, 90) * scale));
  const targets: Slot[] = [];
  const rowX = (count: number, i: number) => width / 2 + (i - (count - 1) / 2) * (size + 16);
  if (structure === 'stack') {
    for (let i = 0; i < n; i++) targets.push({x: width / 2, y: topH - size / 2 - 8 - i * size, layer: i});
  } else if (structure === 'pattern' && n > 1) {
    const bottom = Math.ceil(n / 2);
    for (let i = 0; i < bottom; i++) targets.push({x: rowX(bottom, i), y: topH - size / 2 - 8, layer: 0});
    for (let i = 0; i < n - bottom; i++) targets.push({x: rowX(n - bottom, i), y: topH - size * 1.5 - 8, layer: 1});
  } else {
    for (let i = 0; i < n; i++) targets.push({x: rowX(n, i), y: topH / 2, layer: 0});
  }
  const shapes = sample(rng, BLOCK_SHAPES, n);
  const tint = sample(rng, BLOCK_COLORS, n);
  const blocks = shapes.map((shape, i) => ({shape, color: tint[i]}));
  const trayOrder = shuffle(rng, blocks.map((_, i) => i));
  const trayY = topH + (height - topH) / 2;
  const tray = trayOrder.map((_, k) => ({x: rowX(n, k), y: trayY}));
  return {size, targets, tray, blocks, trayOrder};
}

export function BlockShapeSvg({shape, size, fill, dashed}: {shape: BlockShape; size: number; fill: string; dashed?: boolean}) {
  const stroke = dashed ? {stroke: fill, strokeWidth: 6, strokeDasharray: '10 7', fill: fill + '33'} : {fill, stroke: 'rgba(29,43,79,0.18)', strokeWidth: 3};
  return (
    <Svg width={size} height={size} viewBox="0 0 100 100">
      {shape === 'square' ? <Rect x={9} y={9} width={82} height={82} rx={14} {...stroke} />
        : shape === 'circle' ? <Circle cx={50} cy={50} r={42} {...stroke} />
        : shape === 'triangle' ? <Path d="M50 9 L92 88 L8 88 Z" strokeLinejoin="round" {...stroke} />
        : shape === 'star' ? <Path d="M50 6 L62 37 L95 38 L69 58 L79 91 L50 72 L21 91 L31 58 L5 38 L38 37 Z" strokeLinejoin="round" {...stroke} />
        : <Path d="M50 88 C 18 66, 6 46, 14 28 C 22 10, 44 12, 50 28 C 56 12, 78 10, 86 28 C 94 46, 82 66, 50 88 Z" {...stroke} />}
    </Svg>
  );
}

function Block({spec, size, at, selected, onTap, onDrop, onGrab, testID}: {spec: BlockSpec; size: number; at: {x: number; y: number}; selected: boolean; onTap(): void; onDrop(c: {x: number; y: number}): void; onGrab(): void; testID: string}) {
  const [d, setD] = useState({dx: 0, dy: 0});
  const handlers = useDrag({
    onStart: onGrab,
    onMove: setD,
    onEnd: e => {
      setD({dx: 0, dy: 0});
      if (Math.hypot(e.dx, e.dy) < 10) onTap();
      else onDrop({x: at.x + e.dx, y: at.y + e.dy});
    },
  });
  const lifted = selected || d.dx !== 0 || d.dy !== 0;
  return (
    <View {...handlers} testID={testID} accessibilityRole="button" accessibilityLabel={`${spec.shape} block`}
      style={[styles.block, {left: at.x - size / 2, top: at.y - size / 2, width: size, height: size, zIndex: lifted ? 10 : 2,
        transform: [{translateX: d.dx}, {translateY: d.dy}, {scale: lifted ? 1.08 : 1}]}]}>
      <BlockShapeSvg shape={spec.shape} size={size} fill={spec.color} />
    </View>
  );
}

export function BlockStackerBoard({config, seed, width, height, scale, reducedMotion, onInteract, onAttemptEnd}: BoardProps) {
  const layout = useRef(buildBlockLayout(config, seed, width, height, scale)).current;
  const {size, targets, tray, blocks, trayOrder} = layout;
  const [placed, setPlaced] = useState<boolean[]>(blocks.map(() => false));
  const [selected, setSelected] = useState<number | null>(null);
  const stats = useRef({releaseMisses: 0, wrongTarget: 0, orderErrors: 0, drags: 0, taps: 0});
  const done = useRef(false);
  const tolerance = num(config.placementToleranceDp, 30) * num(config.snapMultiplier, 1) * scale;
  const tapToPlace = config.snapMultiplier !== undefined; // help level ≥ 2 also accepts tap-then-tap
  const isOpen = (t: number, filled = placed) => targets.every((o, j) => o.layer >= targets[t].layer || filled[j]);

  const fail = () => {
    const s = stats.current;
    if (s.releaseMisses + s.wrongTarget + s.orderErrors >= DEFAULTS.attemptFailure.dropFailLimit) {
      done.current = true;
      setTimeout(() => onAttemptEnd({success: false, motor: {...s, blocks: blocks.length, toleranceDp: Math.round(tolerance)}}), 600);
    }
  };

  const resolve = (block: number, target: number | null) => {
    if (done.current) return;
    setSelected(null);
    if (target === null) { stats.current.releaseMisses++; return fail(); }
    if (target !== block) { stats.current.wrongTarget++; return fail(); }
    if (!isOpen(target)) { stats.current.orderErrors++; return fail(); }
    const next = placed.map((p, i) => p || i === block);
    setPlaced(next);
    if (next.every(Boolean)) {
      done.current = true;
      setTimeout(() => onAttemptEnd({success: true, motor: {...stats.current, blocks: blocks.length, toleranceDp: Math.round(tolerance)}}), 500);
    }
  };

  const targetAt = (c: {x: number; y: number}) => {
    let best: number | null = null, bestD = Infinity;
    targets.forEach((t, i) => {
      if (placed[i]) return;
      const dist = Math.hypot(t.x - c.x, t.y - c.y);
      if (dist <= tolerance + size * 0.35 && dist < bestD) { best = i; bestD = dist; }
    });
    return best;
  };

  const nextOpen = targets.findIndex((_, i) => !placed[i] && isOpen(i));
  const focus = selected ?? null;
  return (
    <View style={{width, height}}>
      <View style={[styles.ground, {top: height * 0.6 - 4}]} />
      {targets.map((t, i) => (
        <Pulse key={i} active={(!!config.pulseTarget && i === nextOpen) || (!!config.highlight && focus === i)} reducedMotion={reducedMotion}
          style={[styles.slot, {left: t.x - size / 2, top: t.y - size / 2, width: size, height: size}]}>
          <Pressable testID={`block-target-${i}`} accessibilityLabel={`${blocks[i].shape} space`} disabled={!tapToPlace || selected === null || placed[i]}
            onPress={() => { if (selected !== null) { stats.current.taps++; resolve(selected, i); } }}>
            {placed[i] ? <BlockShapeSvg shape={blocks[i].shape} size={size} fill={blocks[i].color} />
              : <View style={{opacity: isOpen(i) ? 1 : 0.45}}><BlockShapeSvg shape={blocks[i].shape} size={size} fill={blocks[i].color} dashed /></View>}
          </Pressable>
        </Pulse>
      ))}
      {trayOrder.map((b, k) => placed[b] ? null : (
        <Block key={b} testID={`block-${b}`} spec={blocks[b]} size={size} at={tray[k]} selected={selected === b}
          onGrab={() => { if (!done.current) onInteract(); }}
          onTap={() => { if (tapToPlace && !done.current) setSelected(selected === b ? null : b); }}
          onDrop={c => { stats.current.drags++; resolve(b, targetAt(c)); }} />
      ))}
    </View>
  );
}

export const blockInstruction = () => ({text: 'Put each block in its space', spoken: 'Slide each block onto its matching shape.'});

const styles = StyleSheet.create({
  ground: {position: 'absolute', left: 24, right: 24, height: 8, borderRadius: 4, backgroundColor: colors.line},
  slot: {position: 'absolute'},
  block: {position: 'absolute'},
});
