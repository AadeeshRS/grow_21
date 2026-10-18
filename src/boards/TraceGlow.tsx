import React, {useEffect, useMemo, useRef, useState} from 'react';
import {StyleSheet, Text, View} from 'react-native';
import Svg, {Circle, Path} from 'react-native-svg';
import {DEFAULTS} from '../config';
import {colors} from '../theme';
import type {GameConfig} from '../types';
import {useDrag} from '../ui/drag';
import {makeRng, num, pick} from './random';
import type {BoardProps} from './types';

export interface Pt { x: number; y: number }
export interface TracePath { points: Pt[]; width: number; shape: string }

/** Dense polyline (≈6 px spacing) for the configured shape, fitted inside the board. */
export function buildTracePath(config: GameConfig, seed: number, boardW: number, boardH: number, scale = 1): TracePath {
  const rng = makeRng(seed);
  const pathWidth = Math.max(44, Math.min(num(config.pathWidthDp, 80) * scale, Math.min(boardW, boardH) * 0.3));
  const margin = pathWidth / 2 + 36;
  const boxW = Math.max(60, boardW - 2 * margin);
  const boxH = Math.max(60, boardH - 2 * margin);
  const span = Math.min(boxW, Math.max(160, num(config.pathLengthDp, 300) * scale));
  const curvature = Math.max(0.25, num(config.curvature, 0.3));
  const turns = Math.max(0, Math.round(num(config.turnCount)));
  const cx = boardW / 2;
  const cy = boardH / 2;
  const left = cx - span / 2;
  const amp = Math.min(boxH / 2, span * 0.28) * Math.min(1, curvature + 0.25);
  const shapeType = String(config.shapeType ?? 'straight');
  const raw: Pt[] = [];
  let shape = shapeType;
  if (shapeType === 'straight') {
    raw.push({x: left, y: cy}, {x: left + span, y: cy});
  } else if (shapeType === 'gentleCurve') {
    const periods = 0.5 + Math.min(1.5, turns / 4);
    for (let i = 0; i <= 60; i++) { const t = i / 60; raw.push({x: left + span * t, y: cy - amp * Math.sin(Math.PI * 2 * periods * t)}); }
  } else if (shapeType === 'zigzag') {
    const segs = Math.max(2, turns);
    for (let i = 0; i <= segs; i++) raw.push({x: left + (span * i) / segs, y: cy + (i % 2 ? -amp : amp) * (i === 0 || i === segs ? 0.6 : 1)});
  } else {
    shape = pick(rng, ['circle', 'u', 's']);
    const r = Math.min(boxW, boxH) / 2;
    if (shape === 'circle') for (let i = 0; i <= 80; i++) { const a = -Math.PI / 2 + (i / 80) * Math.PI * 1.6; raw.push({x: cx + r * Math.cos(a), y: cy + r * Math.sin(a)}); }
    else if (shape === 'u') raw.push({x: cx - r, y: cy - r}, {x: cx - r, y: cy + r * 0.6}, {x: cx - r * 0.6, y: cy + r}, {x: cx + r * 0.6, y: cy + r}, {x: cx + r, y: cy + r * 0.6}, {x: cx + r, y: cy - r});
    else for (let i = 0; i <= 80; i++) { const t = i / 80; raw.push({x: cx - r + 2 * r * t, y: cy - r * 0.8 * Math.sin(Math.PI * 2 * t)}); }
  }
  const points: Pt[] = [raw[0]];
  for (let i = 1; i < raw.length; i++) {
    const a = raw[i - 1], b = raw[i];
    const steps = Math.max(1, Math.ceil(Math.hypot(b.x - a.x, b.y - a.y) / 6));
    for (let k = 1; k <= steps; k++) points.push({x: a.x + ((b.x - a.x) * k) / steps, y: a.y + ((b.y - a.y) * k) / steps});
  }
  return {points, width: pathWidth, shape};
}

const toD = (pts: Pt[]) => pts.map((p, i) => `${i ? 'L' : 'M'}${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(' ');

export function TraceGlowBoard({config, seed, width, height, scale, reducedMotion, onInteract, onAttemptEnd}: BoardProps) {
  const path = useMemo(() => buildTracePath(config, seed, width, height, scale), [config, seed, width, height, scale]);
  const pts = path.points;
  const tolerance = (path.width / 2) * (config.maximumTolerance ? 1.7 : 1.3) + 8;
  const resumeRadius = Math.max(path.width, 56 * scale) * num(config.resumeRadiusMultiplier, 1);
  const [progress, setProgress] = useState(0);
  const [onPath, setOnPath] = useState(true);
  const [ghost, setGhost] = useState(0);
  const origin = useRef<Pt | null>(null);
  const stats = useRef({offPathExits: 0, lifts: 0, maxDeviation: 0, devSum: 0, devN: 0});
  const progressRef = useRef(0);
  const onPathRef = useRef(true);
  const done = useRef(false);

  useEffect(() => {
    if (!config.ghostTrail || reducedMotion) return;
    const id = setInterval(() => setGhost(g => (g + 4 >= pts.length ? progressRef.current : g + 4)), 40);
    return () => clearInterval(id);
  }, [config.ghostTrail, reducedMotion, pts.length]);

  const end = (success: boolean) => {
    if (done.current) return;
    done.current = true;
    const s = stats.current;
    onAttemptEnd({success, motor: {offPathExits: s.offPathExits, lifts: s.lifts, maxDeviationDp: Math.round(s.maxDeviation), meanDeviationDp: s.devN ? Math.round(s.devSum / s.devN) : null, coverage: +(progressRef.current / (pts.length - 1)).toFixed(3), pathWidthDp: Math.round(path.width), shape: path.shape}});
  };

  const track = (p: Pt) => {
    let best = -1, bestD = Infinity;
    const from = Math.max(0, progressRef.current - 6), to = Math.min(pts.length - 1, progressRef.current + 30);
    for (let i = from; i <= to; i++) { const d = Math.hypot(pts[i].x - p.x, pts[i].y - p.y); if (d < bestD) { bestD = d; best = i; } }
    if (bestD <= tolerance) {
      if (!onPathRef.current) { onPathRef.current = true; setOnPath(true); }
      stats.current.maxDeviation = Math.max(stats.current.maxDeviation, bestD);
      stats.current.devSum += bestD; stats.current.devN++;
      if (best > progressRef.current) { progressRef.current = best; setProgress(best); }
      if (progressRef.current >= pts.length - 3) end(true);
    } else if (onPathRef.current) {
      onPathRef.current = false;
      setOnPath(false);
      stats.current.offPathExits++;
      if (stats.current.offPathExits >= DEFAULTS.attemptFailure.dropFailLimit) end(false);
    }
  };

  const handlers = useDrag({
    onStart: loc => {
      if (done.current) return;
      const at = pts[progressRef.current];
      if (Math.hypot(loc.x - at.x, loc.y - at.y) > resumeRadius) { origin.current = null; return; } // must start (or resume) at the glowing point
      onInteract();
      origin.current = loc;
      onPathRef.current = true; setOnPath(true);
    },
    onMove: d => { if (origin.current && !done.current) track({x: origin.current.x + d.dx, y: origin.current.y + d.dy}); },
    onEnd: () => { if (origin.current && !done.current) stats.current.lifts++; origin.current = null; },
  });

  const head = pts[progress];
  const goal = pts[pts.length - 1];
  const glowW = path.width * (config.strongGlow ? 0.8 : 0.6);
  return (
    <View testID="trace-board" {...handlers} style={{width, height}}>
      <View pointerEvents="none" style={StyleSheet.absoluteFill}>
      <Svg width={width} height={height}>
        <Path d={toD(pts)} stroke={colors.motorSoft} strokeWidth={path.width + 18} strokeLinecap="round" strokeLinejoin="round" fill="none" />
        <Path d={toD(pts)} stroke="#9FE3D8" strokeWidth={path.width} strokeLinecap="round" strokeLinejoin="round" fill="none" />
        <Path d={toD(pts)} stroke="#FFFFFF" strokeWidth={4} strokeDasharray="2 14" strokeLinecap="round" fill="none" />
        {progress > 0 ? <Path d={toD(pts.slice(0, progress + 1))} stroke={colors.sun} strokeOpacity={0.45} strokeWidth={glowW * 1.5} strokeLinecap="round" strokeLinejoin="round" fill="none" /> : null}
        {progress > 0 ? <Path d={toD(pts.slice(0, progress + 1))} stroke={colors.hint} strokeWidth={glowW * 0.6} strokeLinecap="round" strokeLinejoin="round" fill="none" /> : null}
        {config.ghostTrail && ghost > progress ? <Circle cx={pts[ghost].x} cy={pts[ghost].y} r={path.width * 0.22} fill={colors.hint} opacity={0.55} /> : null}
        <Circle cx={head.x} cy={head.y} r={resumeRadius * 0.5} fill={onPath ? colors.sun : colors.sunSoft} opacity={0.35} />
      </Svg>
      </View>
      <View pointerEvents="none" style={[styles.sprite, {left: goal.x - 32, top: goal.y - 32}]}><Text style={styles.spriteText}>🌸</Text></View>
      <View pointerEvents="none" style={[styles.sprite, {left: head.x - 32, top: head.y - 32}]}><Text style={styles.spriteText}>🐞</Text></View>
    </View>
  );
}

export const traceInstruction = () => ({text: 'Help the ladybug to the flower', spoken: 'Put your finger on the ladybug. Slide along the path to the flower.'});

const styles = StyleSheet.create({
  sprite: {position: 'absolute', width: 64, height: 64, alignItems: 'center', justifyContent: 'center'},
  spriteText: {fontSize: 46, lineHeight: 58},
});
