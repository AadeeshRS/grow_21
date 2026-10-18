import React, {useRef, useState} from 'react';
import {StyleSheet, View} from 'react-native';
import type {GameConfig} from '../types';
import {Tile} from '../ui/components';
import {ColorObject, DEFAULT_FILL, OBJECT_KINDS, SIMILAR, type ObjectKind} from './ColorObject';
import {COLORS, type ColorName} from './content';
import {clampInt, makeRng, num, pick, shuffle} from './random';
import type {BoardProps, Instruction} from './types';

export interface SearchOption { kind: ObjectKind; color?: ColorName; big?: boolean }
export interface SearchRound { options: SearchOption[]; answerIndex: number; instruction: Instruction; attributes: number }

export const describeOption = (o: SearchOption) => [o.big === undefined ? '' : o.big ? 'big' : 'small', o.color ?? '', o.kind].filter(Boolean).join(' ');
const same = (a: SearchOption, b: SearchOption) => a.kind === b.kind && a.color === b.color && a.big === b.big;

/**
 * One target plus distractors. Distractors always differ from the target in at least one attribute
 * named in the instruction; higher visual similarity makes them share more with the target.
 */
export function buildSearchRound(config: GameConfig, seed: number): SearchRound {
  const rng = makeRng(seed);
  const attributes = clampInt(num(config.attributeCount, 1), 1, 3);
  const optionCount = clampInt(num(config.optionCount, 2), 2, 3);
  const similarity = num(config.visualSimilarity);
  const anyColor = () => pick(rng, COLORS).name;
  const target: SearchOption = {kind: pick(rng, OBJECT_KINDS), color: attributes >= 2 ? anyColor() : undefined, big: attributes >= 3 ? rng() < 0.5 : undefined};
  const options: SearchOption[] = [target];
  for (let guard = 0; options.length < optionCount && guard < 300; guard++) {
    const shareKind = attributes >= 2 && rng() < 0.25 + similarity;
    const kind = shareKind ? target.kind : rng() < similarity ? SIMILAR[target.kind] : pick(rng, OBJECT_KINDS.filter(k => k !== target.kind));
    const d: SearchOption = {
      kind,
      color: attributes >= 2 ? (rng() < similarity ? target.color : anyColor()) : undefined,
      big: attributes >= 3 ? (rng() < similarity ? target.big : rng() < 0.5) : undefined,
    };
    if (!options.some(o => same(o, d))) options.push(d);
  }
  const shuffled = shuffle(rng, options);
  const complexity = clampInt(num(config.instructionComplexity, 1), 1, 3);
  const phrase = `the ${describeOption(target)}`;
  const spoken = complexity === 1 ? `Find ${phrase}.` : complexity === 2 ? `Can you find ${phrase}?` : `Look carefully. Find ${phrase}.`;
  return {options: shuffled, answerIndex: shuffled.indexOf(target), attributes, instruction: {text: `Find ${phrase}`, spoken}};
}

export function AudioSearchBoard({config, seed, width, height, reducedMotion, onInteract, onAttemptEnd}: BoardProps) {
  const round = useRef(buildSearchRound(config, seed)).current;
  const [locked, setLocked] = useState(false);
  const wiggles = useRef(round.options.map(() => ({current: null as null | (() => void)}))).current;
  const n = round.options.length;
  const size = Math.max(120, Math.min(280, (width - 40 * (n + 1)) / n, height * 0.75));

  const choose = (i: number) => {
    if (locked) return;
    onInteract();
    setLocked(true);
    const task = {chosenIndex: i, answerIndex: round.answerIndex, options: n, attributes: round.attributes};
    if (i === round.answerIndex) onAttemptEnd({success: true, task});
    else {
      wiggles[i].current?.();
      setTimeout(() => onAttemptEnd({success: false, task}), 700);
    }
  };

  return (
    <View style={[styles.root, {width, height}]}>
      {round.options.map((o, i) => {
        const fill = o.color ? COLORS.find(c => c.name === o.color)!.fill : DEFAULT_FILL[o.kind];
        return (
          <Tile key={i} testID={`search-option-${i}`} accessibilityLabel={describeOption(o)} size={size}
            content={<ColorObject kind={o.kind} fill={fill} size={size * (o.big === false ? 0.42 : o.big ? 0.86 : 0.7)} />}
            onPress={() => choose(i)} disabled={locked} wiggleRef={wiggles[i]} reducedMotion={reducedMotion} />
        );
      })}
    </View>
  );
}

export const searchInstruction = (config: GameConfig, seed: number) => buildSearchRound(config, seed).instruction;

const styles = StyleSheet.create({
  root: {flexDirection: 'row', alignItems: 'center', justifyContent: 'space-evenly', flexWrap: 'wrap'},
});
