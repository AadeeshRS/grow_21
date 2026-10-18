import React, {useRef, useState} from 'react';
import {Animated, Easing, StyleSheet, Text, View} from 'react-native';
import {colors, radius, shadow} from '../theme';
import {NATIVE_DRIVER, Pulse, Tile} from '../ui/components';
import {LOOKALIKE_GROUPS, type Item} from './content';
import {clampInt, makeRng, num, pick, sample, shuffle} from './random';
import type {GameConfig} from '../types';
import type {BoardProps} from './types';

export interface PatternRound {
  sequence: Item[];
  unitLength: number;
  answer: Item;
  choices: Item[];
  answerIndex: number;
}

const UNITS: Record<number, number[][]> = {
  1: [[0, 1]],
  2: [[0, 1, 2], [0, 0, 1]],
  3: [[0, 0, 1, 1], [0, 1, 1, 2], [0, 1, 2, 3]],
};

export function buildPatternRound(config: GameConfig, seed: number): PatternRound {
  const rng = makeRng(seed);
  const unit = pick(rng, UNITS[clampInt(num(config.ruleCount, 1), 1, 3)]);
  const distinct = Math.max(...unit) + 1;
  const similar = num(config.visualSimilarity) >= 0.35;
  const groups = shuffle(rng, LOOKALIKE_GROUPS);
  const pool = similar ? sample(rng, groups[0], distinct) : groups.slice(0, distinct).map(g => pick(rng, g));
  const length = Math.max(clampInt(num(config.sequenceLength, 3), 3, 7), unit.length + 1);
  const sequence = Array.from({length}, (_, i) => pool[unit[i % unit.length]]);
  const answer = sequence[length - 1];
  const nChoices = clampInt(num(config.answerChoices, num(config.distractorCount, 1) + 1), 2, 4);
  // Most plausible distractors first: other pattern members, then look-alikes, then anything else.
  const others = [
    ...pool.filter(p => p !== answer),
    ...(similar ? groups[0] : groups[distinct % groups.length]).filter(p => !pool.includes(p)),
  ];
  const choices = shuffle(rng, [answer, ...others.slice(0, nChoices - 1)]);
  return {sequence, unitLength: unit.length, answer, choices, answerIndex: choices.indexOf(answer)};
}

export function PatternTrainBoard({config, seed, width, height, reducedMotion, onInteract, onAttemptEnd}: BoardProps) {
  const round = useRef(buildPatternRound(config, seed)).current;
  const [filled, setFilled] = useState(false);
  const [locked, setLocked] = useState(false);
  const drive = useRef(new Animated.Value(0)).current;
  const wiggles = useRef<Array<React.MutableRefObject<(() => void) | null>>>(round.choices.map(() => ({current: null}))).current;
  const highlightUnit = !!config.highlightUnit;
  const animateUnit = !!config.animateUnit;

  const cars = round.sequence.length;
  const car = Math.min(132, Math.max(54, (width - 40) / (cars + 1.4)), height * 0.3);
  const choice = Math.min(150, Math.max(84, (width - 60) / round.choices.length - 24), height * 0.32);

  const choose = (i: number) => {
    if (locked) return;
    onInteract();
    setLocked(true);
    if (i === round.answerIndex) {
      setFilled(true);
      Animated.timing(drive, {toValue: 1, duration: reducedMotion ? 1 : 900, delay: 350, easing: Easing.in(Easing.quad), useNativeDriver: NATIVE_DRIVER}).start();
      setTimeout(() => onAttemptEnd({success: true, task: {chosenIndex: i, answerIndex: round.answerIndex, choices: round.choices.length, unitLength: round.unitLength, sequenceLength: cars}}), 1250);
    } else {
      wiggles[i].current?.();
      setTimeout(() => onAttemptEnd({success: false, task: {chosenIndex: i, answerIndex: round.answerIndex, choices: round.choices.length, unitLength: round.unitLength, sequenceLength: cars}}), 700);
    }
  };

  return (
    <View style={[styles.root, {width, height}]}>
      <Animated.View style={[styles.train, {transform: [{translateX: drive.interpolate({inputRange: [0, 1], outputRange: [0, width * 1.2]})}]}]}>
        <Text style={{fontSize: car * 0.8}} accessibilityLabel="train">🚂</Text>
        {round.sequence.map((item, i) => {
          const last = i === cars - 1;
          const inUnit = i < round.unitLength;
          return (
            <Pulse key={i} active={animateUnit && inUnit} reducedMotion={reducedMotion}>
              <View testID={last ? 'pattern-blank' : undefined} style={[styles.car, shadow(3), {width: car, height: car * 0.86},
                last && !filled ? styles.blank : null,
                highlightUnit && inUnit ? styles.unit : null]}>
                <Text style={{fontSize: car * 0.46}} allowFontScaling={false}>{last ? (filled ? item.emoji : '❓') : item.emoji}</Text>
                <View style={[styles.wheel, {left: car * 0.14, width: car * 0.17, height: car * 0.17, borderRadius: car, bottom: -car * 0.08}]} />
                <View style={[styles.wheel, {right: car * 0.14, width: car * 0.17, height: car * 0.17, borderRadius: car, bottom: -car * 0.08}]} />
              </View>
            </Pulse>
          );
        })}
      </Animated.View>
      <View style={styles.track} />
      <View style={styles.choices}>
        {round.choices.map((item, i) => (
          <Tile key={i} testID={`pattern-choice-${i}`} emoji={item.emoji} accessibilityLabel={item.word} size={choice} onPress={() => choose(i)} disabled={locked} wiggleRef={wiggles[i]} reducedMotion={reducedMotion} />
        ))}
      </View>
    </View>
  );
}

export const patternInstruction = () => ({text: 'What comes next?', spoken: 'What comes next? Finish the train.'});

const styles = StyleSheet.create({
  root: {alignItems: 'center', justifyContent: 'space-evenly', overflow: 'hidden'},
  train: {flexDirection: 'row', alignItems: 'flex-end', columnGap: 8, rowGap: 18, flexWrap: 'wrap', justifyContent: 'center'},
  car: {backgroundColor: colors.paper, borderRadius: radius.sm, alignItems: 'center', justifyContent: 'center', borderWidth: 3, borderColor: colors.cognitiveSoft},
  blank: {borderStyle: 'dashed', borderColor: colors.cognitive, backgroundColor: colors.cognitiveSoft},
  unit: {borderColor: colors.hint, borderWidth: 5},
  wheel: {position: 'absolute', backgroundColor: colors.ink},
  track: {height: 6, alignSelf: 'stretch', marginHorizontal: 24, marginTop: -18, borderRadius: 3, backgroundColor: colors.line},
  choices: {flexDirection: 'row', gap: 24, flexWrap: 'wrap', justifyContent: 'center'},
});

