import React, {useRef, useState} from 'react';
import {StyleSheet, Text, View} from 'react-native';
import {DEFAULTS} from '../config';
import {say} from '../speech';
import {colors, fonts, radius} from '../theme';
import type {GameConfig} from '../types';
import {Tile} from '../ui/components';
import {SENTENCES, SENTENCE_DISTRACTORS, type Item} from './content';
import {clampInt, makeRng, num, pick, sample, shuffle} from './random';
import type {BoardProps, Instruction} from './types';

export interface SentenceRound { tokens: Item[]; choices: Item[]; instruction: Instruction }

export function buildSentenceRound(config: GameConfig, seed: number): SentenceRound {
  const rng = makeRng(seed);
  const length = clampInt(num(config.sentenceLength, 2), 2, 5);
  const vocab = clampInt(num(config.vocabLevel, 1), 1, 3);
  const bank = SENTENCES[length];
  const sentence = pick(rng, bank.filter(s => s.vocab <= vocab).length ? bank.filter(s => s.vocab <= vocab) : bank);
  const words = new Set(sentence.tokens.map(t => t.word));
  const distractors = sample(rng, SENTENCE_DISTRACTORS.filter(d => !words.has(d.word)), clampInt(num(config.distractorCount), 0, 4));
  const text = sentence.tokens.map(t => t.word).join(' ');
  return {tokens: sentence.tokens, choices: shuffle(rng, [...sentence.tokens, ...distractors]), instruction: {text: `Make: “${text}”`, spoken: `Make the sentence. ${text}.`}};
}

export function SentenceStripsBoard({config, seed, width, height, reducedMotion, onInteract, onAttemptEnd}: BoardProps) {
  const round = useRef(buildSentenceRound(config, seed)).current;
  const [placed, setPlaced] = useState<number[]>([]); // indexes into choices
  const errors = useRef(0);
  const done = useRef(false);
  const wiggles = useRef(round.choices.map(() => ({current: null as null | (() => void)}))).current;
  const n = round.tokens.length;
  const gap = width < 500 ? 8 : 16;
  const slot = Math.min(150, (width - 2 * gap - 8 - gap * (n - 1)) / n, height * 0.3);
  const perRow = Math.min(round.choices.length, Math.max(3, Math.floor((width - 20) / 150)));
  const choice = Math.min(140, (width - 20) / perRow - 22, height * 0.28);
  const nextWord = round.tokens[placed.length]?.word;

  const tap = (i: number) => {
    if (done.current || placed.includes(i)) return;
    onInteract();
    const task = () => ({errors: errors.current, sentenceLength: n, distractors: round.choices.length - n});
    if (round.choices[i].word === nextWord) {
      const next = [...placed, i];
      setPlaced(next);
      if (next.length === n) {
        done.current = true;
        say(round.tokens.map(t => t.word).join(' '));
        setTimeout(() => onAttemptEnd({success: true, task: task()}), 2000); // let the spoken sentence finish before the praise
      }
    } else {
      wiggles[i].current?.();
      errors.current++;
      if (errors.current >= DEFAULTS.attemptFailure.incorrectFlipLimit) {
        done.current = true;
        setTimeout(() => onAttemptEnd({success: false, task: task()}), 700);
      }
    }
  };

  return (
    <View style={[styles.root, {width, height}]}>
      <View style={[styles.strip, {gap, padding: gap}]}>
        {round.tokens.map((t, k) => {
          const filled = placed[k] !== undefined;
          return (
            <View key={k} testID={`sentence-slot-${k}`} style={[styles.slot, {width: slot, height: slot * 1.08}, filled ? styles.slotFilled : null]}>
              {filled ? <Text style={{fontSize: slot * 0.48}} allowFontScaling={false}>{t.emoji}</Text>
                : config.workedExample ? <Text style={{fontSize: slot * 0.48, opacity: 0.25}} allowFontScaling={false}>{t.emoji}</Text>
                : <Text style={[styles.slotNumber, {fontSize: slot * 0.3}]}>{k + 1}</Text>}
              <Text numberOfLines={1} style={[styles.slotWord, {fontSize: Math.max(14, slot * 0.14)}]}>{filled || config.workedExample ? t.word : ' '}</Text>
            </View>
          );
        })}
      </View>
      <View style={styles.choices}>
        {round.choices.map((c, i) => (
          <Tile key={i} testID={`sentence-choice-${i}`} emoji={c.emoji} label={c.word} size={choice} onPress={() => tap(i)}
            dimmed={placed.includes(i)} disabled={placed.includes(i)} wiggleRef={wiggles[i]} reducedMotion={reducedMotion}
            highlighted={!!config.highlightNext && !placed.includes(i) && c.word === nextWord && round.choices.findIndex((x, j) => x.word === nextWord && !placed.includes(j)) === i} />
        ))}
      </View>
    </View>
  );
}

export const sentenceInstruction = (config: GameConfig, seed: number) => buildSentenceRound(config, seed).instruction;

const styles = StyleSheet.create({
  root: {alignItems: 'center', justifyContent: 'space-evenly'},
  strip: {flexDirection: 'row', gap: 16, padding: 16, borderRadius: radius.lg, backgroundColor: colors.verbalSoft},
  slot: {borderRadius: radius.md, borderWidth: 4, borderStyle: 'dashed', borderColor: colors.verbal, alignItems: 'center', justifyContent: 'center', backgroundColor: 'rgba(255,255,255,0.6)'},
  slotFilled: {borderStyle: 'solid', backgroundColor: colors.paper},
  slotNumber: {fontFamily: fonts.display, color: colors.verbal, opacity: 0.5},
  slotWord: {fontFamily: fonts.strong, color: colors.ink},
  choices: {flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'center', gap: 22, maxWidth: '100%'},
});
