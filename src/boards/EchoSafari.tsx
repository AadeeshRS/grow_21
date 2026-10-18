import React, {useEffect, useRef, useState} from 'react';
import {Pressable, StyleSheet, Text, View} from 'react-native';
import {say, stopSpeaking} from '../speech';
import {colors, fonts, radius, shadow} from '../theme';
import type {GameConfig} from '../types';
import {Pulse} from '../ui/components';
import {createVoiceDetector, type VoiceDetector} from '../voice';
import {ANIMAL_SOUNDS} from './content';
import {clampInt, makeRng, num, shuffle} from './random';
import type {BoardProps} from './types';

type Animal = (typeof ANIMAL_SOUNDS)[number];
export interface EchoRound { prompts: Animal[][] }

export function buildEchoRound(config: GameConfig, seed: number): EchoRound {
  const rng = makeRng(seed);
  const pool = ANIMAL_SOUNDS.slice(0, clampInt(num(config.promptVariety, 2), 2, ANIMAL_SOUNDS.length));
  const perPrompt = config.demand === 'twoSound' ? 2 : 1;
  // Walk a shuffled pool so consecutive prompts use different animals.
  const order = shuffle(rng, pool);
  let k = 0;
  const prompts = Array.from({length: clampInt(num(config.repetitions, 1), 1, 3)}, () => Array.from({length: perPrompt}, () => order[k++ % order.length]));
  return {prompts};
}

const modelSentence = (animals: Animal[], repeats: number) => {
  const sounds = animals.map(a => `${a.sound}!`).join(' ');
  const intro = animals.map(a => `The ${a.word} says ${a.sound}.`).join(' ');
  return `${intro} ${Array.from({length: repeats}, () => sounds).join(' ')} Your turn!`;
};

type Phase = 'model' | 'listen' | 'heard';

/**
 * Echo Safari: the animal models a sound, then the child makes any sound.
 * With a microphone (web), voice activity ends the prompt; without one, a grown-up confirms.
 * There is no pronunciation scoring: any vocalisation counts, as in the source design.
 */
export function EchoSafariBoard({config, seed, width, height, reducedMotion, onInteract, onAttemptEnd}: BoardProps) {
  const round = useRef(buildEchoRound(config, seed)).current;
  const [prompt, setPrompt] = useState(0);
  const [phase, setPhase] = useState<Phase>('model');
  const [mode, setMode] = useState<'pending' | 'vad' | 'observer'>('pending');
  const detector = useRef<VoiceDetector | null>(null);
  const listenStart = useRef(0);
  const latencies = useRef<number[]>([]);
  const peak = useRef(0);
  const done = useRef(false);
  const windowMs = num(config.responseWindowMs, 8000);
  const repeats = 1 + num(config.repeatModel, 0);
  const animals = round.prompts[prompt];

  useEffect(() => {
    let alive = true;
    void createVoiceDetector().then(d => {
      if (!alive) { d?.close(); return; }
      detector.current = d;
      setMode(d ? 'vad' : 'observer');
    });
    return () => { alive = false; detector.current?.close(); stopSpeaking(); };
  }, []);

  const finish = (success: boolean) => {
    if (done.current) return;
    done.current = true;
    onAttemptEnd({success, speech: {mode, prompts: round.prompts.length, promptsCompleted: latencies.current.length, responseLatenciesMs: latencies.current, peakRms: mode === 'vad' ? peak.current : undefined, policy: 'any_detected_vocalization'}});
  };

  const heard = (info?: {peakRms: number}) => {
    if (done.current) return;
    onInteract();
    latencies.current.push(Math.round(performance.now() - listenStart.current));
    if (info) peak.current = Math.max(peak.current, info.peakRms);
    setPhase('heard');
    setTimeout(() => {
      if (prompt + 1 >= round.prompts.length) finish(true);
      else { setPrompt(prompt + 1); setPhase('model'); }
    }, 900);
  };

  // Model the sound, then open the response window (speech end is not reported on every platform).
  useEffect(() => {
    if (phase !== 'model' || mode === 'pending') return;
    let started = false;
    const begin = () => { if (!started) { started = true; listenStart.current = performance.now(); setPhase('listen'); } };
    const sentence = modelSentence(animals, repeats);
    say(sentence, () => setTimeout(begin, 250));
    const fallback = setTimeout(begin, 1000 + sentence.split(' ').length * 520);
    return () => clearTimeout(fallback);
  }, [phase, mode, prompt]);

  useEffect(() => {
    if (phase !== 'listen' || mode !== 'vad' || !detector.current) return;
    const stop = detector.current.listen(info => heard(info));
    const timeout = setTimeout(() => { stop(); finish(false); }, windowMs);
    return () => { stop(); clearTimeout(timeout); };
  }, [phase, mode, prompt]);

  const size = Math.min(height * 0.42, width * 0.32, 260) * (config.strongCue ? 1.15 : 1);
  return (
    <View style={[styles.root, {width, height}]}>
      <View style={styles.dots}>
        {round.prompts.map((_, i) => <View key={i} style={[styles.dot, i < prompt || (i === prompt && phase === 'heard') ? styles.dotDone : null]} />)}
      </View>
      <View style={styles.stage}>
        {animals.map((a, i) => (
          <View key={i} style={styles.animalCol}>
            <Pulse active={phase === 'model' || !!config.strongCue} reducedMotion={reducedMotion}>
              <View style={[styles.animal, shadow(8), {width: size, height: size, borderRadius: size / 2}, phase === 'heard' ? styles.animalHappy : null]}>
                <Text style={{fontSize: size * 0.56}} allowFontScaling={false}>{a.emoji}</Text>
              </View>
            </Pulse>
            <View style={[styles.bubble, shadow(3)]}><Text style={[styles.sound, {fontSize: Math.max(26, size * 0.17)}]}>{a.sound}!</Text></View>
          </View>
        ))}
      </View>
      <View style={styles.footer}>
        {phase === 'heard' ? <Text style={styles.status}>⭐ Lovely sound!</Text>
          : phase === 'model' || mode === 'pending' ? <Text style={styles.status}>👂 Listen…</Text>
          : mode === 'vad' ? <Text testID="echo-listening" style={[styles.status, styles.yourTurn]}>🎤 Your turn!</Text>
          : (
            <View style={styles.observer}>
              <Text style={[styles.status, styles.yourTurn]}>🎤 Your turn!</Text>
              <Text style={styles.observerHint}>Grown-up: tap after your child's turn</Text>
              <View style={styles.observerRow}>
                <Pressable testID="echo-observer-yes" accessibilityRole="button" accessibilityLabel="My child made a sound" onPress={() => heard()} style={[styles.observerBtn, {backgroundColor: colors.success}]}>
                  <Text style={styles.observerText}>✓ Made a sound</Text>
                </Pressable>
                <Pressable testID="echo-observer-no" accessibilityRole="button" accessibilityLabel="Not yet" onPress={() => { onInteract(); finish(false); }} style={[styles.observerBtn, {backgroundColor: colors.inkFaint}]}>
                  <Text style={styles.observerText}>Not yet</Text>
                </Pressable>
              </View>
            </View>
          )}
      </View>
    </View>
  );
}

export const echoInstruction = () => ({text: 'Copy the animal', spoken: 'Listen to the animal. Then you make the sound!'});

const styles = StyleSheet.create({
  root: {alignItems: 'center', justifyContent: 'space-between', paddingVertical: 8},
  dots: {flexDirection: 'row', gap: 10},
  dot: {width: 18, height: 18, borderRadius: 9, backgroundColor: colors.line},
  dotDone: {backgroundColor: colors.sun},
  stage: {flexDirection: 'row', gap: 40, alignItems: 'center'},
  animalCol: {alignItems: 'center', gap: 14},
  animal: {backgroundColor: colors.verbalSoft, alignItems: 'center', justifyContent: 'center', borderWidth: 5, borderColor: colors.verbal},
  animalHappy: {borderColor: colors.success, backgroundColor: colors.successSoft},
  bubble: {backgroundColor: colors.paper, borderRadius: radius.pill, paddingHorizontal: 22, paddingVertical: 8},
  sound: {fontFamily: fonts.display, color: colors.verbal},
  footer: {minHeight: 120, alignItems: 'center', justifyContent: 'center'},
  status: {fontFamily: fonts.display, fontSize: 30, color: colors.inkSoft},
  yourTurn: {color: colors.ink},
  observer: {alignItems: 'center', gap: 8},
  observerHint: {fontFamily: fonts.body, fontSize: 15, color: colors.inkSoft},
  observerRow: {flexDirection: 'row', gap: 16},
  observerBtn: {borderRadius: radius.pill, paddingHorizontal: 22, minHeight: 52, justifyContent: 'center'},
  observerText: {fontFamily: fonts.strong, fontSize: 18, color: '#fff'},
});
