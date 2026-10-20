import {buildSearchRound, describeOption} from '../src/boards/AudioSearch';
import {buildBlockLayout} from '../src/boards/BlockStacker';
import {bubbleSettings} from '../src/boards/BubblePop';
import {buildSortRound, sortLayout} from '../src/boards/CategorySort';
import {buildEchoRound} from '../src/boards/EchoSafari';
import {buildFlipRound} from '../src/boards/FlipMatch';
import {buildPatternRound} from '../src/boards/PatternTrain';
import {buildSentenceRound} from '../src/boards/SentenceStrips';
import {buildTracePath} from '../src/boards/TraceGlow';
import {applyScaffold, difficultyConfig} from '../src/games';
import type {GameId} from '../src/types';
import {isVoice, rms} from '../src/voice';
import {rankVoice} from '../src/speech';

const LEVELS = [0, 1, 2, 3];
const DS = [0, 0.25, 0.5, 0.75, 1];
const SEEDS = Array.from({length: 40}, (_, i) => 1000 + i * 7919);
const cfg = (g: GameId, D: number, level: number) => applyScaffold(g, difficultyConfig(g, D), level);
const each = (g: GameId, fn: (c: ReturnType<typeof cfg>, seed: number, D: number, level: number) => void) => {
  for (const D of DS) for (const level of LEVELS) for (const seed of SEEDS) fn(cfg(g, D, level), seed, D, level);
};

describe('round generators produce valid, solvable content at every difficulty and help level', () => {
  it('Pattern Train: the blank continues the pattern and appears exactly once among the choices', () => {
    each('pattern', (c, seed) => {
      const r = buildPatternRound(c, seed);
      const unit = r.sequence.slice(0, r.unitLength);
      r.sequence.forEach((item, i) => expect(item).toBe(unit[i % r.unitLength]));
      expect(r.choices[r.answerIndex]).toBe(r.answer);
      expect(r.choices.filter(x => x === r.answer)).toHaveLength(1);
      expect(new Set(r.choices).size).toBe(r.choices.length);
      expect(r.choices.length).toBe(c.answerChoices ? Number(c.answerChoices) : Math.min(4, Math.max(2, Number(c.distractorCount) + 1)));
    });
  });

  it('Audio Search: exactly one option matches the spoken description', () => {
    each('audio', (c, seed) => {
      const r = buildSearchRound(c, seed);
      expect(r.options).toHaveLength(Math.round(Number(c.optionCount)));
      const target = describeOption(r.options[r.answerIndex]);
      expect(r.instruction.text).toBe(`Find the ${target}`);
      expect(r.options.filter(o => describeOption(o) === target)).toHaveLength(1);
    });
  });

  it('Visual Flip Match: every picture appears exactly twice', () => {
    each('flip', (c, seed) => {
      const r = buildFlipRound(c, seed);
      expect(r.cards).toHaveLength(Number(c.cardCount) >= 6 ? 6 : 4);
      const counts = new Map<unknown, number>();
      r.cards.forEach(x => counts.set(x, (counts.get(x) ?? 0) + 1));
      expect([...counts.values()].every(n => n === 2)).toBe(true);
    });
  });

  it('Picture-Sentence Strips: every word of the sentence is available, distractors are not sentence words', () => {
    let shuffled = 0;
    each('sentence', (c, seed) => {
      const r = buildSentenceRound(c, seed);
      expect(r.tokens).toHaveLength(Math.max(2, Math.min(5, Math.round(Number(c.sentenceLength)))));
      const words = r.tokens.map(t => t.word);
      words.forEach(w => expect(r.choices.some(x => x.word === w)).toBe(true));
      expect(r.choices.length - r.tokens.length).toBe(Math.max(0, Math.min(4, Math.round(Number(c.distractorCount)))));
      if (r.choices.slice(0, r.tokens.length).map(x => x.word).join() !== words.join()) shuffled++;
    });
    expect(shuffled).toBeGreaterThan(DS.length * LEVELS.length * SEEDS.length * 0.6); // choices are not shown in answer order
  });

  it('Category Sorting: both baskets are needed and every item belongs to its side', () => {
    each('sort', (c, seed) => {
      const r = buildSortRound(c, seed);
      expect(r.items).toHaveLength(Math.max(3, Math.min(8, Math.round(Number(c.itemCount)))));
      expect(new Set(r.items.map(i => i.side))).toEqual(new Set([0, 1]));
      r.items.forEach(({item, side}) => expect((side === 0 ? r.pair.left : r.pair.right).items).toContain(item));
    });
    for (const [w, h] of [[976, 616], [343, 560], [1180, 700]]) {
      const l = sortLayout(w, h);
      l.bins.forEach(b => { expect(b.x).toBeGreaterThanOrEqual(0); expect(b.x + b.w).toBeLessThanOrEqual(w); expect(b.y + b.h).toBeLessThanOrEqual(h); });
      expect(l.item.y + l.item.size).toBeLessThanOrEqual(l.bins[0].y);
    }
  });

  it('Echo Safari: consecutive prompts use different animals', () => {
    each('echo', (c, seed) => {
      const r = buildEchoRound(c, seed);
      for (let i = 1; i < r.prompts.length; i++) expect(r.prompts[i][0].word).not.toBe(r.prompts[i - 1][r.prompts[i - 1].length - 1].word);
    });
  });

  it('Block Stacker: one target per block, all inside the board, layers bottom-up', () => {
    for (const [w, h, scale] of [[976, 616, 1.23], [343, 520, 1]]) {
      each('block', (c, seed) => {
        const l = buildBlockLayout(c, seed, w, h, scale);
        expect(l.targets).toHaveLength(l.blocks.length);
        expect(new Set(l.blocks.map(b => b.shape)).size).toBe(l.blocks.length);
        [...l.targets, ...l.tray].forEach(p => {
          expect(p.x - l.size / 2).toBeGreaterThanOrEqual(-1);
          expect(p.x + l.size / 2).toBeLessThanOrEqual(w + 1);
          expect(p.y - l.size / 2).toBeGreaterThanOrEqual(-1);
          expect(p.y + l.size / 2).toBeLessThanOrEqual(h + 1);
        });
        l.targets.forEach(t => l.targets.forEach(u => { if (u.layer > t.layer) expect(u.y).toBeLessThan(t.y); }));
      });
    }
  });

  it('Trace & Glow: the path stays inside the board and gets narrower as difficulty rises', () => {
    for (const [w, h] of [[976, 616], [343, 560]]) {
      each('trace', (c, seed) => {
        const p = buildTracePath(c, seed, w, h);
        const outside = p.points.filter(pt => pt.x < 0 || pt.x > w || pt.y < 0 || pt.y > h);
        const gaps = p.points.slice(1).filter((pt, i) => Math.hypot(pt.x - p.points[i].x, pt.y - p.points[i].y) > 6.01);
        expect([outside.length, gaps.length]).toEqual([0, 0]);
      });
    }
    const easy = buildTracePath(cfg('trace', 0, 0), 1, 976, 616).width;
    const hard = buildTracePath(cfg('trace', 1, 0), 1, 976, 616).width;
    expect(hard).toBeLessThan(easy);
  });

  it('Bubble Pop: bubbles shrink and speed up with difficulty but never drop below the minimum target', () => {
    const easy = bubbleSettings(cfg('bubble', 0, 0), 976, 616);
    const hard = bubbleSettings(cfg('bubble', 1, 0), 976, 616);
    expect(hard.size).toBeLessThan(easy.size);
    expect(hard.speed).toBeGreaterThan(easy.speed);
    expect(hard.size).toBeGreaterThanOrEqual(72);
    expect(bubbleSettings(cfg('bubble', 1, 3), 976, 616).concurrent).toBe(1);
  });
});

describe('voice activity rule (any vocalisation counts)', () => {
  it('needs sustained energy clearly above the room noise', () => {
    expect(isVoice([0.05, 0.06, 0.07], 0.01)).toBe(true);
    expect(isVoice([0.05, 0.004, 0.07], 0.01)).toBe(false); // a single click is not a voice
    expect(isVoice([0.025, 0.025, 0.025], 0.01)).toBe(false); // below 3x a noisy room
    expect(isVoice([0.03, 0.03], 0.001)).toBe(false); // too short
    expect(rms([0.5, -0.5, 0.5, -0.5])).toBeCloseTo(0.5);
  });
});

describe('device voice choice', () => {
  it('prefers natural English voices and never picks robotic ones', () => {
    const voices = [
      {name: 'English (espeak)', language: 'en'},
      {name: 'Microsoft Aria Online (Natural) - English (United States)', language: 'en-US'},
      {name: 'Google US English', language: 'en-US'},
      {name: 'Google Deutsch', language: 'de-DE'},
    ];
    const best = [...voices].sort((a, b) => rankVoice(b, 'en-IN') - rankVoice(a, 'en-IN'));
    expect(best[0].name).toContain('Natural');
    expect(best[best.length - 1].language).toBe('de-DE');
    expect(rankVoice(voices[0], 'en-IN')).toBeLessThan(rankVoice(voices[2], 'en-IN'));
  });
});
