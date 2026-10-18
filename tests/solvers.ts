/**
 * Test helpers that play each board through its real interaction surface (taps, drags, responder
 * events), using the same seeded generators as the boards so the right answers are known.
 */
import {act, fireEvent, screen} from '@testing-library/react-native';
import {buildSearchRound} from '../src/boards/AudioSearch';
import {buildBlockLayout} from '../src/boards/BlockStacker';
import {buildFlipRound} from '../src/boards/FlipMatch';
import {buildPatternRound} from '../src/boards/PatternTrain';
import {seedFromString} from '../src/boards/random';
import {buildSentenceRound} from '../src/boards/SentenceStrips';
import {buildSortRound} from '../src/boards/CategorySort';
import {buildTracePath} from '../src/boards/TraceGlow';
import {applyScaffold, difficultyConfig} from '../src/games';
import type {GameConfig, GameId} from '../src/types';

export const STAGE = {width: 1000, height: 640};
export const BOARD = {width: STAGE.width - 24, height: STAGE.height - 24, scale: Math.max(1, Math.min(1.6, Math.min(STAGE.width, STAGE.height) / 520))};

/** Fake timers in unit tests; real waiting in the live integration suite (set `timing.real = true`). */
export const timing = {real: false};
export const advance = async (ms: number) => {
  await act(async () => {
    if (timing.real) await new Promise(r => setTimeout(r, ms));
    else jest.advanceTimersByTime(ms);
  });
};

/** Gives the play area a size so the board renders (layout does not run in the test renderer). */
export async function layoutStage() {
  await act(async () => {
    fireEvent(screen.getByTestId('game-stage'), 'layout', {nativeEvent: {layout: STAGE}});
  });
}

export const displayConfig = (gameId: GameId, D: number, level = 0): GameConfig => applyScaffold(gameId, difficultyConfig(gameId, D), level);

const press = async (testID: string) => { await act(async () => { fireEvent.press(screen.getByTestId(testID)); }); };

const responder = async (testID: string, events: Array<['responderGrant' | 'responderMove' | 'responderRelease', {pageX: number; pageY: number; locationX?: number; locationY?: number}]>) => {
  for (const [type, e] of events) {
    await act(async () => { fireEvent(screen.getByTestId(testID), type, {nativeEvent: {locationX: 0, locationY: 0, ...e}}); });
  }
};

/** Completes the current attempt successfully. */
export async function solve(gameId: GameId, taskId: string, D: number, level = 0) {
  const config = displayConfig(gameId, D, level);
  const seed = seedFromString(taskId);
  switch (gameId) {
    case 'pattern': return press(`pattern-choice-${buildPatternRound(config, seed).answerIndex}`);
    case 'audio': return press(`search-option-${buildSearchRound(config, seed).answerIndex}`);
    case 'flip': {
      const {cards} = buildFlipRound(config, seed);
      await advance(Number(config.exposureMs) + 50);
      const done = new Set<number>();
      for (let i = 0; i < cards.length; i++) {
        if (done.has(i)) continue;
        const j = cards.findIndex((c, k) => k !== i && c === cards[i]);
        done.add(i); done.add(j);
        await press(`flip-card-${i}`);
        await press(`flip-card-${j}`);
        await advance(400);
      }
      return;
    }
    case 'sentence': {
      const {tokens, choices} = buildSentenceRound(config, seed);
      const used = new Set<number>();
      for (const t of tokens) {
        const i = choices.findIndex((c, k) => !used.has(k) && c.word === t.word);
        used.add(i);
        await press(`sentence-choice-${i}`);
      }
      return;
    }
    case 'sort': {
      const {items} = buildSortRound(config, seed);
      for (const it of items) { await press(`sort-bin-${it.side}`); await advance(10); }
      return;
    }
    case 'echo': {
      const prompts = Number(config.repetitions);
      for (let p = 0; p < prompts; p++) {
        await advance(12000); // the model sound plays, then the response window opens
        await press('echo-observer-yes');
        await advance(1000);
      }
      return;
    }
    case 'bubble': {
      for (let i = 0; i < Number(config.popTarget); i++) {
        const bubble = screen.getAllByTestId(/^bubble-\d+$/)[0];
        await act(async () => { fireEvent.press(bubble); });
        await advance(20);
      }
      return;
    }
    case 'block': {
      const layout = buildBlockLayout(config, seed, BOARD.width, BOARD.height, BOARD.scale);
      const order = layout.targets.map((t, i) => ({i, layer: t.layer})).sort((a, b) => a.layer - b.layer).map(x => x.i);
      for (const b of order) {
        const at = layout.tray[layout.trayOrder.indexOf(b)];
        const t = layout.targets[b];
        await responder(`block-${b}`, [['responderGrant', {pageX: 0, pageY: 0}], ['responderMove', {pageX: (t.x - at.x) / 2, pageY: (t.y - at.y) / 2}], ['responderRelease', {pageX: t.x - at.x, pageY: t.y - at.y}]]);
      }
      return;
    }
    case 'trace': {
      const {points} = buildTracePath(config, seed, BOARD.width, BOARD.height, BOARD.scale);
      const p0 = points[0];
      const events: Parameters<typeof responder>[1] = [['responderGrant', {pageX: p0.x, pageY: p0.y, locationX: p0.x, locationY: p0.y}]];
      for (let i = 2; i < points.length; i += 2) events.push(['responderMove', {pageX: points[i].x, pageY: points[i].y}]);
      const last = points[points.length - 1];
      events.push(['responderMove', {pageX: last.x, pageY: last.y}], ['responderRelease', {pageX: last.x, pageY: last.y}]);
      return responder('trace-board', events);
    }
  }
}

/** Ends an attempt through genuine misses: empty-sky taps (bubble) or drops far from any space (block). */
export async function failMotor(gameId: 'bubble' | 'block', taskId: string, D: number, level = 0) {
  if (gameId === 'bubble') {
    for (let i = 0; i < 4; i++) await act(async () => { fireEvent.press(screen.getByTestId('bubble-sky'), {nativeEvent: {locationX: 5, locationY: 5}}); });
  } else {
    const layout = buildBlockLayout(displayConfig('block', D, level), seedFromString(taskId), BOARD.width, BOARD.height, BOARD.scale);
    const b = layout.trayOrder[0];
    for (let i = 0; i < 3; i++) await responder(`block-${b}`, [['responderGrant', {pageX: 0, pageY: 0}], ['responderRelease', {pageX: 0, pageY: -BOARD.height}]]);
  }
  await advance(800);
}

/** Ends the current attempt unsuccessfully through a genuine wrong response (single-choice games only). */
export async function failChoice(gameId: 'pattern' | 'audio', taskId: string, D: number, level = 0) {
  const config = displayConfig(gameId, D, level);
  const seed = seedFromString(taskId);
  const round = gameId === 'pattern' ? buildPatternRound(config, seed) : buildSearchRound(config, seed);
  const wrong = round.answerIndex === 0 ? 1 : 0;
  await press(gameId === 'pattern' ? `pattern-choice-${wrong}` : `search-option-${wrong}`);
  await advance(800);
}

/** Echo Safari: the grown-up reports that the child has not made the sound yet. */
export async function failEcho() {
  await advance(12000);
  await press('echo-observer-no');
  await advance(800);
}
