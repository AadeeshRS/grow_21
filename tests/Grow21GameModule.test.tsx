import React from 'react';
import {act, fireEvent, render, screen} from '@testing-library/react-native';
import * as Speech from 'expo-speech';
import {Grow21GameModule} from '../src';
import {GAME_NAMES, SKILLS} from '../src/games';
import {MockHostBridge} from '../src/MockHostBridge';
import type {EngineClient, GameId, NextTaskRequest, TaskDirective, TelemetryEvent} from '../src/types';
import {advance, displayConfig, failChoice, layoutStage, solve} from './solvers';

jest.mock('expo-speech', () => ({speak: jest.fn(), stop: jest.fn()}));

type Pending<T> = {promise: Promise<T>; resolve: (v: T) => void};
const deferred = <T,>(): Pending<T> => { let resolve!: (v: T) => void; const promise = new Promise<T>(r => { resolve = r; }); return {promise, resolve}; };
const taskFor = (gameId: GameId, D: number, taskId = `t-${gameId}`): TaskDirective => ({schemaVersion: 1, task: {taskId, gameId, skillId: SKILLS[gameId], D}});

/** Engine double that records traffic; directives may be values or deferred promises. */
class ScriptedEngine implements EngineClient {
  requests: NextTaskRequest[] = [];
  events: TelemetryEvent[] = [];
  constructor(private script: Array<TaskDirective | Pending<TaskDirective> | Error>) {}
  async getNextTask(request: NextTaskRequest): Promise<TaskDirective> {
    this.requests.push(request);
    const next = this.script.shift();
    if (!next) return {schemaVersion: 1, endSession: {reason: 'script_exhausted'}};
    if (next instanceof Error) throw next;
    return 'promise' in next ? next.promise : next;
  }
  async submitTelemetry(event: TelemetryEvent) {
    this.events.push(event);
    return {ack: true as const};
  }
}

const mount = (engine: EngineClient, host = new MockHostBridge(), onExit = jest.fn()) => {
  render(<Grow21GameModule childId="kid1" sessionId="s1" locale="en-IN" mode="play" engine={engine} host={host} onExit={onExit} />);
  return onExit;
};

/** Mounts one task, waits for it, and gives the stage a size. */
async function start(gameId: GameId, D: number, extra: TaskDirective[] = []) {
  const engine = new ScriptedEngine([taskFor(gameId, D), ...extra]);
  const onExit = mount(engine);
  await advance(0);
  await layoutStage();
  return {engine, onExit};
}

beforeEach(() => { jest.useFakeTimers(); (Speech.speak as jest.Mock).mockClear(); });
afterEach(() => { jest.clearAllTimers(); jest.useRealTimers(); });

describe('every game is playable through its real interaction and reports correct telemetry', () => {
  const D = 0.5;
  it.each(Object.keys(GAME_NAMES) as GameId[])('%s', async gameId => {
    const {engine} = await start(gameId, D);
    await solve(gameId, `t-${gameId}`, D);
    await advance(4000); // celebration, then the event is emitted
    expect(engine.events).toHaveLength(1);
    const e = engine.events[0];
    expect(e).toMatchObject({gameId, skillId: SKILLS[gameId], taskId: `t-${gameId}`, difficulty: D, outcome: 'success', scaffoldStartLevel: 0, scaffoldFinalLevel: 0, assisted: false, retries: 0, gameConfig: displayConfig(gameId, D)});
    expect(e.attempts).toHaveLength(1);
    expect(e.attempts[0]).toMatchObject({attemptIndex: 0, scaffoldLevel: 0, outcome: 'success'});
    expect(e.tFirstResponseMs).toBeGreaterThanOrEqual(e.tPresentedMs);
  });
});

describe('Grow21GameModule flow', () => {
  it('shows a calm loading state while the backend decides, then speaks the instruction', async () => {
    const first = deferred<TaskDirective>();
    mount(new ScriptedEngine([first]));
    expect(screen.getByText('Get ready…')).toBeTruthy();
    await act(async () => first.resolve(taskFor('pattern', 0.4)));
    expect(screen.getByText('What comes next?')).toBeTruthy();
    expect(Speech.speak).toHaveBeenCalledWith('What comes next? Finish the train.', expect.anything());
  });

  it('retries gently, escalates help after two misses, and reports an exhausted task as a failure', async () => {
    const {engine} = await start('pattern', 0.5);
    for (let attempt = 0; attempt < 8; attempt++) {
      const level = Math.min(3, Math.floor(attempt / 2));
      await failChoice('pattern', 't-pattern', 0.5, level);
      if (attempt === 0) expect(screen.getByText("Let's try again.")).toBeTruthy();
      if (attempt === 1) expect(screen.getByText("Let's try together. Here is a little help.")).toBeTruthy();
      await advance(2000);
      if (attempt === 0) expect(screen.queryByText('💡 Help is on')).toBeNull();
      if (attempt === 1) expect(screen.getByText('💡 Help is on')).toBeTruthy();
    }
    await advance(4000);
    expect(engine.events).toHaveLength(1);
    const e = engine.events[0];
    expect(e).toMatchObject({outcome: 'failure', scaffoldExhausted: true, scaffoldFinalLevel: 3, assisted: true, retries: 7});
    expect(e.attempts.map(a => a.scaffoldLevel)).toEqual([0, 0, 1, 1, 2, 2, 3, 3]);
    expect(e.attempts.every(a => a.outcome === 'failure')).toBe(true);
  });

  it('a task solved after a first-attempt miss keeps that miss as attempt 0', async () => {
    const {engine} = await start('audio', 0.6);
    await failChoice('audio', 't-audio', 0.6);
    await advance(1500);
    await solve('audio', 't-audio', 0.6);
    await advance(4000);
    expect(engine.events[0].attempts.map(a => a.outcome)).toEqual(['failure', 'success']);
    expect(engine.events[0].outcome).toBe('success');
  });

  it('keeps sorting placement misses separate from classification errors', async () => {
    const {engine} = await start('sort', 0.2);
    for (let k = 0; k < 4; k++) {
      // A drag released far from both baskets: a placement miss, never a failed attempt.
      await act(async () => { fireEvent(screen.getByTestId('sort-item'), 'responderGrant', {nativeEvent: {pageX: 0, pageY: 0, locationX: 0, locationY: 0}}); });
      await act(async () => { fireEvent(screen.getByTestId('sort-item'), 'responderRelease', {nativeEvent: {pageX: 400, pageY: -40}}); });
    }
    expect(screen.queryByText("Let's try again.")).toBeNull();
    await solve('sort', 't-sort', 0.2);
    await advance(4000);
    expect(engine.events[0].attempts[0].task).toMatchObject({placementMisses: 4, classificationErrors: 0});
  });

  it('counts stars, requests the next task, and gives each task fresh state', async () => {
    const {engine} = await start('pattern', 0.3, [taskFor('audio', 0.3, 'task-2')]);
    await solve('pattern', 't-pattern', 0.3);
    await advance(4000);
    expect(screen.getByTestId('star-count').props.children).toBe(1);
    await layoutStage();
    await solve('audio', 'task-2', 0.3);
    await advance(4000);
    expect(engine.events.map(e => [e.taskId, e.taskIndexInSession, e.attempts.length])).toEqual([['t-pattern', 0, 1], ['task-2', 1, 1]]);
    expect(engine.requests.map(r => r.tasksCompletedThisSession)).toEqual([0, 1, 2]);
  });

  it('repeats the instruction after inactivity without counting it as a failure', async () => {
    const {engine} = await start('audio', 0.4);
    (Speech.speak as jest.Mock).mockClear();
    await advance(11500);
    expect(Speech.speak).toHaveBeenCalled();
    expect(engine.events).toHaveLength(0);
    expect(screen.queryByText("Let's try again.")).toBeNull();
  });

  it('lets a grown-up leave only by holding the lock', async () => {
    const {onExit} = await start('pattern', 0.4);
    await act(async () => { fireEvent(screen.getByTestId('grown-ups-exit'), 'pressIn'); });
    await advance(400);
    await act(async () => { fireEvent(screen.getByTestId('grown-ups-exit'), 'pressOut'); });
    await advance(2000);
    expect(onExit).not.toHaveBeenCalled();
    await act(async () => { fireEvent(screen.getByTestId('grown-ups-exit'), 'pressIn'); });
    await advance(1300);
    expect(onExit).toHaveBeenCalledWith('userExit');
  });

  it('honours a backend break directive and reports it to the host', async () => {
    const host = new MockHostBridge();
    const notify = jest.spyOn(host, 'notifyBreak');
    const onExit = mount(new ScriptedEngine([{schemaVersion: 1, break: {reason: 'frustration_cooldown', lengthSec: 7200}}]), host);
    await advance(0);
    expect(onExit).toHaveBeenCalledWith('break');
    expect(notify).toHaveBeenCalledWith({reason: 'frustration_cooldown', lengthSec: 7200});
  });

  it('ends the session when the backend says so', async () => {
    const onExit = mount(new ScriptedEngine([{schemaVersion: 1, endSession: {reason: 'session_trial_limit'}}]));
    await advance(0);
    expect(onExit).toHaveBeenCalledWith('completed');
  });

  it('asks for a grown-up instead of inventing a task when the backend fails', async () => {
    const onExit = mount(new ScriptedEngine([new Error('Engine unreachable')]));
    await advance(0);
    expect(screen.getByText("Let's ask a grown-up for help.")).toBeTruthy();
    expect(onExit).toHaveBeenCalledWith('error');
    expect(screen.queryByTestId('game-stage')).toBeNull();
  });
});
