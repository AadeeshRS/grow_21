import React from 'react';
import {act, fireEvent, render, screen, waitFor} from '@testing-library/react-native';
import {Grow21GameModule} from '../src';
import {GameTask} from '../src/GameTask';
import {GAME_NAMES, SKILLS, difficultyConfig} from '../src/games';
import {MockHostBridge} from '../src/MockHostBridge';
import type {EngineClient, GameId, NextTaskRequest, TaskDirective, TelemetryEvent} from '../src/types';

type Pending<T> = {promise: Promise<T>; resolve: (v: T) => void; reject: (e: unknown) => void};
const deferred = <T,>(): Pending<T> => {
  let resolve!: (v: T) => void, reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej; });
  return {promise, resolve, reject};
};
const taskFor = (gameId: GameId, D: number, taskId = `t-${gameId}`): TaskDirective => ({schemaVersion: 1, task: {taskId, gameId, skillId: SKILLS[gameId], D}});

/** Engine double that records traffic; directives may be values or deferred promises. */
class ScriptedEngine implements EngineClient {
  requests: NextTaskRequest[] = [];
  events: TelemetryEvent[] = [];
  constructor(private script: Array<TaskDirective | Pending<TaskDirective> | Error>, private acks: Array<Pending<{ack: true}>> = []) {}
  async getNextTask(request: NextTaskRequest): Promise<TaskDirective> {
    this.requests.push(request);
    const next = this.script.shift();
    if (!next) return {schemaVersion: 1, endSession: {reason: 'script_exhausted'}};
    if (next instanceof Error) throw next;
    return 'promise' in next ? next.promise : next;
  }
  async submitTelemetry(event: TelemetryEvent) {
    this.events.push(event);
    const ack = this.acks.shift();
    return ack ? ack.promise : {ack: true as const};
  }
}

const mount = (engine: EngineClient, host = new MockHostBridge(), onExit = jest.fn()) => {
  render(<Grow21GameModule childId="kid1" sessionId="s1" locale="en-IN" mode="play" engine={engine} host={host} onExit={onExit} />);
  return onExit;
};

describe('Grow21GameModule with a backend engine', () => {
  it('shows a loading state while the backend decides, then renders the game at the returned difficulty', async () => {
    const first = deferred<TaskDirective>();
    mount(new ScriptedEngine([first]));
    expect(screen.getByText('Getting the next activity…')).toBeTruthy();
    await act(async () => first.resolve(taskFor('trace', 0.37)));
    expect(screen.getByText('Trace & Glow')).toBeTruthy();
    const cfg = difficultyConfig('trace', 0.37);
    expect(screen.getByText(`Help level 0 · pathWidthDp: ${Math.round(Number(cfg.pathWidthDp))} · pathLengthDp: ${Math.round(Number(cfg.pathLengthDp))} · curvature: ${Math.round(Number(cfg.curvature))}`)).toBeTruthy();
  });

  it('submits one event per task carrying the issued taskId and difficulty, with fresh state for the next task', async () => {
    const ack = deferred<{ack: true}>();
    const engine = new ScriptedEngine([taskFor('trace', 0.37, 'task-1'), taskFor('bubble', 0.33, 'task-2')], [ack]);
    mount(engine);
    await screen.findByText('Trace & Glow');

    const found = screen.getByText('I found it!');
    fireEvent.press(found);
    fireEvent.press(found); // accidental double tap in the same frame
    // While the backend acknowledges, the game is replaced by the loading state (no further taps possible).
    expect(screen.getByText('Getting the next activity…')).toBeTruthy();
    await act(async () => ack.resolve({ack: true}));
    await screen.findByText('Bubble Pop Brigade');
    expect(engine.events).toHaveLength(1);

    const [e1] = engine.events;
    expect(e1).toMatchObject({childId: 'kid1', sessionId: 's1', taskId: 'task-1', gameId: 'trace', skillId: 'fine_motor', difficulty: 0.37, outcome: 'success', scaffoldStartLevel: 0, taskIndexInSession: 0, retries: 0});
    expect(e1.attempts).toEqual([expect.objectContaining({attemptIndex: 0, scaffoldLevel: 0, outcome: 'success'})]);

    fireEvent.press(screen.getByText('Try again'));
    fireEvent.press(screen.getByText('Try again'));
    fireEvent.press(screen.getByText('I found it!'));
    await waitFor(() => expect(engine.events).toHaveLength(2));
    const e2 = engine.events[1];
    expect(e2).toMatchObject({taskId: 'task-2', gameId: 'bubble', difficulty: 0.33, outcome: 'success', assisted: true, taskIndexInSession: 1, retries: 2});
    expect(e2.attempts.map(a => [a.outcome, a.scaffoldLevel])).toEqual([['failure', 0], ['failure', 0], ['success', 1]]);
    expect(engine.requests.map(r => r.tasksCompletedThisSession)).toEqual([0, 1, 2]);
  });

  it.each(Object.keys(GAME_NAMES) as GameId[])('maps %s to its configured skill in telemetry', async gameId => {
    const engine = new ScriptedEngine([taskFor(gameId, 0.5)]);
    mount(engine);
    await screen.findByText(GAME_NAMES[gameId]);
    fireEvent.press(screen.getByText('I found it!'));
    await waitFor(() => expect(engine.events).toHaveLength(1));
    expect(engine.events[0]).toMatchObject({gameId, skillId: SKILLS[gameId], difficulty: 0.5, gameConfig: difficultyConfig(gameId, 0.5)});
  });

  it('honours a backend break directive and reports it to the host', async () => {
    const host = new MockHostBridge();
    const notify = jest.spyOn(host, 'notifyBreak');
    const onExit = mount(new ScriptedEngine([{schemaVersion: 1, break: {reason: 'frustration_cooldown', lengthSec: 7200}}]), host);
    await waitFor(() => expect(onExit).toHaveBeenCalledWith('break'));
    expect(notify).toHaveBeenCalledWith({reason: 'frustration_cooldown', lengthSec: 7200});
  });

  it('ends the session when the backend says so', async () => {
    const onExit = mount(new ScriptedEngine([{schemaVersion: 1, endSession: {reason: 'session_trial_limit'}}]));
    await waitFor(() => expect(onExit).toHaveBeenCalledWith('completed'));
  });

  it('shows the caregiver notice instead of a mock task when the backend fails', async () => {
    const onExit = mount(new ScriptedEngine([new Error('Engine unreachable')]));
    await screen.findByText('We need a moment. Please return to your caregiver.');
    expect(onExit).toHaveBeenCalledWith('error');
    expect(screen.queryByText('I found it!')).toBeNull();
  });
});

describe('GameTask', () => {
  it('emits a task event at most once even if the child keeps tapping', () => {
    const onComplete = jest.fn();
    render(<GameTask gameId="pattern" skillId="sequential_reasoning" taskId="t1" difficulty={0.4} config={difficultyConfig('pattern', 0.4)} childId="kid1" sessionId="s1" index={0} onComplete={onComplete} />);
    fireEvent.press(screen.getByText('I found it!'));
    fireEvent.press(screen.getByText('I found it!'));
    fireEvent.press(screen.getByText('Try again'));
    expect(onComplete).toHaveBeenCalledTimes(1);
    expect(onComplete.mock.calls[0][0]).toMatchObject({taskId: 't1', attempts: [expect.objectContaining({outcome: 'success'})]});
  });
});
