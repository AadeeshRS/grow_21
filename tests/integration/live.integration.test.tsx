/**
 * Live contract tests: the real module, real game boards and HttpEngineClient against a running
 * Grow21 backend. Run from demo/:  GROW21_API_URL=http://127.0.0.1:8000 npm run test:integration
 * Fails (does not skip) when the backend is unreachable. Synthetic learners only.
 */
import React from 'react';
import {render, screen} from '@testing-library/react-native';
import {Grow21GameModule, HttpEngineClient} from '../../src';
import {GAME_NAMES, SKILLS} from '../../src/games';
import {MockHostBridge} from '../../src/MockHostBridge';
import type {GameId} from '../../src/types';
import {advance, failEcho, failMotor, layoutStage, solve, timing} from '../solvers';

jest.mock('expo-speech', () => ({speak: jest.fn(), stop: jest.fn()}));
jest.setTimeout(120000);
timing.real = true;

const API = process.env.GROW21_API_URL;
if (!API) throw new Error('GROW21_API_URL must point at a running Grow21 backend');
const getJson = async (path: string) => {
  const res = await fetch(API + path);
  return {status: res.status, body: await res.json()};
};
const unique = (p: string) => `${p}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;

function mount(childId: string, sessionId: string) {
  const host = new MockHostBridge();
  const notify = jest.spyOn(host, 'notifyBreak');
  const onExit = jest.fn();
  const view = render(<Grow21GameModule childId={childId} sessionId={sessionId} locale="en-IN" mode="play" engine={new HttpEngineClient(API!)} host={host} onExit={onExit} />);
  return {onExit, notify, view};
}

/** Waits for the backend-issued task to appear and returns what the backend says it issued. */
async function current(child: string, expectGame: GameId) {
  await screen.findByText(new RegExp(GAME_NAMES[expectGame].replace(/[&]/g, '\\$&')), {}, {timeout: 8000});
  await layoutStage();
  const st = (await getJson(`/api/v1/learners/${child}/state`)).body;
  expect(st.lastDecision.gameId).toBe(expectGame);
  return {taskId: st.outstandingTaskId as string, D: st.lastDecision.applied_difficulty as number, state: st};
}

describe('frontend ↔ backend contract (live)', () => {
  it('frontend game→skill mapping matches the backend configuration for all nine games', async () => {
    const {status, body} = await getJson('/api/v1/config');
    expect(status).toBe(200);
    expect(Object.keys(body.games).sort()).toEqual(Object.keys(GAME_NAMES).sort());
    for (const gameId of Object.keys(GAME_NAMES) as GameId[]) expect(SKILLS[gameId]).toBe(body.games[gameId].primarySkill);
  });

  it('plays real games end to end: shared skill evidence, first-attempt failures, and the cooldown break', async () => {
    const child = unique('live');
    const {onExit, notify} = mount(child, 's1');

    // Trace & Glow solved on the first attempt: fine_motor evidence.
    let t = await current(child, 'trace');
    const before = t.state.mastery.fine_motor;
    await solve('trace', t.taskId, t.D);
    await advance(2500);

    // Bubble Pop shares fine_motor: its difficulty decision already used the trace evidence.
    t = await current(child, 'bubble');
    expect(t.state.observationsRecorded).toBe(1);
    expect(t.state.mastery.fine_motor).toBeGreaterThan(before);
    await failMotor('bubble', t.taskId, t.D);  // four taps on empty sky: first attempt fails
    await advance(1500);
    await solve('bubble', t.taskId, t.D);       // solved on the retry
    await advance(2500);

    t = await current(child, 'block');
    expect(t.state.consecutiveFailures).toBe(1); // the first attempt is the observation
    await failMotor('block', t.taskId, t.D);
    await advance(1500);
    await solve('block', t.taskId, t.D);
    await advance(2500);

    t = await current(child, 'echo');
    expect(t.state.consecutiveFailures).toBe(2);
    await failEcho();
    await advance(1500);
    await solve('echo', t.taskId, t.D);
    await advance(2500);

    for (let i = 0; i < 40 && !onExit.mock.calls.length; i++) await advance(250);
    expect(onExit).toHaveBeenCalledWith('break');
    expect(notify).toHaveBeenCalledWith({reason: 'frustration_cooldown', lengthSec: expect.any(Number)});
    const st = (await getJson(`/api/v1/learners/${child}/state`)).body;
    expect(st).toMatchObject({observationsRecorded: 4, sessionTrialsObserved: 4, consecutiveFailures: 3, outstandingTaskId: null});
  });

  it('restores the outstanding task after the module is remounted with the same session', async () => {
    const child = unique('restore');
    const first = mount(child, 's1');
    const t = await current(child, 'trace');
    first.view.unmount();
    mount(child, 's1');
    const again = await current(child, 'trace');
    expect(again.taskId).toBe(t.taskId);
    expect(again.state.tasksIssued).toBe(1);
  });

  it('keeps two learners isolated', async () => {
    const a = unique('iso-a');
    const b = unique('iso-b');
    const ma = mount(a, 's1');
    let t = await current(a, 'trace'); // a is issued a task but plays nothing
    ma.view.unmount();

    mount(b, 's1');
    t = await current(b, 'trace');
    await solve('trace', t.taskId, t.D);
    await advance(2500);
    await current(b, 'bubble');
    const sa = (await getJson(`/api/v1/learners/${a}/state`)).body;
    const sb = (await getJson(`/api/v1/learners/${b}/state`)).body;
    expect(sa.observationsRecorded).toBe(0);
    expect(sb.observationsRecorded).toBe(1);
    expect(sb.mastery.fine_motor).toBeGreaterThan(sa.mastery.fine_motor);
  });

  it('rejects malformed telemetry with a structured error', async () => {
    const res = await fetch(API + '/api/v1/telemetry/submit', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({schemaVersion: 1, gameId: 'chess'})});
    expect(res.status).toBe(422);
    expect((await res.json()).error.code).toBe('invalid_request');
  });
});

