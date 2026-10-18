/**
 * Live contract tests: the real module + HttpEngineClient against a running Grow21 backend.
 * Run with: GROW21_API_URL=http://127.0.0.1:8000 npm run test:integration   (from demo/)
 * Fails (does not skip) when the backend is unreachable. Uses synthetic learners only.
 */
import React from 'react';
import {act, fireEvent, render, screen, waitFor} from '@testing-library/react-native';
import {Grow21GameModule, HttpEngineClient} from '../../src';
import {GAME_NAMES, SKILLS} from '../../src/games';
import {MockHostBridge} from '../../src/MockHostBridge';
import type {GameId} from '../../src/types';

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

async function playCurrent(first: 'success' | 'failure') {
  if (first === 'failure') fireEvent.press(screen.getByText('Try again'));
  fireEvent.press(screen.getByText('I found it!'));
  await act(async () => {});
}

describe('frontend ↔ backend contract (live)', () => {
  it('frontend game→skill mapping matches the backend configuration for all nine games', async () => {
    const {status, body} = await getJson('/api/v1/config');
    expect(status).toBe(200);
    expect(Object.keys(body.games).sort()).toEqual(Object.keys(GAME_NAMES).sort());
    for (const gameId of Object.keys(GAME_NAMES) as GameId[]) expect(SKILLS[gameId]).toBe(body.games[gameId].primarySkill);
  });

  it('plays tasks end to end, applies backend difficulty, shares skill state, and honours the cooldown break', async () => {
    const child = unique('live');
    const {onExit, notify} = mount(child, 's1');

    await screen.findByText('Trace & Glow', {}, {timeout: 5000});
    let st = (await getJson(`/api/v1/learners/${child}/state`)).body;
    expect(st.outstandingTaskId).toBeTruthy();
    const traceD = st.lastDecision.applied_difficulty;
    expect(st.gameDifficulty.trace).toBeCloseTo(traceD, 3);

    await playCurrent('success'); // trace, fine_motor
    await screen.findByText('Bubble Pop Brigade', {}, {timeout: 5000});
    st = (await getJson(`/api/v1/learners/${child}/state`)).body;
    expect(st.observationsRecorded).toBe(1);
    expect(st.mastery.fine_motor).toBeGreaterThan(0.4541); // shared skill updated by Trace & Glow evidence
    expect(st.lastDecision.gameId).toBe('bubble');

    // Three tasks whose first attempt fails (even though later solved with help).
    await playCurrent('failure');
    await screen.findByText('Block Stacker', {}, {timeout: 5000});
    await playCurrent('failure');
    await screen.findByText('Echo Safari', {}, {timeout: 5000});
    await playCurrent('failure');

    await waitFor(() => expect(onExit).toHaveBeenCalledWith('break'), {timeout: 5000});
    expect(notify).toHaveBeenCalledWith({reason: 'frustration_cooldown', lengthSec: expect.any(Number)});
    st = (await getJson(`/api/v1/learners/${child}/state`)).body;
    expect(st).toMatchObject({observationsRecorded: 4, sessionTrialsObserved: 4, consecutiveFailures: 3, outstandingTaskId: null});
  });

  it('restores the outstanding task after the module is remounted with the same session', async () => {
    const child = unique('restore');
    const first = mount(child, 's1');
    await screen.findByText('Trace & Glow', {}, {timeout: 5000});
    const before = (await getJson(`/api/v1/learners/${child}/state`)).body;
    first.view.unmount();

    mount(child, 's1');
    await screen.findByText('Trace & Glow', {}, {timeout: 5000});
    const after = (await getJson(`/api/v1/learners/${child}/state`)).body;
    expect(after.outstandingTaskId).toBe(before.outstandingTaskId);
    expect(after.tasksIssued).toBe(1);
  });

  it('keeps two learners isolated', async () => {
    const a = unique('iso-a');
    const b = unique('iso-b');
    const ma = mount(a, 's1');
    await screen.findByText('Trace & Glow', {}, {timeout: 5000});
    await playCurrent('failure');
    await screen.findByText('Bubble Pop Brigade', {}, {timeout: 5000});
    ma.view.unmount();

    mount(b, 's1');
    await screen.findByText('Trace & Glow', {}, {timeout: 5000});
    await playCurrent('success');
    await screen.findByText('Bubble Pop Brigade', {}, {timeout: 5000});

    const sa = (await getJson(`/api/v1/learners/${a}/state`)).body;
    const sb = (await getJson(`/api/v1/learners/${b}/state`)).body;
    expect(sa.consecutiveFailures).toBe(1);
    expect(sb.consecutiveFailures).toBe(0);
    expect(sb.mastery.fine_motor).toBeGreaterThan(sa.mastery.fine_motor);
  });

  it('rejects malformed telemetry with a structured error', async () => {
    const res = await fetch(API + '/api/v1/telemetry/submit', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({schemaVersion: 1, gameId: 'chess'})});
    expect(res.status).toBe(422);
    expect((await res.json()).error.code).toBe('invalid_request');
  });
});
