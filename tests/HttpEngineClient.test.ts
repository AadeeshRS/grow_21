import {EngineHttpError, HttpEngineClient, parseTaskDirective} from '../src/api/HttpEngineClient';
import {SessionRunner} from '../src/SessionRunner';
import {MockHostBridge} from '../src/MockHostBridge';
import type {NextTaskRequest, TelemetryEvent} from '../src/types';

type Call = {url: string; init: RequestInit};
const json = (status: number, body: unknown) => ({ok: status >= 200 && status < 300, status, text: async () => JSON.stringify(body)}) as unknown as Response;

function fakeFetch(responses: Array<Response | 'hang' | 'network'>) {
  const calls: Call[] = [];
  const impl = (async (url: string, init: RequestInit) => {
    calls.push({url, init});
    const next = responses.shift();
    if (next === 'network') throw new TypeError('Network request failed');
    if (next === 'hang' || next === undefined) {
      return new Promise<Response>((_, reject) => init.signal?.addEventListener('abort', () => reject(new Error('aborted'))));
    }
    return next;
  }) as unknown as typeof fetch;
  return {impl, calls};
}

const request: NextTaskRequest = {schemaVersion: 1, childId: 'kid1', sessionId: 's1', mode: 'play', sessionActiveMs: 0, tasksCompletedThisSession: 0, availability: {microphone: true, affectSignal: false}};
const task = {schemaVersion: 1, task: {taskId: 't1', gameId: 'trace', skillId: 'fine_motor', D: 0.37}};
const event = {schemaVersion: 1, eventId: 'e1', taskId: 't1'} as unknown as TelemetryEvent;

describe('HttpEngineClient', () => {
  it('posts the request as JSON to the next-task endpoint and parses the directive', async () => {
    const {impl, calls} = fakeFetch([json(200, task)]);
    const client = new HttpEngineClient('http://localhost:8000/', {fetchImpl: impl});
    await expect(client.getNextTask(request)).resolves.toEqual(task);
    expect(calls).toHaveLength(1);
    expect(calls[0].url).toBe('http://localhost:8000/api/v1/task/next');
    expect(calls[0].init.method).toBe('POST');
    expect(calls[0].init.headers).toEqual({'Content-Type': 'application/json'});
    expect(JSON.parse(String(calls[0].init.body))).toEqual(request);
  });

  it('rejects malformed base URLs', () => {
    expect(() => new HttpEngineClient('localhost:8000')).toThrow(/http\(s\) URL/);
  });

  it('retries a timed-out request and recovers', async () => {
    const {impl, calls} = fakeFetch(['hang', json(200, task)]);
    const client = new HttpEngineClient('http://h', {fetchImpl: impl, timeoutMs: 20, retryDelayMs: 1});
    await expect(client.getNextTask(request)).resolves.toEqual(task);
    expect(calls).toHaveLength(2);
    expect(calls[1].init.body).toBe(calls[0].init.body);
  });

  it('gives up after the configured retries with a timeout error', async () => {
    const {impl, calls} = fakeFetch(['hang', 'hang']);
    const client = new HttpEngineClient('http://h', {fetchImpl: impl, timeoutMs: 10, retries: 1, retryDelayMs: 1});
    await expect(client.getNextTask(request)).rejects.toThrow(/timed out/);
    expect(calls).toHaveLength(2);
  });

  it('retries network errors and 5xx but not 4xx, surfacing service error codes', async () => {
    const retried = fakeFetch(['network', json(503, {error: {code: 'down', message: 'down'}}), json(200, {ack: true, duplicate: false, observationApplied: true})]);
    const client = new HttpEngineClient('http://h', {fetchImpl: retried.impl, retries: 2, retryDelayMs: 1});
    await expect(client.submitTelemetry(event)).resolves.toEqual({ack: true});
    expect(retried.calls).toHaveLength(3);
    // Each retry carries the same eventId, which the service deduplicates.
    expect(new Set(retried.calls.map(c => c.init.body)).size).toBe(1);

    const rejected = fakeFetch([json(409, {error: {code: 'task_already_observed', message: 'This task already has a recorded event.'}})]);
    const c2 = new HttpEngineClient('http://h', {fetchImpl: rejected.impl, retries: 2});
    const err = await c2.submitTelemetry(event).catch(e => e);
    expect(err).toBeInstanceOf(EngineHttpError);
    expect(err).toMatchObject({status: 409, code: 'task_already_observed'});
    expect(rejected.calls).toHaveLength(1);
  });

  it('rejects non-JSON and unacknowledged telemetry responses', async () => {
    const nonJson = {ok: true, status: 200, text: async () => '<html>'} as unknown as Response;
    await expect(new HttpEngineClient('http://h', {fetchImpl: fakeFetch([nonJson]).impl, retries: 0}).getNextTask(request)).rejects.toThrow(/non-JSON/);
    await expect(new HttpEngineClient('http://h', {fetchImpl: fakeFetch([json(200, {ok: 1})]).impl, retries: 0}).submitTelemetry(event)).rejects.toThrow(/acknowledgement/);
  });
});

describe('parseTaskDirective', () => {
  it.each([
    [{schemaVersion: 1, break: {reason: 'frustration_cooldown', lengthSec: 7200}}],
    [{schemaVersion: 1, endSession: {reason: 'session_trial_limit'}}],
    [task],
  ])('accepts %j', body => expect(parseTaskDirective(body)).toEqual(body));

  it.each([
    ['wrong schema', {schemaVersion: 2, endSession: {reason: 'x'}}],
    ['two shapes', {schemaVersion: 1, endSession: {reason: 'x'}, break: {reason: 'y', lengthSec: 1}}],
    ['no shape', {schemaVersion: 1}],
    ['missing taskId', {schemaVersion: 1, task: {gameId: 'trace', skillId: 'fine_motor', D: 0.3}}],
    ['unknown game', {schemaVersion: 1, task: {taskId: 't', gameId: 'chess', skillId: 'fine_motor', D: 0.3}}],
    ['D out of range', {schemaVersion: 1, task: {taskId: 't', gameId: 'trace', skillId: 'fine_motor', D: 1.3}}],
    ['bad break', {schemaVersion: 1, break: {reason: 'x', lengthSec: 0}}],
  ])('rejects %s', (_, body) => expect(() => parseTaskDirective(body)).toThrow(EngineHttpError));
});

describe('SessionRunner with HttpEngineClient', () => {
  it('rejects a directive whose skill does not match the game mapping', async () => {
    const wrong = {schemaVersion: 1, task: {taskId: 't1', gameId: 'trace', skillId: 'working_memory', D: 0.3}};
    const client = new HttpEngineClient('http://h', {fetchImpl: fakeFetch([json(200, wrong)]).impl});
    const runner = new SessionRunner(client, new MockHostBridge(), 'kid1', 's1', 'play');
    await expect(runner.next()).rejects.toThrow(/mismatch/);
  });

  it('re-serves the last issued task when the engine times out, then recovers', async () => {
    const second = {schemaVersion: 1, task: {taskId: 't2', gameId: 'bubble', skillId: 'fine_motor', D: 0.33}};
    const {impl} = fakeFetch([json(200, task), 'hang', 'hang', json(200, second)]);
    const client = new HttpEngineClient('http://h', {fetchImpl: impl, timeoutMs: 10, retries: 1, retryDelayMs: 1});
    const runner = new SessionRunner(client, new MockHostBridge(), 'kid1', 's1', 'play');
    await expect(runner.next()).resolves.toEqual(task);
    // Engine unavailable: the same task (same taskId) is repeated; the backend will not count it twice.
    await expect(runner.next()).resolves.toEqual(task);
    await expect(runner.next()).resolves.toEqual(second);
  });
});
