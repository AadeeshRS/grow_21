import {GAME_NAMES} from '../games';
import type {EngineClient, GameId, NextTaskRequest, SkillId, TaskDirective, TelemetryEvent} from '../types';

/** Transport or protocol failure talking to the Grow21 learner service. `status` is absent for network errors/timeouts. */
export class EngineHttpError extends Error {
  constructor(message: string, readonly status?: number, readonly code?: string) {
    super(message);
    this.name = 'EngineHttpError';
  }
}

export interface HttpEngineOptions {
  /** Per-attempt timeout. SessionRunner applies its own overall deadline on top. */
  timeoutMs?: number;
  /** Extra attempts after a network error, timeout, or 5xx. Safe because the service is idempotent per eventId/outstanding task. */
  retries?: number;
  retryDelayMs?: number;
  fetchImpl?: typeof fetch;
}

const isRecord = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);
const isFiniteNumber = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const delay = (ms: number) => new Promise<void>(resolve => setTimeout(resolve, ms));

/** Checks that a response body is a well-formed TaskDirective before any game consumes it. */
export function parseTaskDirective(body: unknown): TaskDirective {
  const invalid = (why: string) => new EngineHttpError(`Invalid task directive: ${why}`);
  if (!isRecord(body) || body.schemaVersion !== 1) throw invalid('schemaVersion');
  const {task, break: brk, endSession} = body;
  if ([task, brk, endSession].filter(v => v !== undefined).length !== 1) throw invalid('expected exactly one of task, break, endSession');
  if (task !== undefined) {
    if (!isRecord(task)) throw invalid('task');
    if (typeof task.taskId !== 'string' || !task.taskId) throw invalid('task.taskId');
    if (typeof task.gameId !== 'string' || !(task.gameId in GAME_NAMES)) throw invalid('task.gameId');
    if (typeof task.skillId !== 'string') throw invalid('task.skillId');
    if (!isFiniteNumber(task.D) || task.D < 0 || task.D > 1) throw invalid('task.D');
    return {schemaVersion: 1, task: {taskId: task.taskId, gameId: task.gameId as GameId, skillId: task.skillId as SkillId, D: task.D}};
  }
  if (brk !== undefined) {
    if (!isRecord(brk) || typeof brk.reason !== 'string' || !isFiniteNumber(brk.lengthSec) || brk.lengthSec <= 0) throw invalid('break');
    return {schemaVersion: 1, break: {reason: brk.reason, lengthSec: brk.lengthSec}};
  }
  if (!isRecord(endSession) || typeof endSession.reason !== 'string') throw invalid('endSession');
  return {schemaVersion: 1, endSession: {reason: endSession.reason}};
}

/** EngineClient backed by the Grow21 Python learner service (docs/integration/FRONTEND_BACKEND_API.md in the backend branch). */
export class HttpEngineClient implements EngineClient {
  private readonly baseUrl: string;
  private readonly timeoutMs: number;
  private readonly retries: number;
  private readonly retryDelayMs: number;
  private readonly fetchImpl: typeof fetch;

  constructor(baseUrl: string, options: HttpEngineOptions = {}) {
    if (!/^https?:\/\/[^/]+/.test(baseUrl)) throw new Error(`Grow21 API URL must be an http(s) URL, got "${baseUrl}"`);
    this.baseUrl = baseUrl.replace(/\/+$/, '');
    this.timeoutMs = options.timeoutMs ?? 1000;
    this.retries = options.retries ?? 1;
    this.retryDelayMs = options.retryDelayMs ?? 150;
    this.fetchImpl = options.fetchImpl ?? ((input, init) => fetch(input, init));
  }

  async getNextTask(request: NextTaskRequest): Promise<TaskDirective> {
    return parseTaskDirective(await this.post('/api/v1/task/next', request));
  }

  async submitTelemetry(event: TelemetryEvent): Promise<{ack: true}> {
    const body = await this.post('/api/v1/telemetry/submit', event);
    if (!isRecord(body) || body.ack !== true) throw new EngineHttpError('Invalid telemetry acknowledgement');
    return {ack: true};
  }

  private async post(path: string, payload: unknown): Promise<unknown> {
    let lastError: unknown;
    for (let attempt = 0; attempt <= this.retries; attempt++) {
      if (attempt > 0) await delay(this.retryDelayMs);
      try {
        return await this.postOnce(path, payload);
      } catch (error) {
        lastError = error;
        const retryable = error instanceof EngineHttpError && (error.status === undefined || error.status >= 500);
        if (!retryable) throw error;
      }
    }
    throw lastError;
  }

  private async postOnce(path: string, payload: unknown): Promise<unknown> {
    const abort = new AbortController();
    const timer = setTimeout(() => abort.abort(), this.timeoutMs);
    try {
      const res = await this.fetchImpl(this.baseUrl + path, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload),
        signal: abort.signal,
      });
      const text = await res.text();
      let body: unknown;
      try {
        body = text ? JSON.parse(text) : undefined;
      } catch {
        throw new EngineHttpError(`Engine returned non-JSON (HTTP ${res.status})`, res.status);
      }
      if (!res.ok) {
        const err = isRecord(body) && isRecord(body.error) ? body.error : {};
        throw new EngineHttpError(typeof err.message === 'string' ? err.message : `HTTP ${res.status}`, res.status, typeof err.code === 'string' ? err.code : undefined);
      }
      return body;
    } catch (error) {
      if (error instanceof EngineHttpError) throw error;
      throw new EngineHttpError(abort.signal.aborted ? `Engine request timed out after ${this.timeoutMs} ms` : 'Engine unreachable');
    } finally {
      clearTimeout(timer);
    }
  }
}
