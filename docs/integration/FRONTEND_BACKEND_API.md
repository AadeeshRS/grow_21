# Grow21 Frontend–Backend Integration: Architecture, API Contract, and Validation

**Status:** Implemented and tested with synthetic events (October 18, 2026)
**Backend branch:** `grow21-integration` (based on `model-testing` @ `5c4682f`, which already contains the estimator-robustness work)
**Frontend branch:** `grow21-integration-ui` (based on `ui-game` @ `e695e63`)
**Selected estimator:** `RobustAdaptiveFilterController` (`docs/research/phase_5/ESTIMATOR_ROBUSTNESS_EXTENSION.md`)

> [!IMPORTANT]
> All testing used synthetic events only. Passing tests show that the software behaves as specified;
> they are not evidence of clinical effectiveness, educational efficacy, or accurate mastery estimation
> for real children. Mastery values are internal simulation-scale estimates.

---

## 1. Architecture and state ownership

```
React Native module (ui-game)                         Python service (this repo)
┌──────────────────────────────────┐   HTTP/JSON   ┌───────────────────────────────────────┐
│ GameTask: renders task, records  │               │ service/app.py      FastAPI transport │
│   attempts, builds TelemetryEvent│               │   validates schemas, maps errors      │
│ SessionRunner: timeouts,         │──next-task──▶ │ service/runtime.py  LearnerRuntime    │
│   directive validation           │◀─directive──  │   owns per-learner state, sessions,   │
│ HttpEngineClient: transport,     │──telemetry──▶ │   task protocol, idempotency          │
│   retries, response validation   │◀────ack─────  │ closed_loop.py RobustAdaptiveFilter-  │
│ games.ts ψ_g(D): D → on-screen   │               │   Controller: mastery + difficulty    │
│   parameters (frontend-owned)    │               │ outcome_model.py canonical IRT        │
└──────────────────────────────────┘               └───────────────────────────────────────┘
```

| Concern | Owner |
|---|---|
| Mastery estimation (robust adaptive filter, forgetting, transfer, reliability gating) | Backend: `RobustAdaptiveFilterController`, unchanged equations |
| Difficulty recommendation (bisection to p* = 0.75, deadband, step limit, bounds, safety drop) | Backend: same controller |
| Session trial limit, frustration cooldown pause | Backend runtime, using the controller's configured limits |
| Which game is next | Backend: the controller's existing configured round-robin over the nine games |
| Mapping abstract D to concrete game parameters (ψ_g) and in-task scaffold ladder | Frontend: `src/games.ts`, `src/ScaffoldController.ts` (unchanged) |
| Rendering, attempt recording, timing | Frontend |

The frontend contains no estimator or difficulty controller. `MockEngine` remains for offline demos only, and the demo host labels mock mode visibly. When the backend is configured and fails, the module shows the caregiver notice; it never falls back to mock decisions.

### 1.1 Why a thin FastAPI service

No inference service existed. The reusable model is Python, and the controller holds a numeric belief grid. A small FastAPI layer is the simplest way to keep one authoritative copy of the equations. The research simulator, experiment runners, results, and tests are untouched.

### 1.2 The one change to model code (behaviour-preserving)

`ModelAlignedTargetAdaptiveController.select_next_task(step_idx)` plans **simulated** time gaps internally: 0.02 h between trials, 12 h after 15 trials, and ≥ 2 h on a cooldown. A live system observes real time instead. The method body was split, without changing any equation, into:

- `_consume_safety_trigger(...)`: the existing frustration-safety block.
- `_recommend_difficulty(game_id, time_gap, projected_counter, safety)`: forgetting/diffusion, bisection, deadband, step limit, and diagnostics. This is the existing code moved verbatim.
- `select_difficulty_for_elapsed_time(game_id, elapsed_hours)`: a new live entry point that calls the same two pieces with real elapsed hours.

Evidence that simulator behaviour is unchanged:
- All 73 pre-existing tests pass.
- A paired closed-loop run comparing pre- and post-refactor code produced identical event and mastery DataFrames (810 events across 3 policies, `DataFrame.equals` → True).
- `test_live_selection_matches_simulator_selection_path` checks that both entry points produce identical difficulties, diagnostics, and mastery over a sequence that includes a safety trigger.

### 1.3 Live semantics chosen where the simulator had no live equivalent

| Simulator behaviour | Live behaviour | Rationale |
|---|---|---|
| Planned gap of 0.02 h between trials; forgetting applied at task selection | Real wall-clock hours since forgetting was last applied, applied at task selection only | Same transition model; navigation never triggers decay |
| Session gap of 12 h after `max_session_trials` (15) | Backend returns `endSession {reason: "session_trial_limit"}`; a new `sessionId` resets the session counter | The config defines a trial limit, not a break length |
| Cooldown gap of `cooldown_gap_hours` (2 h) after 3 consecutive observed failures, then D −0.25 | Backend returns `break {reason: "frustration_cooldown", lengthSec}` until 2 h have passed since the last observation; the next task takes the controller's safety path | Faithful to the configured rest. Starting a new session does not bypass it |
| One shared `current_difficulty` across all games | **One D per game**, saved and restored around each controller call; all bounds, steps, and deadbands are unchanged | Each game has its own IRT item parameters; the Integrated Mathematical System v1.1 (§8) says never to transfer D across games. This is the only bookkeeping deviation from the simulator; the estimator is unaffected |

---

## 2. API operations

Base URL from `EXPO_PUBLIC_GROW21_API_URL`. All bodies are JSON. Unknown fields are rejected, and NaN/Infinity are rejected.

| Operation | Purpose |
|---|---|
| `GET /health` | Liveness: `{"status":"ok","schemaVersion":1,"estimator":"robust_adaptive_filter"}` |
| `GET /api/v1/config` | Configured skills, the nine game mappings, difficulty bounds, and safety limits (used by the live contract test) |
| `POST /api/v1/task/next` | `NextTaskRequest` → `TaskDirective` |
| `POST /api/v1/telemetry/submit` | `TelemetryEvent` → `{ack, duplicate, observationApplied}` |
| `GET /api/v1/learners/{childId}/state` | Current mastery estimates, per-game D, session counters, last controller decision (inspection and testing) |

### 2.1 `NextTaskRequest` (frontend `src/types.ts`)

| Field | Type | Use |
|---|---|---|
| `schemaVersion` | `1` | required |
| `childId`, `sessionId` | `^[A-Za-z0-9_-]{1,64}$` | pseudonymous learner and session keys |
| `mode` | `"play"` \| `"baseline"` | only `play` is supported; `baseline` → 422 `unsupported_mode` (no backend baseline procedure exists) |
| `sessionActiveMs`, `tasksCompletedThisSession`, `lastTaskId`, `lastOutcome`, `availability` | validated, not used by the model | the backend has no active-time limit or availability-dependent logic |

### 2.2 `TaskDirective`: exactly one of

- `task: {taskId, gameId, skillId, D}`. `taskId` is issued by the server. `gameId` is the frontend game key. `skillId` is the game's configured primary skill. `D` ∈ [0.08, 0.95], rounded to 3 decimals.
- `break: {reason: "frustration_cooldown", lengthSec}`. `lengthSec` is the remaining cooldown.
- `endSession: {reason: "session_trial_limit"}`.

`startScaffoldLevel` is never sent, because the mastery-to-scaffold mapping L = ⌈L_max(1 − θ)⌉ from the math documents is not implemented in the backend. Frontend scaffolding stays local and in-task.

### 2.3 `TelemetryEvent` and how it becomes an observation

The event schema matches `src/types.ts` (attempts, timings, scaffold levels, `gameConfig`, and the optional `motor`/`speech`/`affect`/`task`), with one additional optional field:

- `reliability` (number in [0, 1]): an observed reliability ρ̂. Send it **only** when a game has a real uncertainty estimate; no current game does. When omitted, the controller applies its documented convention (no explicit uncertainty model → ρ = 1; see `default_reliability_convention` and Integrated Mathematical System §4.3). No values are fabricated.

Mapping to the estimator's binary observed response `y_obs`:

| Event | `y_obs` | Model update |
|---|---|---|
| `outcome` success/failure, `attempts[0].outcome == "success"` | 1 | yes |
| `outcome` success/failure, `attempts[0].outcome == "failure"` | 0 | yes (also counts towards the consecutive-failure safety rule) |
| `attempts[0].outcome == "stalled"` | none | no |
| `outcome` ∈ {abandoned, timeout, skipped} | none | no; task closed, nothing counted |

**Decision: the first attempt defines the observation.** The canonical outcome model predicts *unassisted* clean success at the presented D. Later attempts in the frontend may happen after in-task scaffold escalation, which the model does not represent. A task solved only after help therefore counts as a failure at D. This is conservative: it triggers the 3-failure cooldown sooner than counting final outcomes would. **The team should confirm this choice**; it is isolated in `LearnerRuntime._observed_outcome`.

Latency, retries, `assisted`, and the motor/speech/affect payloads are validated and logged with the observation but **not used by the model**. The backend implements no latency fatigue (B_t), frustration proxy (F_t), or motor-slip model.

### 2.4 Validation and protocol rules (no partial updates)

Everything is validated before any state changes, and changes are applied to a copy that replaces the stored state only on success.

| Condition | Status / `error.code` |
|---|---|
| Schema violation (unknown game, bad enum, out-of-range number, extra field, NaN) | 422 `invalid_request` |
| `gameId`/`skillId`/`difficulty` differ from the issued task | 422 `task_mismatch` |
| `isBaseline: true`, or `mode: "baseline"` | 422 `unsupported_mode` |
| `scaffoldStartLevel ≠ 0`, no attempts, outcome inconsistent with attempts, first attempt not at the start level | 422 `invalid_event` |
| Telemetry for a learner that was never issued a task | 404 `unknown_learner` |
| `taskId` was not issued, or is not the outstanding task | 409 `unknown_task` |
| Event from a superseded session; next-task for a superseded session | 409 `stale_session` / `session_closed` |
| A second event (new `eventId`) for an already-observed task | 409 `task_already_observed` |
| Same `eventId` resent with a different payload | 409 `event_id_conflict` |
| Same `eventId` resent with the identical payload | 200 with `duplicate: true`; state unchanged |
| Stored state created under a different model config | 409 `state_config_mismatch` |

Error body: `{"error": {"code": "...", "message": "...", "details"?: [...]}}`.

**Ordering and retries:** at most one task is outstanding per learner, so events cannot arrive out of order. Repeating `task/next` while a task is outstanding returns the same task without changing state, so client retries never double-step difficulty or decay. If the frontend's local fallback replays a task, that replay's event is rejected (`task_already_observed`) and is not counted twice.

**Session restoration:** remounting with the same `childId`/`sessionId` resumes the outstanding task. A new `sessionId` starts a new session (the trial counter resets; mastery, per-game D, and the consecutive-failure count carry over). Earlier session IDs are then closed.

**Persistence:** state is in memory by default. Set `GROW21_STATE_DIR` to persist each learner as a pickle after every change (atomic replace); state survives restarts. Only point it at a server-local directory you control, because pickle files are executable on load.

---

## 3. Verified game mapping (all nine)

Source: `configs/default.yaml` (backend) and `src/games.ts` / `src/GameTask.tsx` (frontend).

| Game | Frontend key → config ID | Primary skill | Secondary skill | Observations the current UI can produce | Frontend ψ_g(D) parameters | Status |
|---|---|---|---|---|---|---|
| Trace & Glow | `trace` → `trace_glow` | fine_motor | visual_discrimination | per-attempt success/failure via buttons, latency, retries | pathWidthDp, pathLengthDp, curvature, turnCount, shapeType | Connected; generic placeholder UI |
| Bubble Pop Brigade | `bubble` → `bubble_pop_brigade` | fine_motor | visual_discrimination | same | bubbleSizeDp, speedDpPerSec, concurrentBubbles, minSeparationDp | Connected; generic UI |
| Block Stacker | `block` → `block_stacker` | fine_motor | sequential_reasoning | same | blockCount, blockSizeDp, placementToleranceDp, structure | Connected; generic UI |
| Echo Safari | `echo` → `echo_safari` | phonological_awareness | working_memory | same; **no VAD/microphone capture exists** | promptVariety, repetitions, responseWindowMs, demand | Connected; generic UI |
| Picture-Sentence Strips | `sentence` → `picture_sentence_strips` | sequential_reasoning | categorization | same | sentenceLength, vocabLevel, distractorCount, structureType | Connected; generic UI |
| Audio Search | `audio` → `audio_search` | working_memory | phonological_awareness | same (no child ASR, as designed) | optionCount, visualSimilarity, attributeCount, instructionComplexity | Connected; generic UI |
| Category Sorting | `sort` → `category_sorting` | categorization | visual_discrimination | same; no drag/placement telemetry | itemCount, categoryCount, categorySimilarity, attributeComplexity, placementToleranceDp | Connected; generic UI |
| Visual Flip Match | `flip` → `visual_flip_match` | working_memory | visual_discrimination | same | cardCount, visualSimilarity, exposureMs, loadLevel | Connected; generic UI |
| Pattern Train | `pattern` → `pattern_train` | sequential_reasoning | visual_discrimination | same | sequenceLength, ruleCount, distractorCount, visualSimilarity | Connected; generic UI |

Six configured skills are shared across games. For example, Trace, Bubble, and Block all update `fine_motor`, and evidence from one game moves the D recommended for the next (tested).

**No game has a real renderer.** All nine use the same `GameTask` card with "Try again" / "I found it!" buttons; the frontend's `docs/OPEN_ITEMS.md` records this. Everything the UI can genuinely report is connected. Touch paths, drag, VAD, and affect signals do not exist yet, so they are not sent.

Skill-ID alignment: the frontend previously used nine per-game labels (`fine_motor_tracing`, …), none of which exist in the backend config. They were replaced with the six configured IDs in `src/types.ts` and `src/games.ts`. Game keys were **not** renamed; the backend maps them (`FRONTEND_GAME_KEYS`). Parity is checked by the live contract test (`/api/v1/config` against `SKILLS`).

---

## 4. Setup and running

### 4.1 Backend
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .\.venv\Scripts\activate
pip install -e .
python -m uvicorn grow21_simulation_benchmark.service.app:create_app --factory --host 0.0.0.0 --port 8000
```
Environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `GROW21_CONFIG` | `configs/default.yaml` | model configuration |
| `GROW21_STATE_DIR` | unset (in-memory) | persist learner state across restarts |
| `GROW21_CORS_ORIGIN_REGEX` | `^https?://(localhost\|127\.0\.0\.1)(:\d+)?$` | allowed browser origins (only the Expo web host needs CORS) |

The service has **no authentication**. Run it only on a trusted development machine or LAN.

### 4.2 Frontend (from the `grow21-integration-ui` checkout, e.g. `git worktree add ../grow21-frontend grow21-integration-ui`)
```bash
cd demo
npm install
EXPO_PUBLIC_GROW21_API_URL=http://<host>:8000 npx expo start --android   # or --web
```
Choose the backend address for your device:

| Device | `EXPO_PUBLIC_GROW21_API_URL` |
|---|---|
| Expo web on the same machine | `http://localhost:8000` |
| Android emulator | `http://10.0.2.2:8000` |
| Physical device | `http://<computer LAN IP>:8000` (start uvicorn with `--host 0.0.0.0`), or `adb reverse tcp:8000 tcp:8000` and `http://localhost:8000` |

Without the variable, the demo runs the development-only `MockEngine` and says so in its banner. `EXPO_PUBLIC_GROW21_CHILD_ID` (default `demo-child`) sets the pseudonymous learner. Each demo launch, and each "Start new session", uses a fresh `sessionId`. Android release builds typically block cleartext `http://` to non-localhost hosts; development builds normally allow it. This is unverified here.

---

## 5. Tests and results (October 18, 2026, Linux, Python 3.13.14, Node 22.22.1)

| Suite | Command | Result |
|---|---|---|
| Backend baseline before changes | `python -m pytest -q tests` | 73 passed, 1 skipped (matches the robustness report) |
| Backend after integration | `python -m pytest -q tests` | **110 passed, 1 skipped** (37 new in `tests/test_service_api.py`; the same optional full-artifact test skipped) |
| Frontend baseline | none existed; `tsc` failed on pre-existing type errors | — |
| Frontend typecheck | `cd demo && npm run typecheck` | pass |
| Frontend unit/UI | `cd demo && npm test` | **33 passed** |
| Frontend ↔ live backend | `cd demo && GROW21_API_URL=http://127.0.0.1:8000 npm run test:integration` | **5 passed** (real module + `HttpEngineClient` + running uvicorn) |
| Manual end-to-end | Expo web build in a browser against uvicorn | passed (see 5.1) |

Integration scenarios and where they are tested:

| Scenario | Test |
|---|---|
| Success followed by a different game with the same skill | `test_shared_skill_evidence_carries_across_games`; live test 2 |
| Interactions involving different skills | `test_observation_only_updates_the_games_configured_skills` |
| Consecutive failures → safety response; recovery | `test_consecutive_failures_trigger_cooldown_break_then_safety_drop_and_recovery`; live test 2 |
| Missing optional modality/reliability data | `test_missing_reliability_uses_convention_and_low_reliability_dampens_update` |
| Invalid game/skill/difficulty | `test_rejected_events_do_not_mutate_state` (16 cases, byte-identical state check) |
| Timeout followed by recovery | `HttpEngineClient` timeout/retry tests; SessionRunner fallback test |
| Duplicate request | `test_duplicate_event_and_retry_semantics`, `test_next_task_is_idempotent_...`, double-tap UI tests |
| Pause/break/resume | cooldown test (break → early resume keeps break → resume after rest); session-limit test; UI break test |
| Session restoration | `test_restoring_the_same_session_...`, `test_state_persists_across_service_restart`; live test 3 |
| Two isolated learners | `test_learners_are_isolated`; live test 4 |
| Multiskill game | `test_secondary_transfer_is_success_modulated` |
| Forgetting with real elapsed time | `test_forgetting_applies_with_real_elapsed_time_only` |

### 5.1 Manual end-to-end run (real UI, real backend)

The Expo **web** build ran in a browser against uvicorn:
- Trace & Glow appeared at the backend's D = 0.37 (pathWidthDp 92).
- "I found it!" raised `fine_motor` from 0.454 to 0.506 and gave `visual_discrimination` a small transfer gain. The next task was Bubble Pop Brigade, which shares `fine_motor`.
- Three tasks solved only after "Try again" produced a `frustration_cooldown` break, and the app exited with `break`.
- "Start new session" received the break again (7195 s remaining) instead of bypassing the cooldown.
- The UI's 4 telemetry submissions all returned 200. The server log showed no non-200 responses except the live suite's deliberate malformed-request check.

---

## 6. Known limitations and unverified behaviour

- **Not run on Android or iOS.** No emulator or device was available. Tests used the Jest React Native environment and the Expo web target. Native networking (emulator alias, cleartext policy, physical-device LAN access) is unverified.
- **No real game renderers** (see section 3). The observation is whatever the button card records.
- **Not implemented in the backend, so not exposed:** frustration proxy F_t, latency-based fatigue B_t, active-time and daily-time break limits (b_active, b_day), the mastery→scaffold mapping, the skill-priority scheduler (eq. 29–30), baseline initialisation, motor-slip and ASR reliability models, plateau and confirmation policies, and caregiver feedback. Caregiver reports stay a separate channel; `MasteryPriorInitializer` exists but is not wired to any endpoint.
- **Math-document vs implementation discrepancies (unchanged, by design of earlier phases):** the documents specify DKT and p̂ = σ(logit θ − γD^ν) with γ = 3.5, ν = 2, D ∈ [0, 1], a step of 0.05, and a deadband of 0.05. The implemented, validated controller uses the canonical IRT model (Phase 3), D ∈ [0.08, 0.95], a step of 0.08, and a deadband of 0.04. The backend is followed.
- **Engineering settings, not clinical guidance:** the 2-hour cooldown and 15-trial session limit come from the simulator configuration. They are enforced as configured and should be reviewed for real use.
- **First-attempt observation rule** (section 2.3) needs team confirmation.
- **Frontend local fallback:** if `task/next` times out, `SessionRunner` repeats the last task up to twice. Those replays are uncounted by design.
- **Telemetry delivery** is in memory only: a failed submit (after retries and the 1.5 s SessionRunner window) is dropped, and the backend then re-issues the same outstanding task. There is no offline queue.
- **Learner start:** on its first selection every skill moves from 0.45 to ≈0.454. This is the controller's own grid-mean recomputation (truncated Gaussian prior), and it happens in the simulator too.
- **No authentication, rate limiting, or multi-process deployment.** One global lock serialises requests, which is adequate for a single development server.
