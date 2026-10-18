# Grow21 game module

This is a React Native TypeScript module generated from `prdgroww21game.md`. It deliberately keeps learner-state and adaptation decisions outside the module.

## Use

Mount `Grow21GameModule` with `childId`, `sessionId`, `locale`, `mode`, and `onExit`. In an integrated app, pass an `EngineClient` backed by `Grow21Host.engineGetNextTask` / `engineSubmitTelemetry` and a `HostBridge` backed by the remaining native methods. When omitted, development-only `MockEngine` and `MockHostBridge` are used.

The module includes the nine difficulty mappings, cumulative scaffold mapping, a pure scaffold controller, typed engine/telemetry contracts, a thin session runner, and **playable renderers for all nine games** (`src/boards/`). Game rules, help ladders, telemetry and the design research behind them are in [`docs/GAME_DESIGN.md`](docs/GAME_DESIGN.md). It clamps `D` and scaffold start levels, validates game-to-skill directives, honours break/end-session directives, and emits one telemetry event after each completed task.

## Backend integration

`HttpEngineClient` (`src/api/HttpEngineClient.ts`) implements `EngineClient` against the Grow21 Python learner service (backend branch `grow21-integration`, documented in `docs/integration/FRONTEND_BACKEND_API.md` there). The backend owns mastery estimation and difficulty. This module renders the issued task, applies `difficultyConfig(gameId, D)`, records attempts, and echoes the server-issued `taskId` in telemetry. Skill IDs in `src/types.ts` / `src/games.ts` are the backend's six configured skills.

```ts
<Grow21GameModule engine={new HttpEngineClient(process.env.EXPO_PUBLIC_GROW21_API_URL!)} childId="..." sessionId="..." locale="en-IN" mode="play" onExit={...} />
```

Tests (run from `demo/`): `npm run typecheck`, `npm test` (47 unit/UI tests, including every game played through real interactions), and `GROW21_API_URL=http://127.0.0.1:8000 npm run test:integration` (5 tests, needs the backend running).

Hosts must load the Nunito font (`@expo-google-fonts/nunito`: 700/800/900) and provide `react-native-svg` and `expo-speech`; the demo host does all three.

## Important integration work remaining

Still missing: an Android/iOS device run, illustrated art and recorded audio (emoji, SVG drawings and device text-to-speech stand in), native voice-activity detection (web only; native uses a grown-up confirmation), and a persistent telemetry queue. Those production-specific parts are listed in `docs/OPEN_ITEMS.md`; the contracts are intentionally ready for their implementations without changing game mappings or session flow.
