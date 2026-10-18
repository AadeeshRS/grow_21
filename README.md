# Grow21 game module

This is a React Native TypeScript module generated from `prdgroww21game.md`. It deliberately keeps learner-state and adaptation decisions outside the module.

## Use

Mount `Grow21GameModule` with `childId`, `sessionId`, `locale`, `mode`, and `onExit`. In an integrated app, pass an `EngineClient` backed by `Grow21Host.engineGetNextTask` / `engineSubmitTelemetry` and a `HostBridge` backed by the remaining native methods. When omitted, development-only `MockEngine` and `MockHostBridge` are used.

The module includes the nine difficulty mappings, cumulative scaffold mapping, a pure scaffold controller, typed engine/telemetry contracts, a thin session runner, and a child-safe task surface. It clamps `D` and scaffold start levels, validates game-to-skill directives, honours break/end-session directives, and emits one telemetry event after each completed task.

## Important integration work remaining

The supplied workspace has no Android/RN host, art/audio assets, Skia dependency, gesture capture, VAD service, or persistent queue. Those production-specific parts are listed in `docs/OPEN_ITEMS.md`; the contracts are intentionally ready for their implementations without changing game mappings or session flow.
