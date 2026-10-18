# Open specification items

- Game renderers (October 2026): all nine games now have playable boards (`src/boards/`, see `docs/GAME_DESIGN.md`) built with React Native views, `react-native-svg` and `expo-speech`. Still open: illustrated assets and recorded audio, localisation, native voice-activity detection, and review of the content banks by speech-language / special-education professionals.
- `TODO(spec-ambiguity)`: persistent, disk-backed telemetry buffering is host/platform dependent. The supplied module acknowledges bounded submissions but does not select a storage implementation.
- `TODO(spec-ambiguity)`: fixed baseline sequence and its score proposal need an engine-approved reference-latency configuration before enabling baseline production use.
- `TODO(spec-ambiguity)`: native bridge implementation and lifecycle event subscription belong in the Android host project, which was not provided.
- Backend integration (October 2026): connected through `HttpEngineClient`. Still open: an on-device Android run (only Jest and the Expo web target were exercised); raw touch-path capture and affect signals; an offline telemetry queue; and baseline mode, which the backend rejects because no baseline procedure is implemented there.
