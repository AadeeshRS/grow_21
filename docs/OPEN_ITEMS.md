# Open specification items

- `TODO(spec-ambiguity)`: the source requests concrete React Native game renderers (Skia paths, VAD events, localized assets) but supplies no asset bank, visual design, or native host project. The initial task surface is deliberately generic; replace it with the nine production renderers once those inputs exist.
- `TODO(spec-ambiguity)`: persistent, disk-backed telemetry buffering is host/platform dependent. The supplied module acknowledges bounded submissions but does not select a storage implementation.
- `TODO(spec-ambiguity)`: fixed baseline sequence and its score proposal need an engine-approved reference-latency configuration before enabling baseline production use.
- `TODO(spec-ambiguity)`: native bridge implementation and lifecycle event subscription belong in the Android host project, which was not provided.
