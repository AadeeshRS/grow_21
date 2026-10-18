# Local demo

This is an Expo host for the Grow21 module: a home screen, the nine games, a rest screen for breaks, an end-of-session celebration, and a grown-ups panel. Without a backend URL it runs the development-only `MockEngine`; no microphone or Android native bridge is required.

From this directory, run:

```powershell
npm install
npx expo start --android
```

To use the real learner model, start the backend (`grow21-integration` branch) and set its URL:

```bash
EXPO_PUBLIC_GROW21_API_URL=http://10.0.2.2:8000 npx expo start --android   # emulator; use your LAN IP on a phone, or http://localhost:8000 with --web
```

Without `EXPO_PUBLIC_GROW21_API_URL` the demo uses the development-only mock engine (the grown-ups panel says so).

Design preview (web only): `http://localhost:8081/?preview=trace&D=0.5&level=1` plays one game repeatedly at the given difficulty and help level, without the learner service.

Grown-ups: hold the 🔒 for a second to open the grown-ups panel (connection status, model estimates) or to leave a game.

Use an Android emulator or a phone with Expo Go (`--android`), or a browser (`--web`). Press **Play**; each game explains itself out loud (🔊 repeats it). Two misses in a row turn on the game's help ("💡 Help is on").
