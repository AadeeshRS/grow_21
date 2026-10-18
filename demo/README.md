# Local demo

This is an Expo test host for the Grow21 module. It runs the module with `MockEngine` and `MockHostBridge`; no learner engine, microphone, or Android native bridge is required.

From this directory, run:

```powershell
npm install
npx expo start --android
```

To use the real learner model, start the backend (`grow21-integration` branch) and set its URL:

```bash
EXPO_PUBLIC_GROW21_API_URL=http://10.0.2.2:8000 npx expo start --android   # emulator; use your LAN IP on a phone, or http://localhost:8000 with --web
```

Without `EXPO_PUBLIC_GROW21_API_URL` the banner shows that the development-only mock engine is in use.

Use an Android emulator or a phone with Expo Go. Tap **I found it!** to complete a task and move to the next game. Tap **Try again** twice to confirm that scaffold help increments by one level. Long-press **Exit** to test the exit callback.
