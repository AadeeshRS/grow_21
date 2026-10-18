# Local demo

This is an Expo test host for the Grow21 module. It runs the module with `MockEngine` and `MockHostBridge`; no learner engine, microphone, or Android native bridge is required.

From this directory, run:

```powershell
npm install
npx expo start --android
```

Use an Android emulator or a phone with Expo Go. Tap **I found it!** to complete a task and move to the next game. Tap **Try again** twice to confirm that scaffold help increments by one level. Long-press **Exit** to test the exit callback.
