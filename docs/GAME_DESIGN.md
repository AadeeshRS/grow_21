# Grow21 Game Design: Research, Principles, and the Nine Games

**Branch:** `grow21-integration-ui` · **Date:** October 2026
**Status:** Playable renderers for all nine games, tested in Jest, against the live backend, and in the Expo **web** build. Not yet tried on an Android/iOS device or with children.

---

## 1. What we learned from comparable products and research

| Source | What it does well | What Grow21 takes from it |
|---|---|---|
| **See and Learn** (Down Syndrome Education International) | Small structured steps; picture-based matching before naming; designed around auditory-processing and verbal short-term-memory difficulties | Pictures first, one concept per screen, the same activity format repeated with small changes |
| **Khan Academy Kids** | Minimal text; every instruction narrated and replayable; sparkles on first-try success; collectable rewards | Every instruction is spoken and has a 🔊 replay; immediate celebration; a session star counter |
| **Otsimo** | Special-education games with large targets and positive reinforcement only | No buzzers, red crosses or "wrong" language; calm retry |
| **Speech Blubs** | Voice-activated imitation with immediate encouraging feedback | Echo Safari: a model sound, then "Your turn!"; any vocalisation counts |
| **Proloquo2Go / Leeloo (AAC)** | Picture symbols with words, sentence strips, text-to-speech readback | Picture-Sentence Strips: picture+word tiles; the finished sentence is spoken |
| **Touch-gesture study with people with Down syndrome** (Sensors, PMC7917966) | Tap and slide were easiest; **press-and-drag was the hardest gesture**; double tap was unreliable; recommends starting gestures on the target and confirming completion with audio | Tap is the default everywhere. Drag is used only where the motor skill is the point (Trace, Block, Sort), with generous magnetic tolerance; Block adds tap-then-tap at help ≥ 2; Sort always accepts tapping a basket. No double tap anywhere; long press only for grown-ups |
| **Down syndrome learning profile** (DSE / education guidance) | Strong visual learning, weaker auditory working memory, benefit from errorless learning and consistent routines | One screen layout for all nine games, the same words for retries and hints, escalating help instead of failure, no time pressure |

Sources: [See and Learn](https://www.down-syndrome.org/en-in/resources/see-and-learn), [Khan Academy Kids](https://blog.khanacademy.org/supporting-english-language-acquisition-with-khan-academy-kids/), [Otsimo](https://otsimo.com/en/best-communication-apps-autism/), [Speech Blubs](https://speechblubs.com/blog/best-apps-for-speech-delay-helping-your-child-find-their-voice), [Educators Technology (special-education apps)](https://www.educatorstechnology.com/2024/02/apps-for-special-education.html), [Touchscreen gestures for users with Down syndrome](https://pmc.ncbi.nlm.nih.gov/articles/PMC7917966), [DSi learning profile](https://www.ds-int.org/learning-profile).

## 2. Design principles applied

1. **Picture first, speech second, text third.** Every instruction has a picture context, is spoken at a slightly slower rate (0.85) and can be replayed. Short written cues support emerging readers.
2. **One consistent frame.** Stars on the top left, the game name, and a grown-up lock; then the instruction card with a 🔊 button; then one white play surface. Only the play surface changes between games.
3. **Errorless, escalating help.** A miss gives a calm "Let's try again." After two misses: "Let's try together. Here is a little help." and the game's own help ladder turns on (the "💡 Help is on" chip). After the last level the child hears "Nice trying!" and moves on; there are no punishing sounds or colours.
4. **Large targets.** At least 72 dp, with extra hit padding. On large screens dp-based difficulty sizes are scaled up (×1–1.6) so tablets don't show tiny objects. Sizes still shrink as difficulty rises.
5. **No time pressure for the child.** Bubbles float slowly and escaped ones come back; Flip Match previews before hiding; inactivity triggers at most two spoken reminders per attempt and is never counted as a miss.
6. **Grown-up controls are hidden.** Leaving a game or opening the grown-ups panel needs a 1.2-second hold on the 🔒. Children never see an "Exit" label.
7. **Honest data.** Each game defines exactly what counts as an attempt's success or failure (section 3). Signals that don't exist (gaze, affect, ASR) are never invented.
8. **Responsive.** Tested at 1880×950, 1280×800, 812×375 (landscape phone, with side rails) and 375×812 (portrait phone). No page scrolling and no element outside the play surface.
9. **Robust to slowed animation.** Every action (scoring, hold-to-leave) is driven by timers or events, never by an animation finishing.

## 3. The nine games

Difficulty `D` comes from the backend. `difficultyConfig` and `applyScaffold` in `src/games.ts` (unchanged) turn it into parameters. The rules below are what the boards in `src/boards/` do with those parameters.

| Game | What the child does | Attempt succeeds when | Attempt fails when | Help ladder (scaffold 1→3) | Attempt telemetry |
|---|---|---|---|---|---|
| **Trace & Glow** | Puts a finger on the ladybug and slides along a glowing path to the flower | The path is followed to the end | The finger leaves the path 3 times (`dropFailLimit`); lifting and resuming is allowed | Wider path and stronger glow → larger resume area → a moving ghost dot shows the way, maximum tolerance | `motor`: off-path exits, lifts, max/mean deviation, coverage, path width, shape |
| **Bubble Pop Brigade** | Taps slowly rising bubbles | `popTarget` (5) bubbles popped | 4 taps in a row on empty sky (`consecutiveMissLimit`) | Bigger, slower bubbles → at most 2 at a time with a 👆 hint → one bubble, extra hit padding | `motor`: popped, misses, max consecutive misses, mean miss distance, escaped, size |
| **Block Stacker** | Drags each shape onto its matching outline; stacks are built bottom-up | Every block placed | 3 drops that miss, use the wrong outline, or skip the stacking order | Bigger blocks and tolerance → magnetic snap, matching outline glows, **tap-then-tap placement** → at most 2 blocks, the next space pulses | `motor`: release misses, wrong target, order errors, drags, taps, tolerance |
| **Echo Safari** | Listens to the animal's sound, then makes a sound | Every prompt answered with a detected vocalisation | The response window closes (microphone), or the grown-up taps "Not yet" | Model repeated → stronger cue, longer window → model repeated twice, one prompt | `speech`: mode (`vad`/`observer`), prompts, latencies, peak level, policy `any_detected_vocalization` |
| **Picture-Sentence Strips** | Taps picture-words in order to build the spoken sentence; it is read back | The sentence is complete | 3 out-of-order taps | Fewer distractors → the next word glows, shorter sentence → faint worked example in the slots, no distractors | `task`: errors, length, distractors |
| **Audio Search** | Hears "Find the small blue car" and taps it | The described object is tapped | Any other object is tapped | Instruction repeated → 2 options, no look-alikes → one attribute only, instruction repeated twice | `task`: chosen/answer index, options, attributes |
| **Category Sorting** | Drags (or taps a basket for) one picture at a time into two baskets | All pictures sorted | 3 classification errors (`dropFailLimit`); **drops outside a basket are counted separately and never fail** | Bigger labels, more tolerance → an example in each basket, max 4 items → the right basket glows, snap to basket | `task`: classification errors, placement misses, drags, taps |
| **Visual Flip Match** | Looks at the cards ("👀 Look carefully!"), then finds the pairs | All pairs found | 3 mismatches (`incorrectFlipLimit`) | Longer preview → 4 cards, no look-alikes → a pair is highlighted during the preview | `task`: mismatches, flips, pairs, preview time |
| **Pattern Train** | Chooses what comes next in the train | The right picture is chosen (the train departs) | Any other picture is chosen | One fewer distractor → repeating unit outlined, shorter train → 2 choices, the unit bounces | `task`: chosen/answer index, choices, unit and sequence length |

Content is generated deterministically from the server-issued `taskId`, so a retry shows the same items and tests can compute the right answers. Generators are verified over 40 seeds × 5 difficulties × 4 help levels per game (`tests/boards.test.ts`).

**Why Audio Search uses drawn objects:** emoji can't be recoloured. An early version put the colour on the frame ("find the green fish" showed two blue fish), which is wrong for a receptive-language task. Objects are now SVG drawings whose fill is the named colour.

## 4. Observation rule shared with the backend

The backend uses **the first attempt** of each task as the observation (`y_obs`), because its outcome model predicts unassisted success at the presented difficulty. In practice, a task solved only after help still earns a star and a celebration, but tells the model "not yet at this difficulty". See the backend's `docs/integration/FRONTEND_BACKEND_API.md` §2.3.

## 5. Preview mode for design review (web only)

`http://localhost:8081/?preview=<game>&D=<0..1>&level=<0..3>` plays one game repeatedly with a scripted mock engine. It never contacts the learner service. Example: `?preview=block&D=0.8&level=3`.

## 6. Testing

From `demo/`:
- `npm run typecheck`
- `npm test`: 47 tests. All nine games are played through real interactions (taps, responder drags along exact geometry, the grown-up confirmation for Echo). Also covered: retries and hint escalation to exhaustion, separate sorting statistics, stars and fresh per-task state, inactivity, hold-to-leave, break/end/error flows, generator validity, and the voice-activity rule.
- `GROW21_API_URL=http://127.0.0.1:8000 npm run test:integration`: 5 tests against the running backend. Real games drive a shared-skill update, three first-attempt failures (missed bubbles, dropped blocks, "Not yet" in Echo) and the backend's frustration cooldown break.

## 7. Known limitations

- **Not run on a device.** Only Jest and the Expo web build were exercised. Native gesture feel, TTS voices, safe areas and emoji rendering on Android are unverified.
- **Microphone detection is web-only.** Native builds use the grown-up confirmation until a native voice-activity bridge exists. Microphone detection could not be tested in the review browser (capture is blocked there); the decision rule is unit-tested.
- **No illustrated art or recorded audio yet.** Emoji, simple SVG drawings and device text-to-speech stand in. Emoji look different on each platform.
- **Placeholder content.** The sentence bank, sorting categories and look-alike sets are small, and have not been reviewed by speech-language or special-education professionals.
- **Not evaluated with children.** Usability, engagement and appropriateness for children with Down syndrome have not been tested.
