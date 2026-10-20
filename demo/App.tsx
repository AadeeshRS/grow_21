import React, {useEffect, useMemo, useRef, useState} from 'react';
import {Animated, Easing, Modal, Platform, SafeAreaView, ScrollView, StyleSheet, Text, useWindowDimensions, View} from 'react-native';
import Svg, {Ellipse, Path} from 'react-native-svg';
import {useFonts, Nunito_700Bold, Nunito_800ExtraBold, Nunito_900Black} from '@expo-google-fonts/nunito';
import {configureSpeech} from '../src/speech';
import {BigButton, fonts, Grow21GameModule, HoldButton, HttpEngineClient, Mascot, MockEngine, MockHostBridge, say, SKILLS, theme as colors, type GameId, type TaskDirective} from '../src';

// Inlined by Expo at bundle time. Unset -> development-only MockEngine (labelled in the grown-ups panel).
const API_URL = process.env.EXPO_PUBLIC_GROW21_API_URL;
const CHILD_ID = process.env.EXPO_PUBLIC_GROW21_CHILD_ID || 'demo-child';
const engine = API_URL ? new HttpEngineClient(API_URL) : undefined;
// Neural voice from the learner service when available; the device voice is the fallback.
configureSpeech({ttsUrl: API_URL ? `${API_URL.replace(/\/+$/, '')}/api/v1/tts` : undefined});
const newSessionId = () => `session-${Date.now().toString(36)}`;

/**
 * Design-review preview (web only): /?preview=<gameId>&D=<0..1>&level=<0..3> plays one game
 * repeatedly against a scripted mock engine. It never talks to the learner service.
 */
function readPreview(): {gameId: GameId; D: number; level: number} | null {
  if (Platform.OS !== 'web' || typeof window === 'undefined') return null;
  const q = new URLSearchParams(window.location.search);
  const gameId = q.get('preview') as GameId | null;
  if (!gameId || !(gameId in SKILLS)) return null;
  const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, Number.isFinite(v) ? v : lo));
  return {gameId, D: clamp(Number(q.get('D') ?? 0.5), 0, 1), level: Math.round(clamp(Number(q.get('level') ?? 0), 0, 3))};
}
const PREVIEW = readPreview();
const previewEngine = (p: NonNullable<typeof PREVIEW>) => new MockEngine(Array.from({length: 50}, (_, i): TaskDirective => (
  {schemaVersion: 1, task: {taskId: `preview-${p.gameId}-${i}`, gameId: p.gameId, skillId: SKILLS[p.gameId], D: p.D, startScaffoldLevel: p.level}})));

/** Host bridge that remembers the last break so the rest screen can show it. */
class DemoHost extends MockHostBridge {
  lastBreak: {reason: string; lengthSec: number; at: number} | null = null;
  async notifyBreak(info: {reason: string; lengthSec: number}) { this.lastBreak = {...info, at: Date.now()}; }
}

type Screen = 'home' | 'play' | 'rest' | 'done' | 'help';

const SKILL_NAMES: Record<string, string> = {
  fine_motor: 'Fine motor', visual_discrimination: 'Seeing differences', phonological_awareness: 'Sounds & voice',
  working_memory: 'Remembering', categorization: 'Sorting', sequential_reasoning: 'Patterns & order',
};

export default function App() {
  const [loaded] = useFonts({Nunito_700Bold, Nunito_800ExtraBold, Nunito_900Black});
  const host = useMemo(() => new DemoHost(), []);
  const [screen, setScreen] = useState<Screen>('home');
  const [sessionId, setSessionId] = useState(newSessionId);
  const [stars, setStars] = useState(0);
  const [grownUps, setGrownUps] = useState(false);

  if (!loaded) return <View style={styles.root} />;

  if (PREVIEW) return <PreviewHost preview={PREVIEW} />;

  const play = () => {
    say("Let's play!");
    setSessionId(newSessionId());
    setStars(0);
    setScreen('play');
  };

  return (
    <SafeAreaView style={styles.root}>
      {screen === 'play' ? (
        <Grow21GameModule key={sessionId} engine={engine} host={host} childId={CHILD_ID} sessionId={sessionId} locale="en-IN" mode="play"
          onTaskComplete={e => { if (e.outcome === 'success') setStars(s => s + 1); }}
          onExit={reason => setScreen(reason === 'break' ? 'rest' : reason === 'completed' ? 'done' : reason === 'error' ? 'help' : 'home')} />
      ) : (
        <View style={styles.page}>
          <Scenery />
          <View style={styles.corner}><HoldButton testID="grown-ups-open" onHold={() => setGrownUps(true)} label="Grown-ups: hold" /></View>
          {screen === 'home' ? <Home onPlay={play} /> : null}
          {screen === 'rest' ? <Rest host={host} onReady={play} onHome={() => setScreen('home')} /> : null}
          {screen === 'done' ? <Done stars={stars} onHome={() => setScreen('home')} /> : null}
          {screen === 'help' ? <Help onRetry={play} onHome={() => setScreen('home')} /> : null}
        </View>
      )}
      <GrownUps visible={grownUps} onClose={() => setGrownUps(false)} />
    </SafeAreaView>
  );
}

function Floaty({children, delay = 0}: {children: React.ReactNode; delay?: number}) {
  const y = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    const loop = Animated.loop(Animated.sequence([
      Animated.timing(y, {toValue: -10, duration: 1400, delay, easing: Easing.inOut(Easing.sin), useNativeDriver: Platform.OS !== 'web'}),
      Animated.timing(y, {toValue: 0, duration: 1400, easing: Easing.inOut(Easing.sin), useNativeDriver: Platform.OS !== 'web'}),
    ]));
    loop.start();
    return () => loop.stop();
  }, [y, delay]);
  return <Animated.View style={{transform: [{translateY: y}]}}>{children}</Animated.View>;
}

function Home({onPlay}: {onPlay: () => void}) {
  const {width, height} = useWindowDimensions();
  const mascot = Math.max(150, Math.min(280, height * 0.34, width * 0.5));
  const icon = Math.min(64, width * 0.12);
  return (
    <View style={styles.center}>
      <View style={styles.sceneRow}>
        <Floaty delay={200}><Text style={{fontSize: icon}}>🫧</Text></Floaty>
        <Floaty><Mascot size={mascot} mood="happy" /></Floaty>
        <Floaty delay={500}><Text style={{fontSize: icon}}>🚂</Text></Floaty>
      </View>
      <Text style={styles.title}>Hi! Let's play.</Text>
      <BigButton testID="play" title="Play" icon={'\u25B6\uFE0E'} onPress={onPlay} />
    </View>
  );
}

/** Soft hills and clouds behind the non-game screens: a place, not a form. */
function Scenery() {
  const {width, height} = useWindowDimensions();
  return (
    <View pointerEvents="none" style={StyleSheet.absoluteFill}>
      <Svg width={width} height={height}>
        <Ellipse cx={width * 0.18} cy={height * 0.14} rx={70} ry={26} fill="#FFFFFF" opacity={0.9} />
        <Ellipse cx={width * 0.24} cy={height * 0.12} rx={46} ry={30} fill="#FFFFFF" opacity={0.9} />
        <Ellipse cx={width * 0.8} cy={height * 0.2} rx={84} ry={28} fill="#FFFFFF" opacity={0.85} />
        <Ellipse cx={width * 0.86} cy={height * 0.17} rx={50} ry={32} fill="#FFFFFF" opacity={0.85} />
        <Path d={`M0 ${height * 0.86} Q ${width * 0.25} ${height * 0.74} ${width * 0.5} ${height * 0.84} T ${width} ${height * 0.8} L ${width} ${height} L 0 ${height} Z`} fill="#BFEBCB" />
        <Path d={`M0 ${height * 0.92} Q ${width * 0.3} ${height * 0.84} ${width * 0.62} ${height * 0.92} T ${width} ${height * 0.9} L ${width} ${height} L 0 ${height} Z`} fill="#9EDFB2" />
      </Svg>
    </View>
  );
}

function PreviewHost({preview}: {preview: NonNullable<typeof PREVIEW>}) {
  const [run, setRun] = useState(0);
  const engine = useMemo(() => previewEngine(preview), [preview, run]);
  return (
    <SafeAreaView style={styles.root}>
      <Grow21GameModule key={run} engine={engine} childId="preview" sessionId={`preview-${run}`} locale="en-IN" mode="play" onExit={() => setRun(r => r + 1)} />
      <View pointerEvents="none" style={styles.previewTag}><Text style={styles.previewText}>Preview · {preview.gameId} · D {preview.D.toFixed(2)} · help {preview.level}</Text></View>
    </SafeAreaView>
  );
}

function Rest({host, onReady, onHome}: {host: DemoHost; onReady: () => void; onHome: () => void}) {
  const breathe = useRef(new Animated.Value(0)).current;
  const [left, setLeft] = useState(() => remaining(host));
  useEffect(() => { say('Time for a little rest. Breathe in, and out.'); }, []);
  useEffect(() => {
    const loop = Animated.loop(Animated.sequence([
      Animated.timing(breathe, {toValue: 1, duration: 3500, easing: Easing.inOut(Easing.sin), useNativeDriver: Platform.OS !== 'web'}),
      Animated.timing(breathe, {toValue: 0, duration: 3500, easing: Easing.inOut(Easing.sin), useNativeDriver: Platform.OS !== 'web'}),
    ]));
    loop.start();
    const id = setInterval(() => setLeft(remaining(host)), 1000);
    return () => { loop.stop(); clearInterval(id); };
  }, [breathe, host]);
  return (
    <View style={styles.center}>
      <View style={styles.breatheBox}>
        <Animated.View style={[styles.breathe, {transform: [{scale: breathe.interpolate({inputRange: [0, 1], outputRange: [0.75, 1.15]})}]}]} />
        <View style={styles.above}><Mascot size={170} mood="rest" /></View>
      </View>
      <Text style={styles.title}>Time for a little rest</Text>
      {left > 0 ? (
        <>
          <Text style={styles.subtle}>Grown-ups: play can start again in {formatDuration(left)} (cool-down after a tricky run).</Text>
          <BigButton small title="Home" icon="🏠" color={colors.inkFaint} onPress={onHome} />
        </>
      ) : <BigButton title="Ready to play" icon={'\u25B6\uFE0E'} onPress={onReady} />}
    </View>
  );
}

const remaining = (host: DemoHost) => host.lastBreak ? Math.max(0, Math.round(host.lastBreak.lengthSec - (Date.now() - host.lastBreak.at) / 1000)) : 0;
const formatDuration = (s: number) => s >= 3600 ? `${Math.floor(s / 3600)} h ${Math.floor((s % 3600) / 60)} min` : s >= 60 ? `${Math.ceil(s / 60)} min` : `${s} s`;

function Done({stars, onHome}: {stars: number; onHome: () => void}) {
  useEffect(() => { say(`Super playing today! You got ${stars} stars.`); }, []);
  return (
    <View style={styles.center}>
      <Mascot size={200} mood="cheer" />
      <Text style={styles.title}>Super playing today!</Text>
      <Text style={styles.starsLine} accessibilityLabel={`${stars} stars`}>{stars > 0 ? '⭐'.repeat(Math.min(stars, 15)) : '🌟'}</Text>
      <BigButton title="Home" icon="🏠" onPress={onHome} />
    </View>
  );
}

function Help({onRetry, onHome}: {onRetry: () => void; onHome: () => void}) {
  return (
    <View style={styles.center}>
      <Mascot size={170} mood="rest" />
      <Text style={styles.title}>Let's ask a grown-up</Text>
      <Text style={styles.subtle}>Grown-ups: the learning service did not answer{API_URL ? ` at ${API_URL}` : ''}. Check it is running, then try again.</Text>
      <View style={styles.row}>
        <BigButton small title="Try again" icon="↻" onPress={onRetry} />
        <BigButton small title="Home" icon="🏠" color={colors.inkFaint} onPress={onHome} />
      </View>
    </View>
  );
}

type LearnerState = {mastery: Record<string, number>; sessionId: string | null; sessionTrialsObserved: number; consecutiveFailures: number; tasksIssued: number; observationsRecorded: number};

function GrownUps({visible, onClose}: {visible: boolean; onClose: () => void}) {
  const [status, setStatus] = useState<'checking' | 'ok' | 'down' | 'mock'>('checking');
  const [state, setState] = useState<LearnerState | null>(null);
  const [limit, setLimit] = useState<number | null>(null);
  useEffect(() => {
    if (!visible) return;
    if (!API_URL) { setStatus('mock'); return; }
    setStatus('checking');
    fetch(`${API_URL}/health`).then(r => setStatus(r.ok ? 'ok' : 'down')).catch(() => setStatus('down'));
    fetch(`${API_URL}/api/v1/learners/${CHILD_ID}/state`).then(r => (r.ok ? r.json() : null)).then(setState).catch(() => setState(null));
    fetch(`${API_URL}/api/v1/config`).then(r => r.json()).then(c => setLimit(c?.safety?.maxSessionTrials ?? null)).catch(() => setLimit(null));
  }, [visible]);
  return (
    <Modal visible={visible} transparent animationType="fade" onRequestClose={onClose}>
      <View style={styles.scrim}>
        <View style={styles.panel}>
          <ScrollView contentContainerStyle={{gap: 14}}>
            <Text style={styles.panelTitle}>Grown-ups</Text>
            <Row label="Learner" value={CHILD_ID} />
            <Row label="Learning service" value={status === 'mock' ? 'Not connected — practice mode (no learner model)' : status === 'ok' ? `Connected (${API_URL})` : status === 'down' ? `Not reachable (${API_URL})` : 'Checking…'} />
            {state ? (
              <>
                <Row label="This session" value={`${state.sessionTrialsObserved}${limit ? ` of ${limit}` : ''} activities · ${state.consecutiveFailures} tricky in a row`} />
                <Text style={styles.panelHeading}>Model estimates</Text>
                {Object.entries(state.mastery).map(([k, v]) => (
                  <View key={k} style={styles.barRow}>
                    <Text style={styles.barLabel}>{SKILL_NAMES[k] ?? k}</Text>
                    <View style={styles.barTrack}><View style={[styles.barFill, {width: `${Math.round(v * 100)}%`}]} /></View>
                  </View>
                ))}
                <Text style={styles.disclaimer}>These are the app's internal estimates used to choose difficulty. They are not a clinical or developmental assessment.</Text>
              </>
            ) : status === 'ok' ? <Text style={styles.subtleLeft}>No activity recorded for this learner yet.</Text> : null}
            <View style={{alignItems: 'flex-end'}}><BigButton small title="Close" onPress={onClose} color={colors.ink} /></View>
          </ScrollView>
        </View>
      </View>
    </Modal>
  );
}

const Row = ({label, value}: {label: string; value: string}) => (
  <View style={styles.infoRow}><Text style={styles.infoLabel}>{label}</Text><Text style={styles.infoValue}>{value}</Text></View>
);

const styles = StyleSheet.create({
  root: {flex: 1, backgroundColor: colors.sky},
  page: {flex: 1},
  corner: {position: 'absolute', top: 16, right: 16, zIndex: 5},
  center: {flex: 1, alignItems: 'center', justifyContent: 'center', gap: 22, padding: 24},
  sceneRow: {flexDirection: 'row', alignItems: 'flex-end', gap: 18},
  sceneIcon: {fontSize: 64},
  title: {fontFamily: fonts.display, fontSize: 46, color: colors.ink, textAlign: 'center'},
  previewTag: {position: 'absolute', bottom: 6, alignSelf: 'center', backgroundColor: 'rgba(29,43,79,0.75)', borderRadius: 999, paddingHorizontal: 14, paddingVertical: 4},
  previewText: {fontFamily: fonts.strong, color: '#fff', fontSize: 13},
  subtle: {fontFamily: fonts.body, fontSize: 18, color: colors.inkSoft, textAlign: 'center', maxWidth: 520},
  subtleLeft: {fontFamily: fonts.body, fontSize: 16, color: colors.inkSoft},
  starsLine: {fontSize: 40, textAlign: 'center', maxWidth: 640},
  row: {flexDirection: 'row', gap: 16, flexWrap: 'wrap', justifyContent: 'center'},
  breatheBox: {width: 260, height: 260, alignItems: 'center', justifyContent: 'center'},
  breathe: {position: 'absolute', width: 240, height: 240, borderRadius: 120, backgroundColor: colors.skyDeep},
  above: {position: 'relative', zIndex: 1},
  scrim: {flex: 1, backgroundColor: 'rgba(29,43,79,0.35)', alignItems: 'center', justifyContent: 'center', padding: 20},
  panel: {width: '100%', maxWidth: 560, maxHeight: '90%', backgroundColor: colors.paper, borderRadius: 28, padding: 26},
  panelTitle: {fontFamily: fonts.display, fontSize: 30, color: colors.ink},
  panelHeading: {fontFamily: fonts.strong, fontSize: 18, color: colors.ink, marginTop: 6},
  infoRow: {gap: 2},
  infoLabel: {fontFamily: fonts.strong, fontSize: 14, color: colors.inkFaint, textTransform: 'uppercase', letterSpacing: 0.6},
  infoValue: {fontFamily: fonts.body, fontSize: 17, color: colors.ink},
  barRow: {flexDirection: 'row', alignItems: 'center', gap: 12},
  barLabel: {fontFamily: fonts.body, fontSize: 15, color: colors.ink, width: 150},
  barTrack: {flex: 1, height: 14, borderRadius: 7, backgroundColor: colors.sky, overflow: 'hidden'},
  barFill: {height: 14, borderRadius: 7, backgroundColor: colors.motor},
  disclaimer: {fontFamily: fonts.body, fontSize: 14, color: colors.inkSoft, lineHeight: 20},
});
