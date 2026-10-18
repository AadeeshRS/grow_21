import React, {useEffect, useRef, useState} from 'react';
import {Animated, Easing, Platform, Pressable, StyleSheet, Text, View, type StyleProp, type ViewStyle} from 'react-native';
import Svg, {Circle, Ellipse, Path} from 'react-native-svg';
import {colors, fonts, MIN_TARGET, radius, shadow} from '../theme';

export const NATIVE_DRIVER = Platform.OS !== 'web';

/** Gentle horizontal wiggle used when a choice is not the right one (never a buzzer or red cross). */
export function useWiggle() {
  const x = useRef(new Animated.Value(0)).current;
  const wiggle = () => {
    x.setValue(0);
    Animated.sequence([-10, 10, -6, 6, 0].map(v => Animated.timing(x, {toValue: v, duration: 70, useNativeDriver: NATIVE_DRIVER}))).start();
  };
  return {style: {transform: [{translateX: x}]}, wiggle};
}

/** Slow breathing scale loop used to draw attention to a hint target. */
export function Pulse({active, children, style, reducedMotion}: {active: boolean; children: React.ReactNode; style?: StyleProp<ViewStyle>; reducedMotion?: boolean}) {
  const s = useRef(new Animated.Value(1)).current;
  useEffect(() => {
    if (!active || reducedMotion) { s.setValue(1); return; }
    const loop = Animated.loop(Animated.sequence([
      Animated.timing(s, {toValue: 1.08, duration: 650, easing: Easing.inOut(Easing.quad), useNativeDriver: NATIVE_DRIVER}),
      Animated.timing(s, {toValue: 1, duration: 650, easing: Easing.inOut(Easing.quad), useNativeDriver: NATIVE_DRIVER}),
    ]));
    loop.start();
    return () => loop.stop();
  }, [active, reducedMotion, s]);
  return <Animated.View style={[style, {transform: [{scale: s}]}]}>{children}</Animated.View>;
}

export interface TileProps {
  emoji?: string;
  /** Custom picture (e.g. an SVG object) instead of an emoji. */
  content?: React.ReactNode;
  label?: string;
  size: number;
  onPress?: () => void;
  background?: string;
  ring?: string;
  highlighted?: boolean;
  dimmed?: boolean;
  disabled?: boolean;
  testID?: string;
  accessibilityLabel?: string;
  emojiScale?: number;
  reducedMotion?: boolean;
  wiggleRef?: React.MutableRefObject<(() => void) | null>;
}

/** Large picture card: the basic choice element in every tap game. */
export function Tile({emoji, content, label, size, onPress, background = colors.paper, ring, highlighted, dimmed, disabled, testID, accessibilityLabel, emojiScale = 0.5, reducedMotion, wiggleRef}: TileProps) {
  const press = useRef(new Animated.Value(1)).current;
  const {style: wiggleStyle, wiggle} = useWiggle();
  if (wiggleRef) wiggleRef.current = wiggle;
  const to = (v: number) => Animated.spring(press, {toValue: v, useNativeDriver: NATIVE_DRIVER, speed: 40, bounciness: 8}).start();
  const side = Math.max(MIN_TARGET, size);
  return (
    <Pulse active={!!highlighted} reducedMotion={reducedMotion}>
      <Animated.View style={[wiggleStyle, {transform: [{scale: press}]}]}>
        <Pressable
          testID={testID}
          accessibilityRole="button"
          accessibilityLabel={accessibilityLabel ?? label}
          disabled={disabled}
          onPress={onPress}
          onPressIn={() => to(0.94)}
          onPressOut={() => to(1)}
          style={[
            styles.tile,
            shadow(highlighted ? 10 : 5),
            {width: side, height: side, backgroundColor: background, opacity: dimmed ? 0.35 : 1},
            ring ? {borderColor: ring, borderWidth: Math.max(4, side * 0.05)} : null,
            highlighted ? {borderColor: colors.hint, borderWidth: Math.max(5, side * 0.06)} : null,
          ]}>
          {content ?? <Text style={{fontSize: side * emojiScale, lineHeight: side * emojiScale * 1.2}} allowFontScaling={false}>{emoji}</Text>}
          {label ? <Text numberOfLines={1} style={[styles.tileLabel, {fontSize: Math.max(15, side * 0.14)}]}>{label}</Text> : null}
        </Pressable>
      </Animated.View>
    </Pulse>
  );
}

/** Long-press control for adults only (exit, settings). Children see a quiet lock, never an "Exit" label. */
export function HoldButton({onHold, label = 'Grown-ups: hold', holdMs = 1200, testID}: {onHold: () => void; label?: string; holdMs?: number; testID?: string}) {
  const progress = useRef(new Animated.Value(0)).current;
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [holding, setHolding] = useState(false);
  useEffect(() => () => { if (timer.current) clearTimeout(timer.current); }, []);
  // The timer decides; the fill is only feedback (animations may be slowed or paused).
  const start = () => {
    setHolding(true);
    progress.setValue(0);
    Animated.timing(progress, {toValue: 1, duration: holdMs, easing: Easing.linear, useNativeDriver: false}).start();
    timer.current = setTimeout(() => { timer.current = null; setHolding(false); progress.setValue(0); onHold(); }, holdMs);
  };
  const cancel = () => {
    if (timer.current) { clearTimeout(timer.current); timer.current = null; }
    setHolding(false);
    progress.stopAnimation();
    progress.setValue(0);
  };
  return (
    <Pressable testID={testID} accessibilityRole="button" accessibilityLabel={label} accessibilityHint="Press and hold" onPressIn={start} onPressOut={cancel} style={styles.hold} hitSlop={8}>
      <Animated.View style={[styles.holdFill, {width: progress.interpolate({inputRange: [0, 1], outputRange: ['0%', '100%']})}]} />
      <Text style={styles.holdIcon}>🔒</Text>
      {holding ? <Text style={styles.holdText}>{label}</Text> : null}
    </Pressable>
  );
}

export function SpeakerButton({onPress, size = 64, testID = 'speaker'}: {onPress: () => void; size?: number; testID?: string}) {
  return (
    <Pressable testID={testID} accessibilityRole="button" accessibilityLabel="Hear it again" onPress={onPress} style={[styles.speaker, shadow(4), {width: size, height: size, borderRadius: size / 2}]} hitSlop={8}>
      <Text style={{fontSize: size * 0.45}}>🔊</Text>
    </Pressable>
  );
}

const BURST = ['⭐', '🌟', '✨', '⭐', '🎉', '⭐', '✨', '🌟', '⭐', '🎈'];

/** Short success burst. Celebration is immediate, visual and spoken (audio confirms completion). */
export function Celebration({text, reducedMotion}: {text: string; reducedMotion: boolean}) {
  const t = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    Animated.timing(t, {toValue: 1, duration: reducedMotion ? 1 : 900, easing: Easing.out(Easing.cubic), useNativeDriver: NATIVE_DRIVER}).start();
  }, [t, reducedMotion]);
  return (
    <View pointerEvents="none" style={styles.celebrate}>
      <Animated.View style={[StyleSheet.absoluteFill, styles.celebrateScrim, {opacity: t.interpolate({inputRange: [0, 0.3, 1], outputRange: [0, 1, 1]})}]} />
      {!reducedMotion && BURST.map((e, i) => {
        const angle = (i / BURST.length) * Math.PI * 2;
        const r = 190 + (i % 3) * 45;
        return (
          <Animated.Text key={i} style={[styles.burst, {
            opacity: t.interpolate({inputRange: [0, 0.2, 0.85, 1], outputRange: [0, 1, 1, 0]}),
            transform: [
              {translateX: t.interpolate({inputRange: [0, 1], outputRange: [0, Math.cos(angle) * r]})},
              {translateY: t.interpolate({inputRange: [0, 1], outputRange: [0, Math.sin(angle) * r]})},
              {scale: t.interpolate({inputRange: [0, 0.4, 1], outputRange: [0.3, 1.3, 1]})},
            ],
          }]}>{e}</Animated.Text>
        );
      })}
      <Animated.View style={[styles.celebrateCard, shadow(12), {transform: [{scale: t.interpolate({inputRange: [0, 0.5, 1], outputRange: [0.6, 1.08, 1]})}]}]}>
        <Text style={styles.celebrateStar}>⭐</Text>
        <Text style={styles.celebrateText}>{text}</Text>
      </Animated.View>
    </View>
  );
}

/** "Gro", the Grow21 guide: a friendly sprout. Drawn in SVG so it scales crisply on any screen. */
export function Mascot({size = 160, mood = 'happy'}: {size?: number; mood?: 'happy' | 'rest' | 'cheer'}) {
  const eyesClosed = mood === 'rest';
  return (
    <Svg width={size} height={size} viewBox="0 0 200 200" accessibilityLabel="Gro the sprout">
      <Path d="M100 62 C 88 30, 52 26, 40 40 C 56 58, 80 66, 100 62 Z" fill="#43C26E" />
      <Path d="M100 62 C 112 26, 150 20, 164 34 C 148 56, 122 66, 100 62 Z" fill="#5BD681" />
      <Path d="M100 64 L100 80" stroke="#2E9E55" strokeWidth={6} strokeLinecap="round" />
      <Ellipse cx="100" cy="128" rx="66" ry="60" fill="#7FDB8F" />
      <Ellipse cx="100" cy="140" rx="44" ry="34" fill="#B8F0C2" />
      {eyesClosed ? (
        <>
          <Path d="M70 118 Q 80 126 90 118" stroke={colors.ink} strokeWidth={5} fill="none" strokeLinecap="round" />
          <Path d="M110 118 Q 120 126 130 118" stroke={colors.ink} strokeWidth={5} fill="none" strokeLinecap="round" />
        </>
      ) : (
        <>
          <Circle cx="80" cy="116" r="9" fill={colors.ink} />
          <Circle cx="120" cy="116" r="9" fill={colors.ink} />
          <Circle cx="83" cy="113" r="3" fill="#fff" />
          <Circle cx="123" cy="113" r="3" fill="#fff" />
        </>
      )}
      <Circle cx="64" cy="134" r="8" fill="#FF9DB0" opacity={0.7} />
      <Circle cx="136" cy="134" r="8" fill="#FF9DB0" opacity={0.7} />
      <Path d={mood === 'cheer' ? 'M82 138 Q 100 160 118 138 Z' : 'M84 138 Q 100 152 116 138'} stroke={colors.ink} strokeWidth={5} fill={mood === 'cheer' ? '#E5546A' : 'none'} strokeLinecap="round" />
      {mood === 'cheer' ? (
        <>
          <Path d="M38 120 Q 18 96 28 78" stroke="#7FDB8F" strokeWidth={14} fill="none" strokeLinecap="round" />
          <Path d="M162 120 Q 182 96 172 78" stroke="#7FDB8F" strokeWidth={14} fill="none" strokeLinecap="round" />
        </>
      ) : null}
    </Svg>
  );
}

export function BigButton({title, onPress, color = colors.success, icon, testID, small}: {title: string; onPress: () => void; color?: string; icon?: string; testID?: string; small?: boolean}) {
  const press = useRef(new Animated.Value(1)).current;
  const to = (v: number) => Animated.spring(press, {toValue: v, useNativeDriver: NATIVE_DRIVER, speed: 40, bounciness: 8}).start();
  return (
    <Animated.View style={{transform: [{scale: press}]}}>
      <Pressable testID={testID} accessibilityRole="button" accessibilityLabel={title} onPress={onPress} onPressIn={() => to(0.95)} onPressOut={() => to(1)}
        style={[styles.big, shadow(8), {backgroundColor: color, minHeight: small ? 64 : 88, paddingHorizontal: small ? 28 : 44}]}>
        {icon ? <Text style={{fontSize: small ? 24 : 32, color: '#fff', fontFamily: fonts.display}}>{icon}</Text> : null}
        <Text style={[styles.bigText, {fontSize: small ? 22 : 30}]}>{title}</Text>
      </Pressable>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  tile: {borderRadius: radius.md, alignItems: 'center', justifyContent: 'center', borderWidth: 3, borderColor: 'transparent'},
  tileLabel: {fontFamily: fonts.strong, color: colors.ink, marginTop: -2},
  hold: {minWidth: 56, height: 56, borderRadius: 28, backgroundColor: 'rgba(255,255,255,0.9)', alignItems: 'center', justifyContent: 'center', flexDirection: 'row', paddingHorizontal: 14, overflow: 'hidden', borderWidth: 2, borderColor: colors.line},
  holdFill: {position: 'absolute', left: 0, top: 0, bottom: 0, backgroundColor: colors.sunSoft},
  holdIcon: {fontSize: 22},
  holdText: {fontFamily: fonts.strong, color: colors.inkSoft, fontSize: 15, marginLeft: 8},
  speaker: {backgroundColor: colors.paper, alignItems: 'center', justifyContent: 'center'},
  celebrate: {...StyleSheet.absoluteFillObject, alignItems: 'center', justifyContent: 'center', zIndex: 20},
  burst: {position: 'absolute', fontSize: 56, zIndex: 2},
  celebrateScrim: {backgroundColor: 'rgba(255,248,214,0.82)'},
  celebrateCard: {backgroundColor: colors.paper, borderRadius: radius.lg, paddingHorizontal: 56, paddingVertical: 30, alignItems: 'center', zIndex: 3, borderWidth: 5, borderColor: colors.sun},
  celebrateStar: {fontSize: 96},
  celebrateText: {fontFamily: fonts.display, fontSize: 44, color: colors.ink, marginTop: 6},
  big: {borderRadius: radius.pill, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 14},
  bigText: {fontFamily: fonts.display, color: '#fff'},
});
