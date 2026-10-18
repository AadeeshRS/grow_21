import React from 'react';
import Svg, {Circle, Ellipse, G, Path, Rect} from 'react-native-svg';

/** Simple vector objects whose colour is part of the object (Audio Search asks for "the red car"). */
export const OBJECT_KINDS = ['balloon', 'ball', 'car', 'bus', 'fish', 'bird', 'cup', 'hat'] as const;
export type ObjectKind = (typeof OBJECT_KINDS)[number];

/** Look-alike pairs used when visual similarity is high. */
export const SIMILAR: Record<ObjectKind, ObjectKind> = {balloon: 'ball', ball: 'balloon', car: 'bus', bus: 'car', fish: 'bird', bird: 'fish', cup: 'hat', hat: 'cup'};

/** Natural default colour when colour is not part of the instruction. */
export const DEFAULT_FILL: Record<ObjectKind, string> = {balloon: '#F25C54', ball: '#3D7BF5', car: '#F25C54', bus: '#FFC531', fish: '#FF9F1C', bird: '#3D7BF5', cup: '#2DB862', hat: '#A879FF'};

const OUT = 'rgba(29,43,79,0.22)';

export function ColorObject({kind, fill, size}: {kind: ObjectKind; fill: string; size: number}) {
  const line = {stroke: OUT, strokeWidth: 3};
  return (
    <Svg width={size} height={size} viewBox="0 0 100 100">
      {kind === 'balloon' ? (
        <G>
          <Path d="M50 74 C 47 82, 54 88, 50 98" stroke="#5B6B8C" strokeWidth={2.5} fill="none" />
          <Ellipse cx={50} cy={40} rx={27} ry={33} fill={fill} {...line} />
          <Path d="M45 73 L55 73 L50 79 Z" fill={fill} {...line} />
          <Ellipse cx={40} cy={28} rx={6} ry={10} fill="#fff" opacity={0.55} />
        </G>
      ) : kind === 'ball' ? (
        <G>
          <Circle cx={50} cy={52} r={36} fill={fill} {...line} />
          <Path d="M16 46 C 36 58, 64 58, 84 46" stroke="#fff" strokeWidth={6} fill="none" opacity={0.85} />
          <Path d="M44 17 C 34 40, 34 66, 44 87" stroke="#fff" strokeWidth={6} fill="none" opacity={0.85} />
        </G>
      ) : kind === 'car' ? (
        <G>
          <Path d="M28 44 L36 28 L66 28 L76 44 Z" fill={fill} {...line} />
          <Rect x={8} y={42} width={84} height={28} rx={10} fill={fill} {...line} />
          <Path d="M38 32 L46 32 L46 43 L33 43 Z" fill="#CFE8FF" />
          <Path d="M52 32 L63 32 L70 43 L52 43 Z" fill="#CFE8FF" />
          <Circle cx={28} cy={72} r={10} fill="#2B3550" /><Circle cx={72} cy={72} r={10} fill="#2B3550" />
          <Circle cx={28} cy={72} r={4} fill="#C9D3E6" /><Circle cx={72} cy={72} r={4} fill="#C9D3E6" />
        </G>
      ) : kind === 'bus' ? (
        <G>
          <Rect x={6} y={24} width={88} height={50} rx={10} fill={fill} {...line} />
          {[14, 34, 54, 74].map(x => <Rect key={x} x={x} y={32} width={14} height={14} rx={3} fill="#CFE8FF" />)}
          <Rect x={6} y={54} width={88} height={6} fill="rgba(255,255,255,0.45)" />
          <Circle cx={26} cy={76} r={9} fill="#2B3550" /><Circle cx={74} cy={76} r={9} fill="#2B3550" />
        </G>
      ) : kind === 'fish' ? (
        <G>
          <Path d="M70 50 L94 32 L90 50 L94 68 Z" fill={fill} {...line} />
          <Ellipse cx={44} cy={50} rx={32} ry={21} fill={fill} {...line} />
          <Path d="M40 34 C 46 44, 46 56, 40 66" stroke="rgba(255,255,255,0.6)" strokeWidth={4} fill="none" />
          <Circle cx={26} cy={45} r={5} fill="#fff" /><Circle cx={25} cy={45} r={2.5} fill="#1D2B4F" />
        </G>
      ) : kind === 'bird' ? (
        <G>
          <Ellipse cx={46} cy={58} rx={30} ry={22} fill={fill} {...line} />
          <Circle cx={68} cy={38} r={15} fill={fill} {...line} />
          <Path d="M82 36 L94 40 L82 44 Z" fill="#FF9F1C" />
          <Path d="M28 54 C 40 46, 56 50, 58 62 C 46 66, 34 64, 28 54 Z" fill="rgba(0,0,0,0.15)" />
          <Circle cx={71} cy={35} r={3.5} fill="#1D2B4F" />
          <Path d="M40 80 L38 90 M52 80 L54 90" stroke="#FF9F1C" strokeWidth={3} />
        </G>
      ) : kind === 'cup' ? (
        <G>
          <Path d="M72 40 C 90 40, 90 64, 70 64" stroke={fill} strokeWidth={8} fill="none" />
          <Path d="M20 26 L78 26 L70 84 L28 84 Z" fill={fill} {...line} />
          <Rect x={20} y={24} width={58} height={8} rx={4} fill="rgba(255,255,255,0.5)" />
        </G>
      ) : (
        <G>
          <Ellipse cx={50} cy={72} rx={42} ry={11} fill={fill} {...line} />
          <Path d="M26 70 C 26 34, 34 22, 50 22 C 66 22, 74 34, 74 70 Z" fill={fill} {...line} />
          <Rect x={26} y={56} width={48} height={8} fill="rgba(255,255,255,0.5)" />
        </G>
      )}
    </Svg>
  );
}
