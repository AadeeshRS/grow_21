import type {ComponentType} from 'react';
import type {GameConfig, GameId} from '../types';
import {AudioSearchBoard, searchInstruction} from './AudioSearch';
import {BlockStackerBoard, blockInstruction} from './BlockStacker';
import {BubblePopBoard, bubbleInstruction} from './BubblePop';
import {CategorySortBoard, sortInstruction} from './CategorySort';
import {EchoSafariBoard, echoInstruction} from './EchoSafari';
import {FlipMatchBoard, flipInstruction} from './FlipMatch';
import {PatternTrainBoard, patternInstruction} from './PatternTrain';
import {SentenceStripsBoard, sentenceInstruction} from './SentenceStrips';
import {TraceGlowBoard, traceInstruction} from './TraceGlow';
import type {BoardProps, Instruction} from './types';

export const BOARDS: Record<GameId, ComponentType<BoardProps>> = {
  trace: TraceGlowBoard,
  bubble: BubblePopBoard,
  block: BlockStackerBoard,
  echo: EchoSafariBoard,
  sentence: SentenceStripsBoard,
  audio: AudioSearchBoard,
  sort: CategorySortBoard,
  flip: FlipMatchBoard,
  pattern: PatternTrainBoard,
};

export function instructionFor(gameId: GameId, config: GameConfig, seed: number): Instruction {
  switch (gameId) {
    case 'trace': return traceInstruction();
    case 'bubble': return bubbleInstruction(config);
    case 'block': return blockInstruction();
    case 'echo': return echoInstruction();
    case 'sentence': return sentenceInstruction(config, seed);
    case 'audio': return searchInstruction(config, seed);
    case 'sort': return sortInstruction(config, seed);
    case 'flip': return flipInstruction();
    case 'pattern': return patternInstruction();
  }
}

export type {AttemptResult, BoardProps, Instruction} from './types';
export {seedFromString} from './random';
