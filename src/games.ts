import {clamp,lerp,roundLerp} from './config'; import type {GameConfig,GameId,SkillId} from './types';
export const SKILLS:Record<GameId,SkillId>={trace:'fine_motor_tracing',bubble:'motor_targeting',block:'drag_precision',echo:'vocal_activation',sentence:'sentence_sequencing',audio:'receptive_language',sort:'classification',flip:'visual_working_memory',pattern:'visual_sequencing'};
export const GAME_NAMES:Record<GameId,string>={trace:'Trace & Glow',bubble:'Bubble Pop Brigade',block:'Block Stacker',echo:'Echo Safari',sentence:'Picture-Sentence Strips',audio:'Audio Search',sort:'Category Sorting',flip:'Visual Flip Match',pattern:'Pattern Train'};
export function difficultyConfig(game:GameId,D:number):GameConfig { const d=clamp(D); switch(game){
case 'trace': return {pathWidthDp:lerp(120,44,d),pathLengthDp:lerp(220,620,d),curvature:lerp(0,.8,d),turnCount:roundLerp(0,6,d),shapeType:d<.2?'straight':d<.5?'gentleCurve':d<.8?'zigzag':'shapes'};
case 'bubble': return {bubbleSizeDp:lerp(140,70,d),speedDpPerSec:lerp(30,90,d),concurrentBubbles:roundLerp(1,5,d),minSeparationDp:lerp(200,80,d),popTarget:5};
case 'block': return {blockCount:roundLerp(1,5,d),blockSizeDp:lerp(110,60,d),placementToleranceDp:lerp(56,18,d),structure:d<.25?'single':d<.5?'row':d<.75?'stack':'pattern'};
case 'echo': return {promptVariety:roundLerp(2,6,d),repetitions:roundLerp(1,3,d),responseWindowMs:lerp(8000,4000,d),demand:d<.7?'single':'twoSound'};
case 'sentence': return {sentenceLength:roundLerp(2,5,d),vocabLevel:d<.34?1:d<.67?2:3,distractorCount:roundLerp(0,4,d),structureType:d<.4?'SV':d<.75?'SVO':'SVOmodifier'};
case 'audio': return {optionCount:roundLerp(2,3,d),visualSimilarity:lerp(0,.7,d),attributeCount:roundLerp(1,3,d),instructionComplexity:roundLerp(1,3,d)};
case 'sort': return {itemCount:roundLerp(3,8,d),categoryCount:2,categorySimilarity:lerp(0,.6,d),attributeComplexity:d<.6?1:2,placementToleranceDp:lerp(64,28,d)};
case 'flip': return {cardCount:d<.5?4:6,visualSimilarity:lerp(0,.6,d),exposureMs:lerp(4000,1500,d),loadLevel:d<.34?1:d<.67?2:3};
case 'pattern': return {sequenceLength:roundLerp(3,7,d),ruleCount:d<.4?1:d<.75?2:3,distractorCount:roundLerp(1,4,d),visualSimilarity:lerp(0,.6,d)}; }}
export function applyScaffold(game:GameId, base:GameConfig, level:number):GameConfig { const l=Math.max(0,Math.min(3,Math.floor(level))); const c={...base}; if(!l)return c;
if(game==='trace'){c.pathWidthDp=Number(c.pathWidthDp)*1.2;c.strongGlow=true;if(l>=2)c.resumeRadiusMultiplier=2;if(l>=3){c.ghostTrail=true;c.maximumTolerance=true}}
if(game==='bubble'){c.bubbleSizeDp=Number(c.bubbleSizeDp)*1.15;c.speedDpPerSec=Number(c.speedDpPerSec)*.8;if(l>=2){c.concurrentBubbles=Math.min(2,Number(c.concurrentBubbles));c.hint=true}if(l>=3){c.concurrentBubbles=1;c.hitPaddingDp=16;c.speedDpPerSec=30}}
if(game==='block'){c.blockSizeDp=Number(c.blockSizeDp)*1.15;c.placementToleranceDp=Number(c.placementToleranceDp)*1.3;if(l>=2){c.snapMultiplier=2;c.highlight=true}if(l>=3){c.blockCount=Math.min(2,Number(c.blockCount));c.pulseTarget=true}}
if(game==='echo'){c.repeatModel=l>=3?2:1;if(l>=2){c.strongCue=true;c.responseWindowMs=Number(c.responseWindowMs)*1.5}if(l>=3)c.repetitions=1}
if(game==='sentence'){c.distractorCount=Math.floor(Number(c.distractorCount)*.5);if(l>=2){c.highlightNext=true;c.sentenceLength=Math.max(2,Number(c.sentenceLength)-1)}if(l>=3){c.workedExample=true;c.distractorCount=0}}
if(game==='audio'){c.repeatInstruction=true;if(l>=2){c.optionCount=2;c.visualSimilarity=0}if(l>=3){c.attributeCount=1;c.instructionComplexity=1;c.instructionRepeats=2}}
if(game==='sort'){c.clearLabels=true;c.placementToleranceDp=Number(c.placementToleranceDp)*1.25;if(l>=2){c.itemCount=Math.min(4,Number(c.itemCount));c.example=true}if(l>=3){c.highlightBin=true;c.snapToBin=true}}
if(game==='flip'){c.exposureMs=Number(c.exposureMs)*1.5;if(l>=2){c.cardCount=4;c.visualSimilarity=0}if(l>=3){c.demo=true;c.cardCount=4}}
if(game==='pattern'){c.distractorCount=Math.max(1,Number(c.distractorCount)-1);if(l>=2){c.highlightUnit=true;c.sequenceLength=Math.max(3,Number(c.sequenceLength)-1)}if(l>=3){c.answerChoices=2;c.animateUnit=true}} return c; }
