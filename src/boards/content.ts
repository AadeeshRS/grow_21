/**
 * Picture content for the nine games. Familiar everyday items, one clear meaning per picture,
 * always paired with a short spoken/written word (pictures are the primary channel).
 */

export interface Item { emoji: string; word: string }

export const ANIMAL_SOUNDS: ReadonlyArray<Item & {sound: string}> = [
  {emoji: '🐮', word: 'cow', sound: 'Moo'},
  {emoji: '🐶', word: 'dog', sound: 'Woof'},
  {emoji: '🐱', word: 'cat', sound: 'Meow'},
  {emoji: '🦆', word: 'duck', sound: 'Quack'},
  {emoji: '🐑', word: 'sheep', sound: 'Baa'},
  {emoji: '🦁', word: 'lion', sound: 'Roar'},
  {emoji: '🦉', word: 'owl', sound: 'Hoo hoo'},
  {emoji: '🐝', word: 'bee', sound: 'Buzz'},
];

export const COLORS = [
  {name: 'red', fill: '#FF6B6B', soft: '#FFE1E1'},
  {name: 'blue', fill: '#4D8DFF', soft: '#DDE9FF'},
  {name: 'green', fill: '#2EBD6B', soft: '#D9F6E6'},
  {name: 'yellow', fill: '#FFC531', soft: '#FFF2C7'},
] as const;
export type ColorName = (typeof COLORS)[number]['name'];

/** Picture sentences, grouped by length; vocab 1 = most familiar words. */
export interface Sentence { tokens: Item[]; vocab: 1 | 2 | 3 }
const T = (emoji: string, word: string): Item => ({emoji, word});
export const SENTENCES: Record<number, Sentence[]> = {
  2: [
    {vocab: 1, tokens: [T('🐶', 'dog'), T('🏃', 'runs')]},
    {vocab: 1, tokens: [T('👶', 'baby'), T('😴', 'sleeps')]},
    {vocab: 1, tokens: [T('🐦', 'bird'), T('🪽', 'flies')]},
    {vocab: 2, tokens: [T('🐸', 'frog'), T('🦘', 'jumps')]},
    {vocab: 3, tokens: [T('👧', 'girl'), T('💃', 'dances')]},
  ],
  3: [
    {vocab: 1, tokens: [T('🐱', 'cat'), T('😋', 'eats'), T('🐟', 'fish')]},
    {vocab: 1, tokens: [T('👦', 'boy'), T('🦵', 'kicks'), T('⚽', 'ball')]},
    {vocab: 2, tokens: [T('👧', 'girl'), T('🥤', 'drinks'), T('🥛', 'milk')]},
    {vocab: 2, tokens: [T('🐶', 'dog'), T('🔍', 'finds'), T('🦴', 'bone')]},
    {vocab: 3, tokens: [T('👩', 'mum'), T('📖', 'reads'), T('📚', 'book')]},
  ],
  4: [
    {vocab: 1, tokens: [T('🐱', 'cat'), T('😋', 'eats'), T('🔴', 'red'), T('🍎', 'apple')]},
    {vocab: 2, tokens: [T('👦', 'boy'), T('🦵', 'kicks'), T('🔵', 'blue'), T('⚽', 'ball')]},
    {vocab: 3, tokens: [T('👧', 'girl'), T('🤗', 'hugs'), T('🟤', 'brown'), T('🧸', 'teddy')]},
  ],
  5: [
    {vocab: 1, tokens: [T('😊', 'happy'), T('🐶', 'dog'), T('😋', 'eats'), T('🔴', 'red'), T('🍎', 'apple')]},
    {vocab: 2, tokens: [T('😪', 'sleepy'), T('👶', 'baby'), T('🤗', 'hugs'), T('🟤', 'brown'), T('🧸', 'teddy')]},
    {vocab: 3, tokens: [T('😊', 'happy'), T('👧', 'girl'), T('🥤', 'drinks'), T('⚪', 'white'), T('🥛', 'milk')]},
  ],
};
export const SENTENCE_DISTRACTORS: Item[] = [T('🚗', 'car'), T('🌳', 'tree'), T('🍌', 'banana'), T('🛏️', 'bed'), T('☀️', 'sun'), T('🧦', 'sock'), T('🎈', 'balloon'), T('🐢', 'turtle')];

/** Sorting pairs from easy (very different categories) to hard (close categories). */
export interface SortPair { similarity: 0 | 1 | 2; left: {name: string; icon: string; items: Item[]}; right: {name: string; icon: string; items: Item[]} }
export const SORT_PAIRS: SortPair[] = [
  {similarity: 0, left: {name: 'Food', icon: '🍽️', items: [T('🍎', 'apple'), T('🍌', 'banana'), T('🍞', 'bread'), T('🧀', 'cheese'), T('🥕', 'carrot'), T('🍪', 'cookie')]},
    right: {name: 'Clothes', icon: '👕', items: [T('👕', 'shirt'), T('👖', 'jeans'), T('🧦', 'socks'), T('👟', 'shoe'), T('🧢', 'cap'), T('🧤', 'gloves')]}},
  {similarity: 0, left: {name: 'Animals', icon: '🐾', items: [T('🐶', 'dog'), T('🐱', 'cat'), T('🐮', 'cow'), T('🐷', 'pig'), T('🐔', 'hen'), T('🐴', 'horse')]},
    right: {name: 'Vehicles', icon: '🚦', items: [T('🚗', 'car'), T('🚌', 'bus'), T('🚲', 'bike'), T('🚂', 'train'), T('✈️', 'plane'), T('🚤', 'boat')]}},
  {similarity: 1, left: {name: 'Fruit', icon: '🧺', items: [T('🍎', 'apple'), T('🍌', 'banana'), T('🍇', 'grapes'), T('🍓', 'strawberry'), T('🍐', 'pear'), T('🍊', 'orange')]},
    right: {name: 'Vegetables', icon: '🥗', items: [T('🥕', 'carrot'), T('🥦', 'broccoli'), T('🌽', 'corn'), T('🥔', 'potato'), T('🍅', 'tomato'), T('🥒', 'cucumber')]}},
  {similarity: 2, left: {name: 'Fly', icon: '☁️', items: [T('🐦', 'bird'), T('🦋', 'butterfly'), T('🐝', 'bee'), T('🦉', 'owl'), T('🦅', 'eagle'), T('🪰', 'fly')]},
    right: {name: 'Swim', icon: '🌊', items: [T('🐟', 'fish'), T('🐬', 'dolphin'), T('🐙', 'octopus'), T('🦀', 'crab'), T('🐳', 'whale'), T('🐢', 'turtle')]}},
];

/** Pattern and pairs sets: each group holds look-alike pictures; groups differ strongly from each other. */
export const LOOKALIKE_GROUPS: Item[][] = [
  [T('🍎', 'red apple'), T('🍏', 'green apple'), T('🍒', 'cherry'), T('🍅', 'tomato')],
  [T('🐶', 'dog'), T('🐺', 'wolf'), T('🦊', 'fox'), T('🐱', 'cat')],
  [T('🚗', 'car'), T('🚕', 'taxi'), T('🚙', 'jeep'), T('🚓', 'police car')],
  [T('⭐', 'star'), T('🌟', 'gold star'), T('☀️', 'sun'), T('🌼', 'flower')],
  [T('🔵', 'blue'), T('🟢', 'green'), T('🟣', 'purple'), T('🟡', 'yellow')],
  [T('🐟', 'fish'), T('🐠', 'stripy fish'), T('🐡', 'puffer fish'), T('🐬', 'dolphin')],
  [T('🎈', 'balloon'), T('🎁', 'present'), T('🎂', 'cake'), T('🧸', 'teddy')],
];

/** Block shapes for Block Stacker. */
export const BLOCK_SHAPES = ['square', 'circle', 'triangle', 'star', 'heart'] as const;
export type BlockShape = (typeof BLOCK_SHAPES)[number];
export const BLOCK_COLORS = ['#FF6B6B', '#4D8DFF', '#2EBD6B', '#FFC531', '#A879FF'] as const;
