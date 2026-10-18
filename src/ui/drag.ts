import {useRef} from 'react';
import type {GestureResponderEvent} from 'react-native';

export interface DragHandlers {
  /** Touch started on the element; location is relative to that element. */
  onStart?(location: {x: number; y: number}): void;
  onMove(delta: {dx: number; dy: number}): void;
  onEnd(delta: {dx: number; dy: number}): void;
}

/**
 * Responder props for a draggable element. The gesture must begin on the element itself
 * (recommended for users with Down syndrome) and the element keeps the touch until release.
 * Deltas come from page coordinates, so they stay correct while the finger leaves the element.
 */
export function useDrag(handlers: DragHandlers) {
  const latest = useRef(handlers);
  latest.current = handlers;
  const start = useRef<{x: number; y: number} | null>(null);
  const delta = (e: GestureResponderEvent) => ({
    dx: e.nativeEvent.pageX - (start.current?.x ?? e.nativeEvent.pageX),
    dy: e.nativeEvent.pageY - (start.current?.y ?? e.nativeEvent.pageY),
  });
  const finish = (e: GestureResponderEvent) => {
    if (!start.current) return;
    const d = delta(e);
    start.current = null;
    latest.current.onEnd(d);
  };
  return {
    onStartShouldSetResponder: () => true,
    onMoveShouldSetResponder: () => true,
    onResponderTerminationRequest: () => false,
    onResponderGrant: (e: GestureResponderEvent) => {
      start.current = {x: e.nativeEvent.pageX, y: e.nativeEvent.pageY};
      latest.current.onStart?.({x: e.nativeEvent.locationX, y: e.nativeEvent.locationY});
    },
    onResponderMove: (e: GestureResponderEvent) => {
      if (start.current) latest.current.onMove(delta(e));
    },
    onResponderRelease: finish,
    onResponderTerminate: finish,
  };
}
