// Opens an SSE connection to /api/triage/stream and folds events into per-arm
// run state. One active run at a time; starting a new run closes the previous.

import { useCallback, useEffect, useRef, useState } from "react";
import { reduceEvent } from "../lib/flowState";
import type { ArmRunState, SSEEvent } from "../lib/types";

type ArmMap = Record<string, ArmRunState>;

export interface RunController {
  arms: ArmMap;
  running: boolean;
  start: (query: string, arms: string[], execute: boolean) => void;
}

export function useRunStream(): RunController {
  const [arms, setArms] = useState<ArmMap>({});
  const [running, setRunning] = useState(false);
  const sourceRef = useRef<EventSource | null>(null);

  useEffect(() => () => sourceRef.current?.close(), []);

  const start = useCallback((query: string, selected: string[], execute: boolean) => {
    sourceRef.current?.close();

    const init: ArmMap = {};
    selected.forEach((arm) => (init[arm] = { status: "running", steps: {} }));
    setArms(init);
    setRunning(true);

    const url =
      `/api/triage/stream?query=${encodeURIComponent(query)}` +
      `&arms=${selected.join(",")}&execute=${execute}`;
    const source = new EventSource(url);
    sourceRef.current = source;

    source.onmessage = (e) => {
      const event = JSON.parse(e.data) as SSEEvent;
      if (event.kind === "run_done") {
        source.close();
        setRunning(false);
        return;
      }
      setArms((prev) => reduceEvent(prev, event));
    };

    source.onerror = () => {
      source.close();
      setRunning(false);
    };
  }, []);

  return { arms, running, start };
}
