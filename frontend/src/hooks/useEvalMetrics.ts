import { useCallback, useEffect, useState } from "react";
import { fetchMetrics } from "../lib/api";
import type { MetricsResponse } from "../lib/types";

export function useEvalMetrics() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback((refresh = false) => {
    setLoading(true);
    fetchMetrics(refresh)
      .then(setMetrics)
      .catch((e) => setMetrics({ arms: {}, source: "error", error: String(e) }))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => load(false), [load]);

  return { metrics, loading, reload: () => load(true) };
}
