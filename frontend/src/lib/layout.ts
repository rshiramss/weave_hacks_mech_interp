// Hierarchical top-to-bottom layout for a flow's nodes via dagre.

import dagre from "@dagrejs/dagre";
import type { FlowNodeSpec } from "./types";

export const NODE_WIDTH = 220;
export const NODE_HEIGHT = 72;
// The roster ("agent") node lists five specialists, so it needs more vertical room.
export const AGENT_NODE_HEIGHT = 188;

export function nodeHeight(kind: string): number {
  return kind === "agent" ? AGENT_NODE_HEIGHT : NODE_HEIGHT;
}

export interface Positioned {
  id: string;
  x: number;
  y: number;
}

export function layoutFlow(
  nodes: FlowNodeSpec[],
  edges: [string, string][],
): Positioned[] {
  const graph = new dagre.graphlib.Graph();
  graph.setGraph({ rankdir: "TB", nodesep: 28, ranksep: 44, marginx: 12, marginy: 12 });
  graph.setDefaultEdgeLabel(() => ({}));

  nodes.forEach((n) =>
    graph.setNode(n.id, { width: NODE_WIDTH, height: nodeHeight(n.kind) }),
  );
  edges.forEach(([from, to]) => graph.setEdge(from, to));
  dagre.layout(graph);

  return nodes.map((n) => {
    const { x, y } = graph.node(n.id);
    return { id: n.id, x: x - NODE_WIDTH / 2, y: y - nodeHeight(n.kind) / 2 };
  });
}
