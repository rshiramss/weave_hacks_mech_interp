import { useMemo } from "react";
import {
  ReactFlow,
  Background,
  type Edge,
  type Node,
  type NodeTypes,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { StepNode } from "./StepNode";
import { AgentNode } from "./AgentNode";
import { layoutFlow } from "../lib/layout";
import type { ArmFlow, ArmRunState } from "../lib/types";

// Defined once at module scope — React Flow warns if nodeTypes is recreated.
const nodeTypes: NodeTypes = { step: StepNode, agent: AgentNode };

// The display-only roster node has no runner step; it's "done" once a sub-agent
// has been chosen so the surrounding edges light up like the real steps.
const AGENT_NODE_ID = "route_agent";

interface FlowGraphProps {
  flow: ArmFlow;
  run?: ArmRunState;
  execute: boolean;
  accent: string;
  agents: Record<string, string>;
}

export function FlowGraph({ flow, run, execute, accent, agents }: FlowGraphProps) {
  const { nodes, edges } = useMemo(() => {
    const chosenAgent = run?.chosenAgent;
    const isDone = (id: string): boolean =>
      id === AGENT_NODE_ID
        ? !!chosenAgent
        : run?.steps[id]?.status === "done";

    const visNodes = flow.nodes.filter((n) => execute || !n.execute_only);
    const ids = new Set(visNodes.map((n) => n.id));
    const visEdges = flow.edges.filter(([a, b]) => ids.has(a) && ids.has(b));
    const positions = layoutFlow(visNodes, visEdges);
    const posById = new Map(positions.map((p) => [p.id, p]));

    const rfNodes: Node[] = visNodes.map((n) => {
      const p = posById.get(n.id)!;
      const base = { id: n.id, position: { x: p.x, y: p.y }, draggable: false, connectable: false };
      if (n.kind === "agent") {
        return {
          ...base,
          type: "agent",
          data: { accent, agents, chosenAgent, candidates: run?.agentCandidates },
        };
      }
      return {
        ...base,
        type: "step",
        data: {
          label: n.label,
          kind: n.kind,
          accent,
          state: run?.steps[n.id],
          current: run?.current === n.id,
        },
      };
    });

    const rfEdges: Edge[] = visEdges.map(([a, b]) => ({
      id: `${a}-${b}`,
      source: a,
      target: b,
      animated: run?.current === b,
      style: {
        stroke: isDone(b) ? accent : "var(--border)",
        strokeWidth: 1.5,
      },
    }));

    return { nodes: rfNodes, edges: rfEdges };
  }, [flow, run, execute, accent, agents]);

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      fitView
      fitViewOptions={{ padding: 0.15 }}
      nodesDraggable={false}
      nodesConnectable={false}
      elementsSelectable={false}
      panOnScroll={false}
      zoomOnScroll={false}
      preventScrolling={false}
      proOptions={{ hideAttribution: true }}
    >
      <Background color="var(--border)" gap={20} size={1} />
    </ReactFlow>
  );
}
