/**
 * Translation between React Flow's canvas model and the platform's
 * `WorkflowGraph` contract.
 *
 * React Flow owns presentation (selection, dragging, handles); the contract
 * owns meaning (node params, edge bindings, branch conditions). Keeping the
 * mapping in one file is what stops canvas concerns from leaking into the saved
 * graph, and vice versa.
 */

import type { Edge, Node } from '@xyflow/react';

import type { NodeType, WorkflowEdge, WorkflowGraph, WorkflowNode } from './types';

export interface NodeData extends Record<string, unknown> {
  label: string;
  nodeType: NodeType;
  params: Record<string, unknown>;
  knowledgeHandle: string | null;
  agentId: string | null;
}

export type CanvasNode = Node<NodeData>;
export type CanvasEdge = Edge;

export interface NodeKind {
  type: NodeType;
  label: string;
  blurb: string;
  icon: string;
  accent: string;
  defaultParams: Record<string, unknown>;
}

/** The palette. Ordered the way a workflow usually reads. */
export const NODE_KINDS: NodeKind[] = [
  {
    type: 'input',
    label: 'Start',
    blurb: 'What the workflow is given when it runs.',
    icon: '▶',
    accent: '#2563eb',
    defaultParams: { fields: ['goal'], defaults: {} },
  },
  {
    type: 'agent',
    label: 'Agent',
    blurb: 'A role with its own skills and memory. It decides how to get there.',
    icon: '◆',
    accent: '#4f46e5',
    defaultParams: { objective_template: 'Achieve this goal:\n{{goal}}' },
  },
  {
    type: 'llm',
    label: 'Think',
    blurb: 'One model call. Use an Agent when the step needs to decide anything.',
    icon: '✳',
    accent: '#7c3aed',
    // Both knobs are carried deliberately: the backend honours what it can
    // and reports the rest as ignored, so one workflow runs on any model.
    defaultParams: {
      system: 'You are a careful assistant.',
      prompt_template: '{{goal}}',
      effort: 'high',
      temperature: 0.7,
      max_tokens: 4096,
    },
  },
  {
    type: 'rag_retrieve',
    label: 'Knowledge',
    blurb: 'Pull relevant passages from your documents.',
    icon: '◎',
    accent: '#0d9488',
    defaultParams: { top_k: 4, query_template: '{{query}}' },
  },
  {
    type: 'transform',
    label: 'Format',
    blurb: 'Reshape the previous result into a template.',
    icon: '⇄',
    accent: '#ca8a04',
    defaultParams: { template: '{{previous}}' },
  },
  {
    type: 'branch',
    label: 'Decision',
    blurb: 'Take one of two paths depending on a value.',
    icon: '⋔',
    accent: '#db2777',
    defaultParams: { left: '$output.text', operator: 'contains', right: '' },
  },
  {
    type: 'http_request',
    label: 'Call service',
    blurb: 'Call an external API. Requires the WRITE_EXTERNAL scope.',
    icon: '⇗',
    accent: '#dc2626',
    defaultParams: { method: 'GET', url: '', headers: {} },
  },
  {
    type: 'output',
    label: 'Result',
    blurb: 'What the workflow hands back when it finishes.',
    icon: '■',
    accent: '#475569',
    defaultParams: { result_template: '{{previous}}' },
  },
];

export function kindFor(type: NodeType): NodeKind {
  return NODE_KINDS.find((kind) => kind.type === type) ?? NODE_KINDS[1];
}

/** Steps that do the work, as opposed to declaring the run's edges. */
export function isWorkingNode(type: NodeType): boolean {
  return type !== 'input' && type !== 'output';
}

export function newId(prefix: string): string {
  const random = Math.random().toString(36).slice(2, 8);
  return `${prefix}_${Date.now().toString(36)}${random}`;
}

export function emptyGraph(): WorkflowGraph {
  const inputId = 'input';
  const outputId = 'output';
  return {
    id: newId('wf'),
    name: 'Untitled workflow',
    version: 1,
    memory_scope_id: null,
    nodes: [
      {
        id: inputId,
        type: 'input',
        label: 'Start',
        params: { fields: ['goal'], defaults: {} },
        position: { x: 40, y: 160 },
        knowledge_handle: null,
        agent_id: null,
      },
      {
        id: outputId,
        type: 'output',
        label: 'Result',
        params: { result_template: '{{previous}}' },
        position: { x: 460, y: 160 },
        knowledge_handle: null,
        agent_id: null,
      },
    ],
    edges: [
      {
        id: 'e_input_output',
        source: inputId,
        target: outputId,
        condition: null,
        bindings: { previous: '$run.input.goal' },
      },
    ],
  };
}

export function toCanvas(graph: WorkflowGraph): {
  nodes: CanvasNode[];
  edges: CanvasEdge[];
} {
  return {
    nodes: graph.nodes.map((node) => ({
      id: node.id,
      type: 'workflowNode',
      position: node.position,
      data: {
        label: node.label || kindFor(node.type).label,
        nodeType: node.type,
        params: node.params ?? {},
        knowledgeHandle: node.knowledge_handle,
        agentId: node.agent_id,
      },
    })),
    edges: graph.edges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      // A branch node's two outgoing edges are distinguished by handle id, which
      // is how the canvas keeps "yes" and "no" visually separate.
      sourceHandle: edge.condition === null ? null : edge.condition ? 'true' : 'false',
      label: edge.condition === null ? undefined : edge.condition ? 'yes' : 'no',
      animated: false,
      data: { bindings: edge.bindings, condition: edge.condition },
    })),
  };
}

export function toGraph(
  base: Pick<WorkflowGraph, 'id' | 'name' | 'version' | 'memory_scope_id'>,
  nodes: CanvasNode[],
  edges: CanvasEdge[],
): WorkflowGraph {
  return {
    ...base,
    nodes: nodes.map((node) => ({
      id: node.id,
      type: node.data.nodeType,
      label: node.data.label,
      params: node.data.params ?? {},
      position: { x: Math.round(node.position.x), y: Math.round(node.position.y) },
      knowledge_handle: node.data.knowledgeHandle ?? null,
      agent_id: node.data.agentId ?? null,
    })),
    edges: edges.map((edge) => {
      const data = (edge.data ?? {}) as {
        bindings?: Record<string, string>;
        condition?: boolean | null;
      };
      const condition =
        edge.sourceHandle === 'true'
          ? true
          : edge.sourceHandle === 'false'
            ? false
            : (data.condition ?? null);
      return {
        id: edge.id,
        source: edge.source,
        target: edge.target,
        condition,
        bindings: data.bindings ?? {},
      } satisfies WorkflowEdge;
    }),
  };
}

/**
 * Client-side pre-flight.
 *
 * The gateway is the authority — it re-validates everything — but reporting
 * these here means the user sees the problem next to the node that causes it
 * instead of as a save failure.
 */
export function describeProblems(graph: WorkflowGraph): string[] {
  const problems: string[] = [];
  if (graph.nodes.length === 0) return ['The canvas is empty — add a Start node.'];

  const targets = new Set(graph.edges.map((edge) => edge.target));
  const roots = graph.nodes.filter((node) => !targets.has(node.id));
  if (roots.length === 0) {
    problems.push('Every node has an incoming connection, so there is no place to start.');
  } else if (roots.length > 1) {
    problems.push(
      `There are ${roots.length} possible starting points (${roots
        .map((node) => node.label || node.id)
        .join(', ')}). A workflow needs exactly one.`,
    );
  }

  for (const node of graph.nodes) {
    const outgoing = graph.edges.filter((edge) => edge.source === node.id);
    if (node.type === 'branch') {
      const conditions = new Set(outgoing.map((edge) => edge.condition));
      if (!conditions.has(true) || !conditions.has(false)) {
        problems.push(`"${node.label || node.id}" needs both a yes and a no path.`);
      }
    } else if (node.type === 'output') {
      if (outgoing.length) {
        problems.push(`"${node.label || node.id}" is a Result node and cannot lead anywhere.`);
      }
    } else if (outgoing.length > 1) {
      problems.push(
        `"${node.label || node.id}" leads to ${outgoing.length} places. Only a Decision node can branch.`,
      );
    }
    if (node.type === 'rag_retrieve' && !node.knowledge_handle) {
      problems.push(`"${node.label || node.id}" needs a Knowledge Context selected.`);
    }
    if (node.type === 'agent' && !node.agent_id) {
      problems.push(
        `"${node.label || node.id}" has no agent assigned. Pick one, or describe the goal again and let the system design it.`,
      );
    }
  }

  return problems;
}

/** Suggests a sensible binding when the user draws a new connection. */
export function defaultBindings(sourceType: NodeType, targetType: NodeType): Record<string, string> {
  if (sourceType === 'input') {
    return targetType === 'rag_retrieve'
      ? { query: '$run.input.goal' }
      : { goal: '$run.input.goal' };
  }
  if (sourceType === 'rag_retrieve') {
    return { goal: '$run.input.goal', context: '$output.context' };
  }
  if (sourceType === 'branch') {
    return { previous: '$output.evaluated' };
  }
  // An agent almost always wants the original goal alongside the previous
  // step's work — otherwise a later agent only sees its predecessor's answer
  // and has to infer what was being asked.
  if (targetType === 'agent') {
    return { goal: '$run.input.goal', previous: '$output.text' };
  }
  return { previous: '$output.text' };
}
