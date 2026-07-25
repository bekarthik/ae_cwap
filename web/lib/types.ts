/**
 * TypeScript mirrors of the platform's v1 contracts.
 *
 * These are hand-maintained rather than generated because the canvas only needs
 * the graph, diagnosis, run and log shapes — a fraction of the registry. The
 * gateway rejects anything that drifts from the real schema, so a mismatch here
 * surfaces immediately as a 422 rather than silently corrupting a saved graph.
 */

export type NodeType =
  | 'input'
  | 'llm'
  | 'rag_retrieve'
  | 'http_request'
  | 'transform'
  | 'branch'
  | 'output';

export interface Position {
  x: number;
  y: number;
}

export interface WorkflowNode {
  id: string;
  type: NodeType;
  label: string;
  params: Record<string, unknown>;
  position: Position;
  knowledge_handle: string | null;
}

export interface WorkflowEdge {
  id: string;
  source: string;
  target: string;
  condition: boolean | null;
  bindings: Record<string, string>;
}

export interface WorkflowGraph {
  id: string;
  name: string;
  version: number;
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
}

export interface SkillRequirement {
  skill: string;
  node_type: NodeType;
  rationale: string;
}

export interface Diagnosis {
  intent: string;
  required_steps: SkillRequirement[];
  suggestions: string[];
  confidence: number;
  gaps: string[];
}

export interface ScaffoldResponse {
  diagnosis: Diagnosis;
  graph: WorkflowGraph;
}

export interface Session {
  access_token: string;
  token_type: string;
  expires_in: number;
  user_id: string;
  tenant_id: string;
  scopes: string[];
}

export interface WorkflowSummary {
  id: string;
  name: string;
  version: number;
  node_count: number;
  updated_at: string;
}

export interface KnowledgeSummary {
  handle: string;
  title: string;
  chunk_count: number;
  created_at: string;
}

export type RunStatus = 'PENDING' | 'RUNNING' | 'SUCCEEDED' | 'FAILED';

export interface RunSummary {
  run_id: string;
  workflow_id: string;
  status: RunStatus;
  steps_executed: number;
  started_at: string;
  finished_at: string | null;
  error: string | null;
}

export interface StepReport {
  node_id: string;
  source_service: string;
  resulting_state: string;
  output: Record<string, unknown>;
  derived_context: Record<string, unknown>;
  duration_ms: number;
  completed_at: string;
}

export interface LogEvent {
  run_id: string;
  seq: number;
  timestamp: string;
  node: string | null;
  level: 'DEBUG' | 'INFO' | 'WARN' | 'ERROR';
  event: string;
  message: string;
  data: Record<string, unknown>;
}

export interface RunReport {
  run: RunSummary;
  result: Record<string, unknown> | null;
  steps: StepReport[];
  logs: LogEvent[];
}

export interface RetrievedChunk {
  chunk_id: string;
  text: string;
  score: number;
  source_title: string;
}

export interface RetrievalResult {
  handle: string;
  query: string;
  chunks: RetrievedChunk[];
}
