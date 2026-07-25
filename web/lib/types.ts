/**
 * TypeScript mirrors of the platform's v2 contracts.
 *
 * These are hand-maintained rather than generated because the canvas only needs
 * the graph, design, agent, skill, memory, run and log shapes — a fraction of
 * the registry. The gateway rejects anything that drifts from the real schema,
 * so a mismatch here surfaces immediately as a 422 rather than silently
 * corrupting a saved graph.
 */

export type NodeType =
  | 'input'
  | 'agent'
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
  /** Set on agent nodes: which stored agent staffs this step. */
  agent_id: string | null;
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
  /** The scope every agent in this workflow shares knowledge through. */
  memory_scope_id: string | null;
}

/* -- the design conversation ------------------------------------------- */

export type DesignStage = 'clarifying' | 'designed';

export interface ClarifyingQuestion {
  id: string;
  question: string;
  /** Why it is being asked. An unexplained question is an interrogation. */
  why: string;
  options: string[];
  default: string;
}

export interface PlannedAgent {
  name: string;
  role: string;
  objective: string;
  skills: string[];
  rationale: string;
}

export interface SkillGap {
  needed_by: string;
  capability: string;
  /** Non-empty when the platform will not build it without a human. */
  blocked_reason: string;
}

export interface DesignResponse {
  stage: DesignStage;
  understanding: string;
  questions: ClarifyingQuestion[];
  agents: PlannedAgent[];
  skill_gaps: SkillGap[];
  graph: WorkflowGraph | null;
  notes: string[];
}

/* -- agents, skills and memory ----------------------------------------- */

export interface AgentMemoryConfig {
  recall: boolean;
  write_learnings: boolean;
  use_workflow_memory: boolean;
  recall_limit: number;
}

export interface AgentDefinition {
  id: string;
  tenant_id: string;
  name: string;
  role: string;
  objective: string;
  instructions: string;
  skill_ids: string[];
  max_iterations: number;
  memory: AgentMemoryConfig;
  model_override: string | null;
  version: number;
}

export type SkillKind = 'prompt' | 'retrieval' | 'http' | 'transform' | 'composite';
export type SkillOrigin = 'builtin' | 'user' | 'synthesized';

export interface SkillParameter {
  name: string;
  type: string;
  description: string;
  required: boolean;
}

export interface SkillDefinition {
  id: string;
  tenant_id: string;
  name: string;
  description: string;
  kind: SkillKind;
  parameters: SkillParameter[];
  definition: Record<string, unknown>;
  origin: SkillOrigin;
  version: number;
  invocations: number;
  failures: number;
}

export type MemoryScope = 'workflow' | 'agent' | 'skill' | 'run';
export type MemoryKind = 'fact' | 'preference' | 'learning' | 'failure' | 'artifact';

export interface MemoryEntry {
  id: string;
  tenant_id: string;
  scope: MemoryScope;
  scope_id: string;
  kind: MemoryKind;
  text: string;
  source_run_id: string | null;
  /** Accumulated trust. Rises when a run goes well, falls when it does not. */
  usefulness: number;
  created_at: string;
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

/** What the configured model backend can actually do. */
export interface ProviderSupport {
  effort: boolean;
  temperature: boolean;
  top_p: boolean;
  stop_sequences: boolean;
  system_prompt: boolean;
  /** Native tool calling. Without it, agents run on the prompted protocol. */
  tools: boolean;
  vision: boolean;
  thinking: boolean;
}

export interface LlmInfo {
  configured: boolean;
  provider: string;
  label?: string;
  model?: string;
  base_url?: string;
  deterministic?: boolean;
  notes?: string;
  error?: string;
  supports: ProviderSupport;
}

export interface EmbeddingInfo {
  configured: boolean;
  provider: string;
  identity?: string;
  dimensions?: number;
  semantic?: boolean;
  error?: string;
}

/** One model the picker can offer, and what it can actually do. */
export interface ModelCard {
  id: string;
  label: string;
  context: number;
  notes: string;
  open_weights: boolean;
  supports: { tools: boolean; vision: boolean; thinking: boolean };
}

export interface AvailableProvider {
  key: string;
  label: string;
  requires_key: boolean;
  local: boolean;
  base_url: string;
  default_model: string;
  notes: string;
  models: ModelCard[];
}

export interface RuntimeInfo {
  llm: LlmInfo;
  embeddings: EmbeddingInfo;
  effort_levels: string[];
  available_providers: AvailableProvider[];
}
