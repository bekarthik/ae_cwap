/**
 * The only module that talks to the API Gateway.
 *
 * Every call goes through `request`, so authentication, error translation and
 * the base URL live in exactly one place. When the gateway rejects a payload
 * with a contract violation, the structured `reason_code` is preserved rather
 * than collapsed into a generic "request failed".
 */

import type {
  AgentDefinition,
  MCPPolicy,
  MCPServerConfig,
  MCPServerRecord,
  MCPTool,
  DetectedModel,
  ModelConfiguration,
  DesignResponse,
  KnowledgeSummary,
  MemoryEntry,
  MemoryKind,
  MemoryScope,
  RuntimeInfo,
  RetrievalResult,
  RunReport,
  RunSummary,
  Session,
  SkillDefinition,
  WorkflowGraph,
  WorkflowSummary,
} from './types';

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? 'http://localhost:8000';

const TOKEN_KEY = 'cwap.session';

export class ApiError extends Error {
  readonly status: number;
  readonly reasonCode?: string;
  readonly detail?: unknown;

  constructor(status: number, message: string, reasonCode?: string, detail?: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.reasonCode = reasonCode;
    this.detail = detail;
  }
}

export function loadSession(): Session | null {
  if (typeof window === 'undefined') return null;
  const raw = window.localStorage.getItem(TOKEN_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as Session;
  } catch {
    window.localStorage.removeItem(TOKEN_KEY);
    return null;
  }
}

export function storeSession(session: Session): void {
  window.localStorage.setItem(TOKEN_KEY, JSON.stringify(session));
}

export function clearSession(): void {
  window.localStorage.removeItem(TOKEN_KEY);
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const session = loadSession();
  const headers = new Headers(init.headers);
  if (session) headers.set('Authorization', `Bearer ${session.access_token}`);
  if (init.body && !(init.body instanceof FormData)) {
    headers.set('Content-Type', 'application/json');
  }

  const response = await fetch(`${API_BASE}${path}`, { ...init, headers });

  if (response.status === 204) return undefined as T;

  const text = await response.text();
  const body = text ? safeParse(text) : null;

  if (!response.ok) {
    throw new ApiError(
      response.status,
      extractMessage(body) ?? `${response.status} ${response.statusText}`,
      typeof body === 'object' && body !== null
        ? (body as Record<string, string>).reason_code
        : undefined,
      body,
    );
  }

  return body as T;
}

function safeParse(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

/** FastAPI reports validation failures as a list under `detail`. */
function extractMessage(body: unknown): string | null {
  if (typeof body === 'string') return body;
  if (typeof body !== 'object' || body === null) return null;
  const record = body as Record<string, unknown>;

  if (typeof record.message === 'string') return record.message;
  if (typeof record.detail === 'string') return record.detail;

  if (Array.isArray(record.detail)) {
    const parts = record.detail
      .map((item) => {
        if (typeof item !== 'object' || item === null) return null;
        const entry = item as { loc?: unknown[]; msg?: string };
        const where = Array.isArray(entry.loc) ? entry.loc.slice(1).join('.') : '';
        return where ? `${where}: ${entry.msg}` : entry.msg ?? null;
      })
      .filter(Boolean);
    if (parts.length) return parts.join('; ');
  }
  return null;
}

export const api = {
  /** What model backend this deployment runs, and which knobs it honours. */
  runtime: () => request<RuntimeInfo>('/api/runtime'),

  /** The active model, this workspace's saved choice, and everything on offer. */
  modelConfig: () => request<ModelConfiguration>('/api/models'),

  /** Ask an endpoint what it actually serves, instead of asking the user. */
  detectModels: (provider: string, baseUrl = '', apiKey = '') =>
    request<{ ok: boolean; error?: string; models: DetectedModel[] }>('/api/models/detect', {
      method: 'POST',
      body: JSON.stringify({ provider, base_url: baseUrl, api_key: apiKey }),
    }),

  /** One real completion, so a broken configuration is caught before it is saved. */
  testModel: (provider: string, model: string, baseUrl = '', apiKey = '') =>
    request<{ ok: boolean; message: string; model: string; latency_ms: number }>(
      '/api/models/test',
      {
        method: 'POST',
        body: JSON.stringify({ provider, model, base_url: baseUrl, api_key: apiKey }),
      },
    ),

  /** `apiKey: null` keeps the stored credential; `''` clears it. */
  saveModel: (provider: string, model: string, baseUrl = '', apiKey: string | null = null) =>
    request<{ stored: unknown; active: unknown }>('/api/models', {
      method: 'PUT',
      body: JSON.stringify({ provider, model, base_url: baseUrl, api_key: apiKey }),
    }),

  clearModel: () => request<{ cleared: boolean }>('/api/models', { method: 'DELETE' }),

  register: (email: string, password: string, tenantId = 'default') =>
    request<Session>('/api/auth/register', {
      method: 'POST',
      body: JSON.stringify({ email, password, tenant_id: tenantId }),
    }),

  login: (email: string, password: string) =>
    request<Session>('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    }),

  /**
   * One turn of the design conversation: with no answers it returns the
   * questions it needs, with answers it returns the finished design.
   */
  design: (
    goal: string,
    answers: Record<string, string> = {},
    knowledgeHandles: string[] = [],
  ) =>
    request<DesignResponse>('/api/design', {
      method: 'POST',
      body: JSON.stringify({ goal, answers, knowledge_handles: knowledgeHandles }),
    }),

  /** Design from defaults without asking anything. */
  designDirect: (goal: string, knowledgeHandles: string[] = []) =>
    request<DesignResponse>('/api/design/direct', {
      method: 'POST',
      body: JSON.stringify({ goal, answers: {}, knowledge_handles: knowledgeHandles }),
    }),

  listAgents: () => request<AgentDefinition[]>('/api/agents'),

  getAgent: (id: string) => request<AgentDefinition>(`/api/agents/${id}`),

  updateAgent: (id: string, agent: Omit<AgentDefinition, 'id' | 'tenant_id' | 'version'>) =>
    request<AgentDefinition>(`/api/agents/${id}`, {
      method: 'PUT',
      body: JSON.stringify(agent),
    }),

  deleteAgent: (id: string) => request<void>(`/api/agents/${id}`, { method: 'DELETE' }),

  listSkills: () => request<SkillDefinition[]>('/api/skills'),

  /** Find a skill for a capability, creating one if none exists. */
  ensureSkill: (capability: string, context = '') =>
    request<{ skill: SkillDefinition; created: boolean }>('/api/skills/ensure', {
      method: 'POST',
      body: JSON.stringify({ capability, context }),
    }),

  deleteSkill: (id: string) => request<void>(`/api/skills/${id}`, { method: 'DELETE' }),

  /* -- MCP connectors -------------------------------------------------- */

  listMcpServers: () =>
    request<{ servers: MCPServerRecord[]; policy: MCPPolicy }>('/api/mcp'),

  /** Try a server without storing anything. */
  testMcpServer: (config: MCPServerConfig, credentials: Record<string, string> = {}) =>
    request<{ ok: boolean; message: string; tools: MCPTool[]; blocked: boolean }>(
      '/api/mcp/test',
      { method: 'POST', body: JSON.stringify({ config, credentials }) },
    ),

  /** Verify, store, and import the server's tools as skills. */
  connectMcpServer: (body: {
    name: string;
    description: string;
    config: MCPServerConfig;
    credentials: Record<string, string>;
  }) =>
    request<{ ok: boolean; message: string; tools: MCPTool[] }>('/api/mcp', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  /** Re-read the tool list, picking up what the server gained or lost. */
  syncMcpServer: (id: string) =>
    request<{ skills: number; server: MCPServerRecord }>(`/api/mcp/${id}/sync`, {
      method: 'POST',
    }),

  setMcpServerEnabled: (id: string, enabled: boolean) =>
    request<MCPServerRecord>(`/api/mcp/${id}/enabled?enabled=${enabled}`, {
      method: 'POST',
    }),

  disconnectMcpServer: (id: string) =>
    request<void>(`/api/mcp/${id}`, { method: 'DELETE' }),

  listMemory: (scope: MemoryScope, scopeId: string) =>
    request<MemoryEntry[]>(
      `/api/memory?scope=${scope}&scope_id=${encodeURIComponent(scopeId)}`,
    ),

  /** Tell an agent, a skill or a workflow something directly. */
  teach: (scope: MemoryScope, scopeId: string, text: string, kind: MemoryKind = 'preference') =>
    request<MemoryEntry>('/api/memory', {
      method: 'POST',
      body: JSON.stringify({ scope, scope_id: scopeId, text, kind }),
    }),

  forgetMemory: (entryId: string) =>
    request<void>(`/api/memory/${entryId}`, { method: 'DELETE' }),

  forgetScope: (scope: MemoryScope, scopeId: string) =>
    request<{ forgotten: number }>(
      `/api/memory?scope=${scope}&scope_id=${encodeURIComponent(scopeId)}`,
      { method: 'DELETE' },
    ),

  listWorkflows: () => request<WorkflowSummary[]>('/api/workflows'),

  getWorkflow: (id: string) =>
    request<{ id: string; name: string; version: number; graph: WorkflowGraph }>(
      `/api/workflows/${id}`,
    ),

  saveWorkflow: (graph: WorkflowGraph) =>
    request<{ id: string; version: number; graph: WorkflowGraph }>(
      `/api/workflows/${graph.id}`,
      { method: 'PUT', body: JSON.stringify({ graph }) },
    ),

  deleteWorkflow: (id: string) =>
    request<void>(`/api/workflows/${id}`, { method: 'DELETE' }),

  listKnowledge: () => request<KnowledgeSummary[]>('/api/knowledge'),

  ingestText: (title: string, content: string) =>
    request<KnowledgeSummary>('/api/knowledge/text', {
      method: 'POST',
      body: JSON.stringify({ title, content }),
    }),

  uploadDocument: (file: File) => {
    const form = new FormData();
    form.append('file', file);
    return request<KnowledgeSummary>('/api/knowledge/upload', {
      method: 'POST',
      body: form,
    });
  },

  previewRetrieval: (handle: string, query: string, topK = 4) =>
    request<RetrievalResult>(`/api/knowledge/${handle}/preview`, {
      method: 'POST',
      body: JSON.stringify({ query, top_k: topK }),
    }),

  deleteKnowledge: (handle: string) =>
    request<void>(`/api/knowledge/${handle}`, { method: 'DELETE' }),

  startRun: (workflowId: string, inputs: Record<string, unknown>, graph?: WorkflowGraph) =>
    request<RunSummary>(`/api/workflows/${workflowId}/runs`, {
      method: 'POST',
      body: JSON.stringify({ inputs, graph: graph ?? null }),
    }),

  getRun: (runId: string) => request<RunReport>(`/api/runs/${runId}`),

  listRuns: (workflowId?: string) =>
    request<RunSummary[]>(
      workflowId ? `/api/runs?workflow_id=${encodeURIComponent(workflowId)}` : '/api/runs',
    ),
};

/**
 * Live log feed for a run.
 *
 * The token travels as a query parameter because a browser cannot set headers
 * on a WebSocket handshake; the gateway verifies the same signed JWT either way.
 */
export function openLogSocket(runId: string): WebSocket | null {
  const session = loadSession();
  if (!session) return null;
  const url = new URL(`${API_BASE}/api/runs/${runId}/logs`);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  url.searchParams.set('token', session.access_token);
  return new WebSocket(url.toString());
}
