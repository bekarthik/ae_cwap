/**
 * The only module that talks to the API Gateway.
 *
 * Every call goes through `request`, so authentication, error translation and
 * the base URL live in exactly one place. When the gateway rejects a payload
 * with a contract violation, the structured `reason_code` is preserved rather
 * than collapsed into a generic "request failed".
 */

import type {
  KnowledgeSummary,
  RuntimeInfo,
  RetrievalResult,
  RunReport,
  RunSummary,
  ScaffoldResponse,
  Session,
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

  diagnose: (goal: string, knowledgeHandles: string[] = []) =>
    request<ScaffoldResponse>('/api/diagnose', {
      method: 'POST',
      body: JSON.stringify({ goal, knowledge_handles: knowledgeHandles }),
    }),

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
