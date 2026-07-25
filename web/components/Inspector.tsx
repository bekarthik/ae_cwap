'use client';

import { useMemo } from 'react';

import { kindFor, type CanvasEdge, type CanvasNode } from '@/lib/graph';
import type { KnowledgeSummary } from '@/lib/types';

interface Props {
  node: CanvasNode | null;
  edge: CanvasEdge | null;
  corpora: KnowledgeSummary[];
  onPatchNode: (id: string, patch: Partial<CanvasNode['data']>) => void;
  onPatchParams: (id: string, params: Record<string, unknown>) => void;
  onPatchEdgeBindings: (id: string, bindings: Record<string, string>) => void;
  onDeleteNode: (id: string) => void;
  onDeleteEdge: (id: string) => void;
}

const OPERATORS = [
  '==',
  '!=',
  '>',
  '>=',
  '<',
  '<=',
  'contains',
  'not_contains',
  'is_empty',
  'is_not_empty',
];

export function Inspector({
  node,
  edge,
  corpora,
  onPatchNode,
  onPatchParams,
  onPatchEdgeBindings,
  onDeleteNode,
  onDeleteEdge,
}: Props) {
  if (edge && !node) {
    return <EdgeInspector edge={edge} onPatch={onPatchEdgeBindings} onDelete={onDeleteEdge} />;
  }
  if (!node) {
    return (
      <div className="panel-section">
        <p className="panel-title">Inspector</p>
        <p className="muted small">
          Select a step to edit what it does, or select a connection to change which values
          flow along it.
        </p>
      </div>
    );
  }
  return (
    <NodeInspector
      node={node}
      corpora={corpora}
      onPatchNode={onPatchNode}
      onPatchParams={onPatchParams}
      onDelete={onDeleteNode}
    />
  );
}

function NodeInspector({
  node,
  corpora,
  onPatchNode,
  onPatchParams,
  onDelete,
}: {
  node: CanvasNode;
  corpora: KnowledgeSummary[];
  onPatchNode: Props['onPatchNode'];
  onPatchParams: Props['onPatchParams'];
  onDelete: (id: string) => void;
}) {
  const kind = kindFor(node.data.nodeType);
  const params = node.data.params ?? {};

  const set = (key: string, value: unknown) =>
    onPatchParams(node.id, { ...params, [key]: value });

  return (
    <div className="panel-section">
      <p className="panel-title">{kind.label} step</p>

      <div className="field">
        <label htmlFor="node-label">Name</label>
        <input
          id="node-label"
          value={node.data.label}
          onChange={(event) => onPatchNode(node.id, { label: event.target.value })}
        />
        <div className="hint">
          Step id <code>{node.id}</code> — used in bindings like{' '}
          <code>$steps.{node.id}.text</code>.
        </div>
      </div>

      {node.data.nodeType === 'input' ? (
        <>
          <div className="field">
            <label htmlFor="fields">Required inputs</label>
            <input
              id="fields"
              value={((params.fields as string[]) ?? []).join(', ')}
              onChange={(event) =>
                set(
                  'fields',
                  event.target.value
                    .split(',')
                    .map((item) => item.trim())
                    .filter(Boolean),
                )
              }
            />
            <div className="hint">
              Comma separated. The run will not start unless each has a value.
            </div>
          </div>
          <div className="field">
            <label htmlFor="default-goal">Default value for “goal”</label>
            <textarea
              id="default-goal"
              value={String(((params.defaults as Record<string, string>) ?? {}).goal ?? '')}
              onChange={(event) =>
                set('defaults', {
                  ...((params.defaults as Record<string, string>) ?? {}),
                  goal: event.target.value,
                })
              }
            />
          </div>
        </>
      ) : null}

      {node.data.nodeType === 'llm' ? (
        <>
          <div className="field">
            <label htmlFor="system">System instruction</label>
            <textarea
              id="system"
              value={String(params.system ?? '')}
              onChange={(event) => set('system', event.target.value)}
            />
          </div>
          <div className="field">
            <label htmlFor="prompt">Prompt</label>
            <textarea
              id="prompt"
              rows={6}
              value={String(params.prompt_template ?? '')}
              onChange={(event) => set('prompt_template', event.target.value)}
            />
            <div className="hint">
              Use <code>{'{{name}}'}</code> for any value arriving on an incoming
              connection — for example <code>{'{{goal}}'}</code> or{' '}
              <code>{'{{context}}'}</code>.
            </div>
          </div>
          <div className="field">
            <label htmlFor="effort">Reasoning effort</label>
            <select
              id="effort"
              value={String(params.effort ?? 'high')}
              onChange={(event) => set('effort', event.target.value)}
            >
              {['low', 'medium', 'high', 'xhigh', 'max'].map((level) => (
                <option key={level} value={level}>
                  {level}
                </option>
              ))}
            </select>
            <div className="hint">
              Higher effort means deeper reasoning, more tokens and more latency.
            </div>
          </div>
        </>
      ) : null}

      {node.data.nodeType === 'rag_retrieve' ? (
        <>
          <div className="field">
            <label htmlFor="corpus">Knowledge Context</label>
            <select
              id="corpus"
              value={node.data.knowledgeHandle ?? ''}
              onChange={(event) =>
                onPatchNode(node.id, { knowledgeHandle: event.target.value || null })
              }
            >
              <option value="">— select uploaded documents —</option>
              {corpora.map((corpus) => (
                <option key={corpus.handle} value={corpus.handle}>
                  {corpus.title} ({corpus.chunk_count} passages)
                </option>
              ))}
            </select>
            {corpora.length === 0 ? (
              <div className="hint">
                Upload documents in the Knowledge panel first — this step cannot run
                without one.
              </div>
            ) : null}
          </div>
          <div className="field">
            <label htmlFor="topk">Passages to retrieve</label>
            <input
              id="topk"
              type="number"
              min={1}
              max={20}
              value={Number(params.top_k ?? 4)}
              onChange={(event) => set('top_k', Number(event.target.value))}
            />
          </div>
          <div className="field">
            <label htmlFor="query">Search query</label>
            <textarea
              id="query"
              value={String(params.query_template ?? '{{goal}}')}
              onChange={(event) => set('query_template', event.target.value)}
            />
          </div>
        </>
      ) : null}

      {node.data.nodeType === 'transform' ? (
        <div className="field">
          <label htmlFor="template">Output template</label>
          <textarea
            id="template"
            rows={5}
            value={String(params.template ?? '')}
            onChange={(event) => set('template', event.target.value)}
          />
        </div>
      ) : null}

      {node.data.nodeType === 'branch' ? (
        <>
          <div className="field">
            <label htmlFor="left">Compare this</label>
            <input
              id="left"
              value={String(params.left ?? '')}
              onChange={(event) => set('left', event.target.value)}
            />
            <div className="hint">
              An expression such as <code>$output.text</code>.
            </div>
          </div>
          <div className="field">
            <label htmlFor="operator">Test</label>
            <select
              id="operator"
              value={String(params.operator ?? 'contains')}
              onChange={(event) => set('operator', event.target.value)}
            >
              {OPERATORS.map((operator) => (
                <option key={operator} value={operator}>
                  {operator}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor="right">Against this</label>
            <input
              id="right"
              value={String(params.right ?? '')}
              onChange={(event) => set('right', event.target.value)}
            />
          </div>
          <div className="notice notice--warn">
            Connect the green handle to the “yes” path and the red handle to the “no”
            path. Both are required.
          </div>
        </>
      ) : null}

      {node.data.nodeType === 'http_request' ? (
        <>
          <div className="field">
            <label htmlFor="method">Method</label>
            <select
              id="method"
              value={String(params.method ?? 'GET')}
              onChange={(event) => set('method', event.target.value)}
            >
              {['GET', 'POST', 'PUT', 'PATCH', 'DELETE'].map((method) => (
                <option key={method} value={method}>
                  {method}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor="url">URL</label>
            <input
              id="url"
              value={String(params.url ?? '')}
              onChange={(event) => set('url', event.target.value)}
            />
          </div>
          <div className="field">
            <label htmlFor="body">Request body</label>
            <textarea
              id="body"
              value={String(params.body ?? '')}
              onChange={(event) => set('body', event.target.value)}
            />
          </div>
          <div className="notice notice--warn">
            External calls need the <code>WRITE_EXTERNAL</code> scope, and the host must
            be allow-listed by an administrator. Each call runs at most once per run,
            even if the step is retried.
          </div>
        </>
      ) : null}

      {node.data.nodeType === 'output' ? (
        <div className="field">
          <label htmlFor="result">Result template</label>
          <textarea
            id="result"
            value={String(params.result_template ?? '')}
            onChange={(event) => set('result_template', event.target.value)}
          />
        </div>
      ) : null}

      <button className="btn btn--danger btn--block" onClick={() => onDelete(node.id)}>
        Delete step
      </button>
    </div>
  );
}

function EdgeInspector({
  edge,
  onPatch,
  onDelete,
}: {
  edge: CanvasEdge;
  onPatch: (id: string, bindings: Record<string, string>) => void;
  onDelete: (id: string) => void;
}) {
  const bindings = useMemo(
    () => ((edge.data ?? {}) as { bindings?: Record<string, string> }).bindings ?? {},
    [edge.data],
  );
  const entries = Object.entries(bindings);

  const update = (index: number, key: string, value: string) => {
    const next = entries.map(([existingKey, existingValue], position) =>
      position === index ? [key, value] : [existingKey, existingValue],
    );
    onPatch(edge.id, Object.fromEntries(next));
  };

  return (
    <div className="panel-section">
      <p className="panel-title">Connection</p>
      <p className="muted small">
        {edge.source} → {edge.target}
        {edge.sourceHandle ? ` (${edge.sourceHandle === 'true' ? 'yes' : 'no'} path)` : ''}
      </p>

      <div className="field">
        <label>Values passed along this connection</label>
        {entries.length === 0 ? (
          <p className="muted small">Nothing is passed yet.</p>
        ) : null}
        {entries.map(([key, value], index) => (
          <div className="binding-row" key={`${key}-${index}`}>
            <input
              aria-label="name"
              value={key}
              onChange={(event) => update(index, event.target.value, value)}
            />
            <input
              aria-label="expression"
              value={value}
              onChange={(event) => update(index, key, event.target.value)}
            />
            <button
              className="btn btn--ghost"
              aria-label={`remove ${key}`}
              onClick={() =>
                onPatch(
                  edge.id,
                  Object.fromEntries(entries.filter((_, position) => position !== index)),
                )
              }
            >
              ✕
            </button>
          </div>
        ))}
        <button
          className="btn btn--block"
          onClick={() => onPatch(edge.id, { ...bindings, value: '$output.text' })}
        >
          Add value
        </button>
        <div className="hint">
          The name on the left becomes <code>{'{{name}}'}</code> in the next step. The
          expression on the right reads from <code>$run.input.*</code>,{' '}
          <code>$output.*</code> (the previous step) or <code>$steps.&lt;id&gt;.*</code>.
        </div>
      </div>

      <button className="btn btn--danger btn--block" onClick={() => onDelete(edge.id)}>
        Remove connection
      </button>
    </div>
  );
}
