'use client';

import { Handle, Position, type NodeProps } from '@xyflow/react';
import { memo } from 'react';

import { kindFor, type CanvasNode } from '@/lib/graph';

/** Per-node run state, pushed in from the live log stream. */
export type NodeRunState = 'idle' | 'running' | 'done' | 'error';

function summarise(nodeType: string, params: Record<string, unknown>): string {
  switch (nodeType) {
    case 'input': {
      const fields = (params.fields as string[] | undefined) ?? [];
      return fields.length ? fields.join(', ') : 'no inputs declared';
    }
    case 'agent':
      return String(params.objective_template ?? '(no objective set)');
    case 'llm':
      return String(params.prompt_template ?? '(no prompt set)');
    case 'rag_retrieve':
      return `top ${params.top_k ?? 4} passages`;
    case 'transform':
      return String(params.template ?? '');
    case 'branch':
      return `${params.left ?? '?'} ${params.operator ?? '=='} ${
        params.right === '' ? '""' : (params.right ?? '?')
      }`;
    case 'http_request':
      return `${params.method ?? 'GET'} ${params.url || '(no url)'}`;
    case 'output':
      return String(params.result_template ?? '{{previous}}');
    default:
      return '';
  }
}

function WorkflowNodeCardImpl({ data, selected }: NodeProps<CanvasNode>) {
  const kind = kindFor(data.nodeType);
  const runState = (data.runState as NodeRunState | undefined) ?? 'idle';
  const summary = summarise(data.nodeType, data.params ?? {});

  const classes = [
    'wf-node',
    selected ? 'is-selected' : '',
    runState === 'running' ? 'is-running' : '',
    runState === 'done' ? 'is-done' : '',
    runState === 'error' ? 'is-error' : '',
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <div className={classes}>
      {data.nodeType !== 'input' ? (
        <Handle type="target" position={Position.Left} />
      ) : null}

      <div className="wf-node__head">
        <span className="palette-icon" style={{ background: kind.accent }}>
          {kind.icon}
        </span>
        <span>
          <div className="wf-node__kind">{kind.label}</div>
          <div className="wf-node__title">{data.label}</div>
        </span>
      </div>

      <div className="wf-node__body">
        {summary.length > 110 ? `${summary.slice(0, 110)}…` : summary}
        {data.nodeType === 'rag_retrieve' ? (
          <div style={{ marginTop: 6 }}>
            <span className="wf-node__badge">
              {data.knowledgeHandle ?? 'no corpus linked'}
            </span>
          </div>
        ) : null}
        {/* An agent node without an agent cannot run, so say so on the card
            rather than only at save time. */}
        {data.nodeType === 'agent' ? (
          <div style={{ marginTop: 6 }}>
            <span className="wf-node__badge">
              {data.agentId ? 'has skills & memory' : 'no agent assigned'}
            </span>
          </div>
        ) : null}
      </div>

      {/* A decision node exposes two labelled source handles; every other node
          has a single outgoing connection point. */}
      {data.nodeType === 'branch' ? (
        <>
          <Handle
            type="source"
            id="true"
            position={Position.Right}
            style={{ top: '35%', background: 'var(--success)' }}
          />
          <Handle
            type="source"
            id="false"
            position={Position.Right}
            style={{ top: '70%', background: 'var(--danger)' }}
          />
        </>
      ) : data.nodeType !== 'output' ? (
        <Handle type="source" position={Position.Right} />
      ) : null}
    </div>
  );
}

export const WorkflowNodeCard = memo(WorkflowNodeCardImpl);
