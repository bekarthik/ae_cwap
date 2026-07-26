'use client';

import { useEffect, useMemo, useState } from 'react';

import { api } from '@/lib/api';
import { kindFor, type CanvasEdge, type CanvasNode } from '@/lib/graph';
import type {
  AgentDefinition,
  KnowledgeSummary,
  ModelConfiguration,
  ProviderOption,
  RuntimeInfo,
  SkillDefinition,
} from '@/lib/types';

import { MemoryPanel } from './MemoryPanel';

interface Props {
  node: CanvasNode | null;
  edge: CanvasEdge | null;
  corpora: KnowledgeSummary[];
  /** Drives which generation controls are shown — see GenerationControls. */
  runtime: RuntimeInfo | null;
  agents: AgentDefinition[];
  skills: SkillDefinition[];
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
  runtime,
  agents,
  skills,
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
      runtime={runtime}
      agents={agents}
      skills={skills}
      onPatchNode={onPatchNode}
      onPatchParams={onPatchParams}
      onDelete={onDeleteNode}
    />
  );
}

function NodeInspector({
  node,
  corpora,
  runtime,
  agents,
  skills,
  onPatchNode,
  onPatchParams,
  onDelete,
}: {
  node: CanvasNode;
  corpora: KnowledgeSummary[];
  runtime: RuntimeInfo | null;
  agents: AgentDefinition[];
  skills: SkillDefinition[];
  onPatchNode: Props['onPatchNode'];
  onPatchParams: Props['onPatchParams'];
  onDelete: (id: string) => void;
}) {
  const kind = kindFor(node.data.nodeType);
  const params = node.data.params ?? {};

  // Both take the whole patch in one call, because `params` here is this
  // render's snapshot: two `set` calls in one handler would each spread the
  // *old* params and the second would silently undo the first. Every existing
  // caller happened to set one key per event, so the trap was there and unsprung
  // until a control needed to change a provider and clear its model together.
  const patch = (changes: Record<string, unknown>) =>
    onPatchParams(node.id, { ...params, ...changes });

  const set = (key: string, value: unknown) => patch({ [key]: value });

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

      {node.data.nodeType === 'agent' ? (
        <AgentControls
          node={node}
          agents={agents}
          skills={skills}
          runtime={runtime}
          onPatchNode={onPatchNode}
          set={set}
          patch={patch}
        />
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
          <GenerationControls params={params} runtime={runtime} set={set} />
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
              value={String(params.query_template ?? '{{query}}')}
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

/**
 * Editing what an agent node is, and seeing what it has become.
 *
 * The step's *objective* lives on the node — the same agent can be pointed at a
 * different job in a different workflow — while its role, skills and memory
 * belong to the agent itself and are shown read-only here, with a link into the
 * roster to change them. Keeping that split visible is what stops a user from
 * editing a shared agent while believing they are editing one step.
 */
function AgentControls({
  node,
  agents,
  skills,
  runtime,
  onPatchNode,
  set,
  patch,
}: {
  node: CanvasNode;
  agents: AgentDefinition[];
  skills: SkillDefinition[];
  runtime: RuntimeInfo | null;
  onPatchNode: Props['onPatchNode'];
  set: (key: string, value: unknown) => void;
  patch: (changes: Record<string, unknown>) => void;
}) {
  const params = node.data.params ?? {};
  const assigned = agents.find((agent) => agent.id === node.data.agentId) ?? null;
  const toolsNative = runtime?.llm.supports?.tools ?? true;
  // Where that answer came from. The platform must not tell someone their model
  // cannot call tools on the strength of a guess — only when the server said so
  // ("observed"), the model is one we have curated ("catalogue"), or an operator
  // forced the prompted protocol ("configured").
  const toolSource = runtime?.llm.tool_support ?? 'assumed';

  return (
    <>
      <div className="field">
        <label htmlFor="agent">Who does this step</label>
        <select
          id="agent"
          value={node.data.agentId ?? ''}
          onChange={(event) => onPatchNode(node.id, { agentId: event.target.value || null })}
        >
          <option value="">— no agent assigned —</option>
          {agents.map((agent) => (
            <option key={agent.id} value={agent.id}>
              {agent.name}
            </option>
          ))}
        </select>
        {agents.length === 0 ? (
          <div className="hint">
            No agents yet. Describe a goal and the system will design and staff the
            whole workflow for you.
          </div>
        ) : null}
      </div>

      <StepModelControls node={node} agent={assigned} patch={patch} />

      <ReviewControls node={node} agents={agents} onPatchNode={onPatchNode} />

      <div className="field">
        <label htmlFor="objective">What it should achieve here</label>
        <textarea
          id="objective"
          rows={5}
          value={String(params.objective_template ?? '')}
          onChange={(event) => set('objective_template', event.target.value)}
        />
        <div className="hint">
          Use <code>{'{{name}}'}</code> for any value arriving on an incoming
          connection — usually <code>{'{{goal}}'}</code> and{' '}
          <code>{'{{previous}}'}</code>.
        </div>
      </div>

      {assigned ? (
        <>
          <div className="field">
            <label>Its role</label>
            <p className="small muted" style={{ margin: 0 }}>
              {assigned.role}
            </p>
          </div>

          <div className="field">
            <label>What it can do</label>
            <div className="row" style={{ flexWrap: 'wrap', gap: 4 }}>
              {assigned.skill_ids.length === 0 ? (
                <span className="muted small">No skills attached.</span>
              ) : (
                assigned.skill_ids.map((id) => (
                  <span className="chip" key={id}>
                    {skills.find((skill) => skill.id === id)?.name.replace(/_/g, ' ') ?? id}
                  </span>
                ))
              )}
            </div>
            <div className="hint">
              It picks which of these to use, and in what order. Up to{' '}
              {assigned.max_iterations} attempts before it must answer with what it
              has.
            </div>
          </div>

          {!toolsNative && toolSource !== 'assumed' ? (
            <div className="notice notice--info">
              {toolSource === 'observed'
                ? `${runtime?.llm.model ?? 'This model'} rejected a tool-calling request, so skills are being offered through a prompted protocol instead.`
                : toolSource === 'configured'
                  ? 'This deployment is set to offer skills through the prompted protocol rather than native tool calling.'
                  : `${runtime?.llm.model ?? 'This model'} does not support native tool calling, so skills are offered through a prompted protocol instead.`}{' '}
              It works; a model with native tools follows a multi-step plan more
              reliably.
            </div>
          ) : null}

          <MemoryPanel
            scope="agent"
            scopeId={assigned.id}
            title="What this agent has learned"
          />
        </>
      ) : null}
    </>
  );
}

/**
 * Only the knobs the configured backend actually honours.
 *
 * Current Claude models reject `temperature` outright; open models have no
 * notion of `effort`. Showing both and silently dropping one would make the
 * canvas lie about what a step does, so the controls follow the backend's
 * declared capabilities — while a value saved for a *different* backend is
 * kept (a workflow stays portable) and flagged as inactive here.
 */
function GenerationControls({
  params,
  runtime,
  set,
}: {
  params: Record<string, unknown>;
  runtime: RuntimeInfo | null;
  set: (key: string, value: unknown) => void;
}) {
  const supports = runtime?.llm.supports;
  const label = runtime?.llm.label ?? 'the configured model';
  // Before capabilities load, show everything rather than flash a partial form.
  const showEffort = supports ? supports.effort : true;
  const showTemperature = supports ? supports.temperature : true;

  return (
    <>
      {showEffort ? (
        <div className="field">
          <label htmlFor="effort">Reasoning effort</label>
          <select
            id="effort"
            value={String(params.effort ?? 'high')}
            onChange={(event) => set('effort', event.target.value)}
          >
            {(runtime?.effort_levels ?? ['low', 'medium', 'high', 'xhigh', 'max']).map(
              (level) => (
                <option key={level} value={level}>
                  {level}
                </option>
              ),
            )}
          </select>
          <div className="hint">
            Higher effort means deeper reasoning, more tokens and more latency.
          </div>
        </div>
      ) : null}

      {showTemperature ? (
        <div className="field">
          <label htmlFor="temperature">
            Temperature {Number(params.temperature ?? 0.7).toFixed(2)}
          </label>
          <input
            id="temperature"
            type="range"
            min={0}
            max={2}
            step={0.05}
            value={Number(params.temperature ?? 0.7)}
            onChange={(event) => set('temperature', Number(event.target.value))}
          />
          <div className="hint">Lower is more predictable, higher is more varied.</div>
        </div>
      ) : null}

      <div className="field">
        <label htmlFor="max-tokens">Maximum output tokens</label>
        <input
          id="max-tokens"
          type="number"
          min={64}
          max={128000}
          step={64}
          value={Number(params.max_tokens ?? 4096)}
          onChange={(event) => set('max_tokens', Number(event.target.value))}
        />
      </div>

      {supports && !showTemperature && params.temperature !== undefined ? (
        <div className="notice notice--warn">
          This step has a temperature saved, but {label} does not accept one. It is
          ignored here and used again if you switch models.
        </div>
      ) : null}
      {supports && !showEffort && params.effort ? (
        <div className="notice notice--warn">
          This step has a reasoning effort saved, but {label} does not support it. It is
          ignored here and used again if you switch models.
        </div>
      ) : null}
    </>
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

/**
 * A quality loop on one step: another agent reviews the answer, this one
 * revises, until the reviewer approves or the rounds run out.
 *
 * The rounds are bounded and the repetition happens *inside* the step, which is
 * why the canvas stays acyclic — an edge looping back to an earlier node would
 * make termination depend on a model's judgement rather than on the structure,
 * and a workflow whose cost cannot be bounded before it runs is one nobody can
 * safely press Run on.
 */
const STEP_EFFORTS = ['', 'low', 'medium', 'high', 'xhigh', 'max'] as const;

/**
 * Which model runs *this step*.
 *
 * The agent editor already answered "which model does this agent use", and that
 * is the wrong grain for a decision as local as "the summarise step in this one
 * workflow needs the big model". Changing the agent changes every workflow that
 * uses it; the only way to vary one step was to clone the agent, which splits
 * its memory in two and quietly makes both halves worse.
 *
 * So this overrides the agent for one node, and nowhere else. Three levels, each
 * inheriting from the next when left blank: step → agent → workspace. That
 * ordering is the whole feature, and the hint under the control says which level
 * is actually in force rather than making anyone deduce it.
 */
function StepModelControls({
  node,
  agent,
  patch,
}: {
  node: CanvasNode;
  agent: AgentDefinition | null;
  patch: (changes: Record<string, unknown>) => void;
}) {
  const params = node.data.params ?? {};
  const provider = String(params.model_provider ?? '');
  const model = String(params.model_override ?? '');
  const effort = String(params.thinking_effort ?? '');

  const [config, setConfig] = useState<ModelConfiguration | null>(null);
  useEffect(() => {
    void api
      .modelConfig()
      .then(setConfig)
      .catch(() => setConfig(null));
  }, []);

  const chosen: ProviderOption | null =
    config?.providers.find((entry) => entry.key === provider) ?? null;
  const reachable =
    !provider || !chosen?.requires_key || (config?.credentials ?? []).includes(provider);

  // What actually runs if this step is left alone. Worth stating: an agent with
  // its own backend is not obvious from a canvas node, and someone wondering
  // why a step is slow should not have to open the agent to find out.
  const inherited = agent?.model_provider
    ? `${agent.name} runs on ${agent.model_provider}${
        agent.model_override ? ` · ${agent.model_override}` : ''
      }`
    : "the workspace's model";

  return (
    <div className="field">
      <label htmlFor="step-provider">Model for this step</label>
      <div className="row">
        <select
          id="step-provider"
          value={provider}
          // Clearing the model alongside the provider is the point of taking a
          // whole patch: a model id belongs to the backend that serves it, and
          // carrying `llama3.1` over to Anthropic would be a request that fails.
          onChange={(event) =>
            patch({ model_provider: event.target.value, model_override: '' })
          }
        >
          <option value="">— whatever the agent uses —</option>
          {(config?.providers ?? []).map((entry) => (
            <option key={entry.key} value={entry.key}>
              {entry.label}
            </option>
          ))}
        </select>
        <input
          aria-label="Model for this step"
          value={model}
          disabled={!provider}
          list="step-model-options"
          placeholder={provider ? chosen?.default_model || 'model id' : ''}
          onChange={(event) => patch({ model_override: event.target.value })}
        />
        <datalist id="step-model-options">
          {(chosen?.models ?? []).map((card) => (
            <option key={card.id} value={card.id}>
              {card.label}
            </option>
          ))}
        </datalist>
        {provider ? (
          <select
            aria-label="Thinking style for this step"
            value={effort}
            onChange={(event) => patch({ thinking_effort: event.target.value })}
          >
            {STEP_EFFORTS.map((level) => (
              <option key={level || 'inherit'} value={level}>
                {level || 'Inherit'}
              </option>
            ))}
          </select>
        ) : null}
      </div>
      <div className="hint">
        {!provider
          ? `Left alone, this step runs on ${inherited}. Set it to run just this step somewhere else, without changing the agent everywhere it is used.`
          : reachable
            ? `This step runs on ${chosen?.label}, whatever the agent is set to. Other steps are unaffected.`
            : `No key saved for ${chosen?.label} — add one in Models, or this step will fail when it runs.`}
      </div>
    </div>
  );
}

function ReviewControls({
  node,
  agents,
  onPatchNode,
}: {
  node: CanvasNode;
  agents: AgentDefinition[];
  onPatchNode: Props['onPatchNode'];
}) {
  const review = node.data.review;
  // An agent reviewing its own draft has already decided it is good — that is
  // why it stopped — so it is not offered.
  const candidates = agents.filter((agent) => agent.id !== node.data.agentId);

  return (
    <div className="field">
      <label htmlFor="reviewer">Reviewed by</label>
      <div className="row">
        <select
          id="reviewer"
          value={review?.agent_id ?? ''}
          onChange={(event) =>
            onPatchNode(node.id, {
              review: event.target.value
                ? {
                    agent_id: event.target.value,
                    max_rounds: review?.max_rounds ?? 2,
                    approval_phrase: review?.approval_phrase ?? 'APPROVED',
                  }
                : null,
            })
          }
        >
          <option value="">— nobody, the answer stands —</option>
          {candidates.map((agent) => (
            <option key={agent.id} value={agent.id}>
              {agent.name}
            </option>
          ))}
        </select>
        {review ? (
          <select
            aria-label="Rounds"
            value={String(review.max_rounds)}
            onChange={(event) =>
              onPatchNode(node.id, {
                review: { ...review, max_rounds: Number(event.target.value) },
              })
            }
          >
            {[1, 2, 3, 4, 5].map((rounds) => (
              <option key={rounds} value={rounds}>
                {rounds} round{rounds > 1 ? 's' : ''}
              </option>
            ))}
          </select>
        ) : null}
      </div>
      <div className="hint">
        {review
          ? 'The reviewer critiques the draft and this agent revises, repeating until the reviewer approves or the rounds run out. The last revision is used either way.'
          : 'Optional. A second agent checking the work catches what the author cannot — it stopped because it already thought the answer was good.'}
      </div>
    </div>
  );
}
