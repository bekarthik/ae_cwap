'use client';

import { useEffect, useState } from 'react';

import { ApiError, api } from '@/lib/api';
import type {
  AgentDefinition,
  ModelConfiguration,
  ProviderOption,
  SkillDefinition,
} from '@/lib/types';

interface Props {
  /** The agent being edited, or null to create one. */
  agent: AgentDefinition | null;
  skills: SkillDefinition[];
  onClose: () => void;
  onSaved: (agent: AgentDefinition) => Promise<void> | void;
}

const EFFORTS = ['', 'low', 'medium', 'high', 'xhigh', 'max'] as const;

/**
 * Editing what an agent *is*.
 *
 * The platform designs agents, and the designs were unimprovable: an agent's
 * instructions, its model and how hard it may think were all decided by the
 * planner and then read-only forever. A workspace whose agents accumulate
 * memory across runs but cannot be corrected is a workspace that gets more
 * confidently wrong, so this is the screen that closes the loop.
 *
 * The model is two fields — provider and model id — because one cannot be
 * resolved without the other: `claude-opus-5` and `qwen3` are not served at the
 * same endpoint, and guessing the provider from the name is the kind of
 * inference this product refuses to make. Leaving both blank is the normal
 * case: the agent runs on whatever the workspace runs on.
 */
export function AgentEditor({ agent, skills, onClose, onSaved }: Props) {
  const [name, setName] = useState(agent?.name ?? '');
  const [role, setRole] = useState(agent?.role ?? '');
  const [instructions, setInstructions] = useState(agent?.instructions ?? '');
  const [skillIds, setSkillIds] = useState<string[]>(agent?.skill_ids ?? []);
  const [maxIterations, setMaxIterations] = useState(String(agent?.max_iterations ?? 6));
  const [provider, setProvider] = useState(agent?.model_provider ?? '');
  const [model, setModel] = useState(agent?.model_override ?? '');
  const [effort, setEffort] = useState(agent?.thinking_effort ?? '');

  const [config, setConfig] = useState<ModelConfiguration | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void api
      .modelConfig()
      .then(setConfig)
      .catch(() => setConfig(null));
  }, []);

  const chosen: ProviderOption | null =
    config?.providers.find((entry) => entry.key === provider) ?? null;
  // A provider with no stored credential will fail at run time. Saying so here
  // is the difference between a filled-in field and a broken workflow.
  const reachable =
    !provider || !chosen?.requires_key || (config?.credentials ?? []).includes(provider);
  // Effort only reaches a model with a reasoning mode; offering it against one
  // without would be a control that silently does nothing.
  const thinks = (chosen?.models ?? []).some(
    (card) => card.id === model && card.supports.thinking,
  );

  async function save() {
    setBusy(true);
    setError(null);
    try {
      const body = {
        name: name.trim(),
        role: role.trim(),
        objective: agent?.objective ?? '',
        instructions: instructions.trim(),
        skill_ids: skillIds,
        max_iterations: Number.parseInt(maxIterations, 10) || 6,
        model_provider: provider,
        model_override: model.trim() || null,
        thinking_effort: effort,
        memory: agent?.memory ?? {
          recall: true,
          recall_limit: 5,
          write_learnings: true,
          use_workflow_memory: true,
        },
      };
      const saved = agent
        ? await api.updateAgent(agent.id, body)
        : await api.createAgent(body);
      await onSaved(saved);
      onClose();
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : 'Could not save that.');
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="modal-backdrop" onClick={onClose} role="presentation">
      <div
        className="modal"
        onClick={(event) => event.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label={agent ? `Edit ${agent.name}` : 'New agent'}
      >
        <h2>{agent ? agent.name : 'New agent'}</h2>
        <p className="lede">
          An agent is reusable: every workflow that uses it gets these changes, and
          what it learns stays with it.
        </p>

        {error ? <div className="notice notice--error">{error}</div> : null}

        <div className="field">
          <label htmlFor="agent-name">Name</label>
          <input
            id="agent-name"
            value={name}
            placeholder="Researcher"
            onChange={(event) => setName(event.target.value)}
          />
        </div>

        <div className="field">
          <label htmlFor="agent-role">What it is</label>
          <textarea
            id="agent-role"
            value={role}
            placeholder="A careful researcher who checks claims before repeating them."
            onChange={(event) => setRole(event.target.value)}
          />
          <div className="hint">One or two sentences, in plain language.</div>
        </div>

        <div className="field">
          <label htmlFor="agent-instructions">Standing instructions</label>
          <textarea
            id="agent-instructions"
            value={instructions}
            placeholder="Optional. Anything it must always do, in every workflow."
            onChange={(event) => setInstructions(event.target.value)}
          />
        </div>

        <div className="field">
          <label htmlFor="agent-provider">Model</label>
          <div className="row">
            <select
              id="agent-provider"
              value={provider}
              onChange={(event) => {
                setProvider(event.target.value);
                setModel('');
              }}
            >
              <option value="">The workspace&apos;s model</option>
              {(config?.providers ?? []).map((entry) => (
                <option key={entry.key} value={entry.key}>
                  {entry.label}
                </option>
              ))}
            </select>
            <input
              value={model}
              disabled={!provider}
              list="agent-model-options"
              placeholder={provider ? chosen?.default_model || 'model id' : ''}
              onChange={(event) => setModel(event.target.value)}
            />
            <datalist id="agent-model-options">
              {(chosen?.models ?? []).map((card) => (
                <option key={card.id} value={card.id}>
                  {card.label}
                </option>
              ))}
            </datalist>
          </div>
          <div className="hint">
            {!provider
              ? 'Leave it blank and this agent runs on whatever the workspace runs on. Set it to mix backends in one workflow — a small local model for triage, a stronger one for synthesis.'
              : reachable
                ? `Runs on ${chosen?.label}. Its credential comes from what this workspace saved for that provider.`
                : `No key saved for ${chosen?.label} — add one in Models, or this agent will fail when it runs.`}
          </div>
        </div>

        <div className="row">
          <div className="field" style={{ flex: 1 }}>
            <label htmlFor="agent-effort">Thinking style</label>
            <select
              id="agent-effort"
              value={effort}
              onChange={(event) => setEffort(event.target.value)}
            >
              {EFFORTS.map((level) => (
                <option key={level || 'inherit'} value={level}>
                  {level || 'Inherit'}
                </option>
              ))}
            </select>
            <div className="hint">
              {thinks
                ? 'How hard this model may think before answering.'
                : 'Honoured by models with a reasoning mode; reported as ignored elsewhere.'}
            </div>
          </div>

          <div className="field" style={{ flex: 1 }}>
            <label htmlFor="agent-budget">Step budget</label>
            <input
              id="agent-budget"
              inputMode="numeric"
              value={maxIterations}
              onChange={(event) =>
                setMaxIterations(event.target.value.replace(/[^0-9]/g, ''))
              }
            />
            <div className="hint">
              How many times it may use a skill and reconsider before it must answer.
            </div>
          </div>
        </div>

        <div className="field">
          <label>Skills</label>
          <div className="row" style={{ flexWrap: 'wrap', gap: 4 }}>
            {skills.length === 0 ? (
              <span className="muted small">None in this workspace yet.</span>
            ) : (
              skills.map((skill) => (
                <button
                  key={skill.id}
                  className={`chip chip--toggle${
                    skillIds.includes(skill.id) ? ' is-on' : ''
                  }`}
                  onClick={() =>
                    setSkillIds((current) =>
                      current.includes(skill.id)
                        ? current.filter((id) => id !== skill.id)
                        : [...current, skill.id],
                    )
                  }
                >
                  {skill.name.replace(/_/g, ' ')}
                </button>
              ))
            )}
          </div>
          <div className="hint">What it can reach for. Its tools, in its own words.</div>
        </div>

        <div className="row">
          <button
            className="btn btn--primary"
            onClick={save}
            disabled={busy || !name.trim() || !role.trim()}
          >
            {busy ? 'Saving…' : agent ? 'Save changes' : 'Create agent'}
          </button>
          <button className="btn btn--ghost" onClick={onClose}>
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
