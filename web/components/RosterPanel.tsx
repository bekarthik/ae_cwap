'use client';

import { useState } from 'react';

import { ApiError, api } from '@/lib/api';
import type { AgentDefinition, SkillDefinition } from '@/lib/types';

import { AgentEditor } from './AgentEditor';
import { MemoryPanel } from './MemoryPanel';

interface Props {
  agents: AgentDefinition[];
  skills: SkillDefinition[];
  onChanged: () => Promise<void> | void;
}

/**
 * The agents and skills this workspace has, outside any one workflow.
 *
 * They are listed here rather than only inside the canvas because they outlive
 * the workflow that created them: an agent improves across every run, and a
 * skill's lessons are inherited by every agent that holds it. Something that
 * accumulates is something the user should be able to look at.
 */
export function RosterPanel({ agents, skills, onChanged }: Props) {
  const [tab, setTab] = useState<'agents' | 'skills'>('agents');
  const [openId, setOpenId] = useState<string | null>(null);
  const [capability, setCapability] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  // `undefined` means the editor is closed; `null` means "new agent".
  const [editing, setEditing] = useState<AgentDefinition | null | undefined>(undefined);

  async function build(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const { skill, created } = await api.ensureSkill(capability);
      setNotice(
        created
          ? `Built “${skill.name}”. It is available to every agent from now on.`
          : `“${skill.name}” already covers that.`,
      );
      setCapability('');
      await onChanged();
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : 'Could not build that.');
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="panel-section">
      <p className="panel-title">Your workforce</p>

      <div className="tabs">
        <button
          className={`tab${tab === 'agents' ? ' is-active' : ''}`}
          onClick={() => setTab('agents')}
        >
          Agents ({agents.length})
        </button>
        <button
          className={`tab${tab === 'skills' ? ' is-active' : ''}`}
          onClick={() => setTab('skills')}
        >
          Skills ({skills.length})
        </button>
      </div>

      {error ? <div className="notice notice--error">{error}</div> : null}
      {notice ? <div className="notice notice--info">{notice}</div> : null}

      {tab === 'agents' ? (
        agents.length === 0 ? (
          <p className="muted small">
            None yet. Describe a goal and the system will design and staff one — or
            write one yourself.
          </p>
        ) : (
          agents.map((agent) => (
            <div className="roster-item" key={agent.id}>
              <button
                className="roster-item__head"
                onClick={() => setOpenId(openId === agent.id ? null : agent.id)}
              >
                <span style={{ flex: 1, textAlign: 'left' }}>
                  <strong>{agent.name}</strong>
                  <div className="muted small">
                    {agent.skill_ids.length} skill(s) · up to {agent.max_iterations} steps
                  </div>
                </span>
                <span className="muted small">{openId === agent.id ? '−' : '+'}</span>
              </button>

              {openId === agent.id ? (
                <div className="roster-item__body">
                  <p className="small">{agent.role}</p>
                  <div className="row" style={{ flexWrap: 'wrap', gap: 4 }}>
                    {agent.skill_ids.map((id) => (
                      <span className="chip" key={id}>
                        {skills.find((skill) => skill.id === id)?.name ?? id}
                      </span>
                    ))}
                  </div>
                  <div className="row" style={{ marginTop: 8 }}>
                    <button className="btn small" onClick={() => setEditing(agent)}>
                      Edit
                    </button>
                    <span className="muted small">
                      {agent.model_provider
                        ? `${agent.model_provider}${
                            agent.model_override ? ` · ${agent.model_override}` : ''
                          }`
                        : 'workspace model'}
                      {agent.thinking_effort ? ` · thinks ${agent.thinking_effort}` : ''}
                    </span>
                  </div>
                  <MemoryPanel
                    scope="agent"
                    scopeId={agent.id}
                    title="What this agent has learned"
                  />
                </div>
              ) : null}
            </div>
          ))
        )
      ) : null}

      {tab === 'agents' ? (
        <button
          className="btn btn--block"
          style={{ marginTop: 6 }}
          onClick={() => setEditing(null)}
        >
          Write an agent yourself
        </button>
      ) : null}

      {tab === 'skills' ? (
        <>
          {skills.map((skill) => (
            <div className="roster-item" key={skill.id}>
              <button
                className="roster-item__head"
                onClick={() => setOpenId(openId === skill.id ? null : skill.id)}
              >
                <span style={{ flex: 1, textAlign: 'left' }}>
                  <strong>{skill.name.replace(/_/g, ' ')}</strong>
                  <div className="muted small">
                    {skill.kind} · {skill.origin}
                    {skill.invocations
                      ? ` · used ${skill.invocations}×${
                          skill.failures ? `, failed ${skill.failures}×` : ''
                        }`
                      : ''}
                  </div>
                </span>
                <span className="muted small">{openId === skill.id ? '−' : '+'}</span>
              </button>

              {openId === skill.id ? (
                <div className="roster-item__body">
                  <p className="small">{skill.description}</p>
                  <MemoryPanel
                    scope="skill"
                    scopeId={skill.id}
                    title="What this skill has learned"
                  />
                </div>
              ) : null}
            </div>
          ))}

          <form onSubmit={build} style={{ marginTop: 10 }}>
            <div className="field">
              <label htmlFor="capability">Need something else?</label>
              <input
                id="capability"
                value={capability}
                placeholder="e.g. reconcile expense claims against receipts"
                onChange={(event) => setCapability(event.target.value)}
              />
              <p className="hint">
                We build it as a prompt, a document search, or a text transform —
                never code, and never an outbound call.
              </p>
            </div>
            <button
              className="btn btn--block"
              type="submit"
              disabled={busy || capability.trim().length < 3}
            >
              {busy ? 'Building…' : 'Build this skill'}
            </button>
          </form>
        </>
      ) : null}

      {editing !== undefined ? (
        <AgentEditor
          agent={editing}
          skills={skills}
          onClose={() => setEditing(undefined)}
          onSaved={async () => {
            setNotice(editing ? 'Saved.' : 'Created.');
            await onChanged();
          }}
        />
      ) : null}
    </div>
  );
}
