'use client';

import { useEffect, useState } from 'react';

import { ApiError, api } from '@/lib/api';
import type { TemplateSummary } from '@/lib/types';

interface Props {
  onDescribeGoal: () => void;
  onBlankCanvas: () => void;
  onUseTemplate: (key: string) => Promise<void> | void;
  onDismiss: () => void;
}

/**
 * The way in.
 *
 * Three routes, because people arrive with different amounts of certainty:
 *
 * 1. **Describe what you need.** The most powerful — the system asks what it
 *    cannot infer and decides the steps. Best when you know the outcome you want
 *    and not the shape of the work.
 * 2. **Start from a template.** Best when you do not yet know what this thing
 *    can do. A worked example beats a feature list, and every template runs as
 *    given before you change anything.
 * 3. **Build it yourself.** Best when you already know the steps and just want
 *    a canvas.
 *
 * Templates are listed with what they need before they would work, because a
 * template that quietly produces a workflow failing on its third step is worse
 * than one that says "upload something first".
 */
export function Launcher({
  onDescribeGoal,
  onBlankCanvas,
  onUseTemplate,
  onDismiss,
}: Props) {
  const [templates, setTemplates] = useState<TemplateSummary[] | null>(null);
  const [openKey, setOpenKey] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .listTemplates()
      .then(setTemplates)
      .catch((caught) =>
        setError(caught instanceof ApiError ? caught.message : 'Could not load templates.'),
      );
  }, []);

  async function use(key: string) {
    setBusy(key);
    setError(null);
    try {
      await onUseTemplate(key);
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : 'Could not use that template.');
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="modal-backdrop" onClick={onDismiss} role="presentation">
      <div
        className="modal modal--wide"
        onClick={(event) => event.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="Start a workflow"
      >
        <h2>What would you like to build?</h2>
        <p className="lede">
          Multi-step AI work, done by agents that improve each time they run.
        </p>

        {error ? <div className="notice notice--error">{error}</div> : null}

        <div className="launcher">
          <button className="launcher__card launcher__card--primary" onClick={onDescribeGoal}>
            <span className="launcher__icon">✎</span>
            <strong>Describe what you need</strong>
            <span className="launcher__blurb">
              Say it in plain language. We work out what it takes, decide the steps,
              and staff each one with an agent — you review the plan before anything
              runs.
            </span>
            <span className="launcher__hint">Best if you know the outcome, not the steps</span>
          </button>

          <button className="launcher__card" onClick={onBlankCanvas}>
            <span className="launcher__icon">▦</span>
            <strong>Build it yourself</strong>
            <span className="launcher__blurb">
              Start with an empty canvas. Drag steps on, draw the connections, and
              map what flows along each one.
            </span>
            <span className="launcher__hint">Best if you already know the steps</span>
          </button>
        </div>

        <p className="panel-title" style={{ marginTop: 18 }}>
          Or start from something that already works
        </p>

        {templates === null ? (
          <p className="muted small">Loading…</p>
        ) : (
          <div className="stack">
            {templates.map((template) => (
              <div className="template" key={template.key}>
                <button
                  className="template__head"
                  onClick={() => setOpenKey(openKey === template.key ? null : template.key)}
                >
                  <span style={{ flex: 1, textAlign: 'left' }}>
                    <strong>{template.title}</strong>
                    <span className="chip" style={{ marginLeft: 6 }}>
                      {template.category}
                    </span>
                    <div className="muted small">{template.summary}</div>
                  </span>
                  <span className="muted small">{openKey === template.key ? '−' : '+'}</span>
                </button>

                {openKey === template.key ? (
                  <div className="template__body">
                    <p className="small">{template.detail}</p>

                    <p className="panel-title small">
                      {template.agents.length} agent
                      {template.agents.length === 1 ? '' : 's'}
                    </p>
                    <div className="stack">
                      {template.agents.map((agent, index) => (
                        <div className="agent-card" key={`${agent.name}-${index}`}>
                          <div className="agent-card__head">
                            <span className="agent-card__step">{index + 1}</span>
                            <strong>{agent.name}</strong>
                          </div>
                          <p className="agent-card__why">{agent.rationale}</p>
                          <div className="row" style={{ flexWrap: 'wrap', gap: 4 }}>
                            {agent.skills.map((skill) => (
                              <span className="chip" key={skill}>
                                {skill.replace(/_/g, ' ')}
                              </span>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>

                    {template.example_input ? (
                      <p className="small muted">
                        Example input: “{template.example_input}”
                      </p>
                    ) : null}

                    {template.requires.length ? (
                      <div className="notice notice--warn">
                        <strong>Set up first:</strong>
                        <ul style={{ margin: '6px 0 0', paddingLeft: 18 }}>
                          {template.requires.map((need) => (
                            <li key={need}>{need}</li>
                          ))}
                        </ul>
                        <div className="small" style={{ marginTop: 6 }}>
                          You can still copy it now and set this up afterwards.
                        </div>
                      </div>
                    ) : null}

                    <button
                      className="btn btn--primary btn--block"
                      onClick={() => void use(template.key)}
                      disabled={busy !== null}
                    >
                      {busy === template.key ? 'Setting it up…' : 'Use this template'}
                    </button>
                  </div>
                ) : null}
              </div>
            ))}
          </div>
        )}

        <button className="btn btn--ghost btn--block" onClick={onDismiss}>
          Close
        </button>
      </div>
    </div>
  );
}
