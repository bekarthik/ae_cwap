'use client';

import { useState } from 'react';

import { ApiError, api } from '@/lib/api';
import type { ScaffoldResponse } from '@/lib/types';

interface Props {
  onClose: () => void;
  onAccept: (scaffold: ScaffoldResponse) => void;
}

const EXAMPLES = [
  'Plan my weekend trip to Denver',
  'Summarise our internal handbook and draft an onboarding email',
  'Research our competitors and write a short comparison',
];

/**
 * Epic 1 in the UI: type a goal, see what the system decided you need, then
 * accept the draft onto the canvas. The diagnosis is shown *before* the graph
 * is applied so the user reviews the reasoning rather than a mystery layout.
 */
export function GoalModal({ onClose, onAccept }: Props) {
  const [goal, setGoal] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ScaffoldResponse | null>(null);

  async function diagnose(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      setResult(await api.diagnose(goal));
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : 'Diagnosis failed.');
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
        aria-label="Describe your goal"
      >
        <h2>What do you want to get done?</h2>
        <p className="lede">
          Describe it in plain language. We&apos;ll work out which steps it needs and lay
          out a starting workflow for you to review.
        </p>

        {error ? <div className="notice notice--error">{error}</div> : null}

        <form onSubmit={diagnose}>
          <div className="field">
            <textarea
              rows={3}
              value={goal}
              placeholder={EXAMPLES[0]}
              onChange={(event) => setGoal(event.target.value)}
              style={{ fontFamily: 'inherit', fontSize: 14 }}
            />
          </div>

          <div className="row" style={{ flexWrap: 'wrap', marginBottom: 12 }}>
            {EXAMPLES.map((example) => (
              <button
                key={example}
                type="button"
                className="btn btn--ghost small"
                onClick={() => setGoal(example)}
              >
                “{example}”
              </button>
            ))}
          </div>

          <div className="row">
            <button
              className="btn btn--primary"
              type="submit"
              disabled={busy || goal.trim().length < 3}
            >
              {busy ? 'Thinking…' : 'Diagnose'}
            </button>
            <button type="button" className="btn" onClick={onClose}>
              Cancel
            </button>
          </div>
        </form>

        {result ? (
          <div style={{ marginTop: 20, borderTop: '1px solid var(--border)', paddingTop: 16 }}>
            <p className="panel-title">Proposed workflow</p>
            <p className="small muted" style={{ marginTop: 0 }}>
              Confidence {Math.round(result.diagnosis.confidence * 100)}% · intent{' '}
              <code>{result.diagnosis.intent}</code>
            </p>

            <div className="stack" style={{ marginBottom: 12 }}>
              {result.diagnosis.required_steps.map((step, index) => (
                <div className="list-item" key={`${step.skill}-${index}`}>
                  <strong>{index + 1}.</strong>
                  <span>
                    <strong>{step.skill.replace(/_/g, ' ')}</strong>{' '}
                    <span className="muted">({step.node_type})</span>
                    <div className="muted small">{step.rationale}</div>
                  </span>
                </div>
              ))}
            </div>

            {result.diagnosis.gaps.length ? (
              <div className="notice notice--warn">
                <strong>Needs your input:</strong>
                <ul style={{ margin: '6px 0 0', paddingLeft: 18 }}>
                  {result.diagnosis.gaps.map((gap) => (
                    <li key={gap}>{gap}</li>
                  ))}
                </ul>
              </div>
            ) : null}

            {result.diagnosis.suggestions.length ? (
              <div className="notice notice--info">
                <ul style={{ margin: 0, paddingLeft: 18 }}>
                  {result.diagnosis.suggestions.map((suggestion) => (
                    <li key={suggestion}>{suggestion}</li>
                  ))}
                </ul>
              </div>
            ) : null}

            <button
              className="btn btn--primary btn--block"
              onClick={() => onAccept(result)}
            >
              Put this on the canvas
            </button>
          </div>
        ) : null}
      </div>
    </div>
  );
}
