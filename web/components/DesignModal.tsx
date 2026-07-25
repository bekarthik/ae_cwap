'use client';

import { useState } from 'react';

import { ApiError, api } from '@/lib/api';
import type { DesignResponse } from '@/lib/types';

interface Props {
  onClose: () => void;
  onAccept: (design: DesignResponse) => void;
}

const EXAMPLES = [
  'Plan my weekend trip to Denver',
  'Summarise our internal handbook and draft an onboarding email',
  'Research our competitors and write a short comparison',
];

/**
 * Epic 1 in the UI, as a conversation rather than a form.
 *
 * The user says what they want. The system says back what it understood and
 * asks only what it could not work out — each question carrying *why* it is
 * being asked, so it reads as help rather than an interrogation. Then it shows
 * the design: which agents it decided on, why each one is there, which skills
 * they were given, and which of those it had to build. Only then does anything
 * land on the canvas.
 *
 * Every question has a default, so the whole thing can be skipped by someone who
 * just wants to see something run.
 */
export function DesignModal({ onClose, onAccept }: Props) {
  const [goal, setGoal] = useState('');
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<DesignResponse | null>(null);

  async function send(nextAnswers: Record<string, string>, direct = false) {
    setBusy(true);
    setError(null);
    try {
      setResult(
        direct ? await api.designDirect(goal) : await api.design(goal, nextAnswers),
      );
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : 'Design failed.');
    } finally {
      setBusy(false);
    }
  }

  function start(event: React.FormEvent) {
    event.preventDefault();
    setAnswers({});
    void send({});
  }

  const clarifying = result?.stage === 'clarifying';
  const designed = result?.stage === 'designed';

  return (
    <div className="modal-backdrop" onClick={onClose} role="presentation">
      <div
        className="modal modal--wide"
        onClick={(event) => event.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="Describe your goal"
      >
        <h2>What do you want to get done?</h2>
        <p className="lede">
          Describe it in plain language. We&apos;ll work out what it takes, decide the
          steps, and staff each one with an agent — you review the plan.
        </p>

        {error ? <div className="notice notice--error">{error}</div> : null}

        <form onSubmit={start}>
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
              {busy ? 'Thinking…' : result ? 'Start over' : 'Design it'}
            </button>
            <button
              type="button"
              className="btn"
              disabled={busy || goal.trim().length < 3}
              onClick={() => void send({}, true)}
              title="Use sensible defaults for everything"
            >
              Skip the questions
            </button>
            <button type="button" className="btn" onClick={onClose}>
              Cancel
            </button>
          </div>
        </form>

        {result ? (
          <div className="design">
            <p className="design__understanding">{result.understanding}</p>

            {clarifying ? (
              <Questions
                design={result}
                answers={answers}
                busy={busy}
                onChange={setAnswers}
                onSubmit={() =>
                  // Send every question that was displayed, including the ones
                  // left blank. "I asked, they chose not to say" is a different
                  // fact from "I have not asked yet" — without the distinction a
                  // question with no default is re-asked forever, because
                  // clicking through never puts its id in the answer set.
                  void send(
                    Object.fromEntries(
                      result.questions.map((question) => [
                        question.id,
                        answers[question.id] ?? '',
                      ]),
                    ),
                  )
                }
              />
            ) : null}

            {designed ? <Design design={result} onAccept={onAccept} /> : null}
          </div>
        ) : null}
      </div>
    </div>
  );
}

function Questions({
  design,
  answers,
  busy,
  onChange,
  onSubmit,
}: {
  design: DesignResponse;
  answers: Record<string, string>;
  busy: boolean;
  onChange: (answers: Record<string, string>) => void;
  onSubmit: () => void;
}) {
  function set(id: string, value: string) {
    onChange({ ...answers, [id]: value });
  }

  return (
    <>
      <p className="panel-title">A few things first</p>
      <div className="stack">
        {design.questions.map((question) => (
          <div className="question" key={question.id}>
            <label className="question__label" htmlFor={`q-${question.id}`}>
              {question.question}
            </label>
            <p className="question__why">{question.why}</p>

            {question.options.length ? (
              <div className="row" style={{ flexWrap: 'wrap' }}>
                {question.options.map((option) => (
                  <button
                    key={option}
                    type="button"
                    className={`btn small ${
                      (answers[question.id] ?? question.default) === option
                        ? 'btn--primary'
                        : 'btn--ghost'
                    }`}
                    onClick={() => set(question.id, option)}
                  >
                    {option}
                  </button>
                ))}
              </div>
            ) : (
              <input
                id={`q-${question.id}`}
                type="text"
                value={answers[question.id] ?? ''}
                placeholder={question.default || 'Optional'}
                onChange={(event) => set(question.id, event.target.value)}
              />
            )}
          </div>
        ))}
      </div>

      {design.notes.map((note) => (
        <p className="small muted" key={note}>
          {note}
        </p>
      ))}

      <button className="btn btn--primary btn--block" disabled={busy} onClick={onSubmit}>
        {busy ? 'Designing…' : 'Design the workflow'}
      </button>
    </>
  );
}

function Design({
  design,
  onAccept,
}: {
  design: DesignResponse;
  onAccept: (design: DesignResponse) => void;
}) {
  const blocked = design.skill_gaps.filter((gap) => gap.blocked_reason);
  const built = design.skill_gaps.filter((gap) => !gap.blocked_reason);

  return (
    <>
      <p className="panel-title">
        {design.agents.length} agent{design.agents.length === 1 ? '' : 's'}, in order
      </p>

      <div className="stack">
        {design.agents.map((agent, index) => (
          <div className="agent-card" key={`${agent.name}-${index}`}>
            <div className="agent-card__head">
              <span className="agent-card__step">{index + 1}</span>
              <strong>{agent.name}</strong>
            </div>
            <p className="agent-card__role">{agent.role}</p>
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

      {built.length ? (
        <div className="notice notice--info">
          <strong>Built for you:</strong>
          <ul style={{ margin: '6px 0 0', paddingLeft: 18 }}>
            {built.map((gap) => (
              <li key={gap.capability}>
                {gap.capability} — for {gap.needed_by}
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {blocked.length ? (
        <div className="notice notice--warn">
          <strong>Needs your input:</strong>
          <ul style={{ margin: '6px 0 0', paddingLeft: 18 }}>
            {blocked.map((gap) => (
              <li key={gap.capability}>
                <strong>{gap.capability}</strong> — {gap.blocked_reason}
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {design.notes.length ? (
        <ul className="small muted" style={{ paddingLeft: 18 }}>
          {design.notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      ) : null}

      <button className="btn btn--primary btn--block" onClick={() => onAccept(design)}>
        Put this on the canvas
      </button>
    </>
  );
}
