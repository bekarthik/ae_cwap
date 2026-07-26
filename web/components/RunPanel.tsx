'use client';

import { useEffect, useRef } from 'react';

import type { LiveOutput, LogEvent, RunReport, RunStatus } from '@/lib/types';

interface Props {
  status: RunStatus | null;
  logs: LogEvent[];
  /** The answer being written right now, if a step is mid-flight. */
  live: LiveOutput | null;
  report: RunReport | null;
  error: string | null;
}

/**
 * Epic 4's outcome in the UI: a trace the user can read to understand *why* the
 * workflow produced what it did — every step, its output, and the errors.
 */
export function RunPanel({ status, logs, live, report, error }: Props) {
  const logRef = useRef<HTMLDivElement>(null);
  // Whether the reader is watching the tail. Following unconditionally means
  // that scrolling up to re-read an earlier step yanks you back to the bottom
  // on the model's next token — which, against a model writing steadily, makes
  // the history unreadable for as long as the run lasts.
  const following = useRef(true);

  const onScroll = () => {
    const node = logRef.current;
    if (!node) return;
    const slack = node.scrollHeight - node.scrollTop - node.clientHeight;
    following.current = slack < 24;
  };

  useEffect(() => {
    // Follow the tail as events stream in — including the answer being typed,
    // which is the thing worth watching while a slow model works.
    if (following.current) {
      logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
    }
  }, [logs.length, live?.text]);

  if (!status && !error) {
    return (
      <div className="panel-section">
        <p className="panel-title">Run</p>
        <p className="muted small">
          Press <strong>Run</strong> to execute the workflow. Every step, its inputs and
          its output will stream here in real time.
        </p>
      </div>
    );
  }

  const finalResult = report?.result
    ? (report.result.result ?? report.result)
    : null;
  const resultText =
    typeof finalResult === 'string' ? finalResult : JSON.stringify(finalResult, null, 2);

  return (
    <>
      <div className="panel-section" style={{ borderBottom: 'none', paddingBottom: 8 }}>
        <div className="row">
          <p className="panel-title" style={{ margin: 0 }}>
            Run
          </p>
          <div className="spacer" />
          {status ? (
            <span className="status-pill" data-status={status}>
              {status.toLowerCase()}
            </span>
          ) : null}
        </div>
      </div>

      {error ? (
        <div style={{ padding: '0 14px 12px' }}>
          <div className="notice notice--error">{error}</div>
        </div>
      ) : null}

      <div className="log" ref={logRef} onScroll={onScroll}>
        {logs.map((event) => (
          <div
            className={`log-line level-${event.level}`}
            data-event={event.event}
            key={`${event.seq}-${event.event}`}
          >
            <span className="log-line__seq">{event.seq}</span>
            <span className="log-line__event">{event.event}</span>
            <span className="log-line__msg">
              {event.node ? <code>[{event.node}] </code> : null}
              {event.message}
            </span>
          </div>
        ))}
        {live ? (
          <div className={`log-live log-live--${live.kind}`}>
            <span className="log-live__label">
              {live.kind === 'reasoning' ? 'thinking' : 'writing'}
            </span>
            <span className="log-live__text">
              {live.text}
              <span className="log-live__caret" />
            </span>
          </div>
        ) : null}
        {logs.length === 0 && !live ? (
          <p className="muted small">Waiting for events…</p>
        ) : null}
      </div>

      {report ? (
        <div className="run-report panel-section">
          <p className="panel-title">Steps</p>
          <div className="stack" style={{ marginBottom: 12 }}>
            {report.steps.map((step) => (
              <details key={`${step.node_id}-${step.completed_at}`} className="list-item">
                <summary style={{ cursor: 'pointer', flex: 1 }}>
                  <strong>{step.node_id}</strong>{' '}
                  <span className="muted">
                    {step.resulting_state.toLowerCase().replace(/_/g, ' ')} ·{' '}
                    {step.duration_ms}ms
                  </span>
                </summary>
                <pre
                  style={{
                    marginTop: 8,
                    maxHeight: 180,
                    overflow: 'auto',
                    fontSize: 11.5,
                    whiteSpace: 'pre-wrap',
                  }}
                >
                  {JSON.stringify(step.output, null, 2)}
                </pre>
              </details>
            ))}
          </div>

          {report.run.error ? (
            <div className="notice notice--error">{report.run.error}</div>
          ) : null}

          {finalResult ? (
            <>
              <div className="row">
                <p className="panel-title" style={{ margin: 0 }}>
                  Result
                </p>
                <div className="spacer" />
                {/* The answer is the thing someone came for, and it usually
                    belongs somewhere else — a document, a ticket, a reply.
                    Selecting it out of a scrolling <pre> is a fiddly way to
                    end an otherwise finished job. */}
                <button
                  className="btn--link small"
                  onClick={() => void navigator.clipboard?.writeText(resultText)}
                >
                  Copy
                </button>
              </div>
              <pre className="run-result">{resultText}</pre>
            </>
          ) : null}
        </div>
      ) : null}
    </>
  );
}
