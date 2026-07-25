'use client';

import { useEffect, useRef } from 'react';

import type { LogEvent, RunReport, RunStatus } from '@/lib/types';

interface Props {
  status: RunStatus | null;
  logs: LogEvent[];
  report: RunReport | null;
  error: string | null;
}

/**
 * Epic 4's outcome in the UI: a trace the user can read to understand *why* the
 * workflow produced what it did — every step, its output, and the errors.
 */
export function RunPanel({ status, logs, report, error }: Props) {
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Follow the tail as events stream in.
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [logs.length]);

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

      <div className="log" ref={logRef}>
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
        {logs.length === 0 ? <p className="muted small">Waiting for events…</p> : null}
      </div>

      {report ? (
        <div className="panel-section" style={{ borderTop: '1px solid var(--border)' }}>
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
              <p className="panel-title">Result</p>
              <pre
                style={{
                  whiteSpace: 'pre-wrap',
                  fontSize: 12.5,
                  background: 'var(--surface-2)',
                  padding: 10,
                  borderRadius: 8,
                  margin: 0,
                  maxHeight: 260,
                  overflow: 'auto',
                }}
              >
                {typeof finalResult === 'string'
                  ? finalResult
                  : JSON.stringify(finalResult, null, 2)}
              </pre>
            </>
          ) : null}
        </div>
      ) : null}
    </>
  );
}
