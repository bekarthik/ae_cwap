'use client';

import { useCallback, useEffect, useState } from 'react';

import { ApiError, api } from '@/lib/api';
import type { MemoryEntry, MemoryScope } from '@/lib/types';

interface Props {
  scope: MemoryScope;
  scopeId: string;
  title: string;
}

/**
 * What one scope has learned, and the controls to correct it.
 *
 * Memory a user cannot see is memory they cannot trust — a workflow that
 * quietly behaves differently on its tenth run than its first is unnerving
 * rather than impressive. So every lesson is listed with how much the system
 * currently trusts it, can be deleted if it is wrong, and can be added to
 * directly when the user knows something the agent has not worked out yet.
 */
export function MemoryPanel({ scope, scopeId, title }: Props) {
  const [entries, setEntries] = useState<MemoryEntry[]>([]);
  const [draft, setDraft] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    try {
      setEntries(await api.listMemory(scope, scopeId));
      setError(null);
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : 'Could not read memory.');
    }
  }, [scope, scopeId]);

  useEffect(() => {
    void load();
  }, [load]);

  async function teach(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    try {
      await api.teach(scope, scopeId, draft.trim());
      setDraft('');
      await load();
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : 'Could not save that.');
    } finally {
      setBusy(false);
    }
  }

  async function forget(entryId: string) {
    await api.forgetMemory(entryId);
    await load();
  }

  return (
    <div className="memory">
      <p className="panel-title small">{title}</p>

      {error ? <div className="notice notice--error">{error}</div> : null}

      {entries.length === 0 ? (
        <p className="muted small">
          Nothing yet. It fills in as runs happen — or tell it something now.
        </p>
      ) : (
        <ul className="memory__list">
          {entries.map((entry) => (
            <li className="memory__item" key={entry.id}>
              <span className={`memory__kind memory__kind--${entry.kind}`}>
                {entry.kind}
              </span>
              <span className="memory__text">{entry.text}</span>
              <span
                className="memory__trust"
                title={`Trust ${entry.usefulness.toFixed(1)} — rises when a run goes well, falls when it does not`}
              >
                {entry.usefulness.toFixed(1)}
              </span>
              <button
                className="btn btn--ghost small"
                onClick={() => void forget(entry.id)}
                aria-label={`Forget: ${entry.text}`}
              >
                ✕
              </button>
            </li>
          ))}
        </ul>
      )}

      <form onSubmit={teach} className="row" style={{ marginTop: 6 }}>
        <input
          value={draft}
          placeholder="Tell it something…"
          onChange={(event) => setDraft(event.target.value)}
          style={{ flex: 1 }}
        />
        <button className="btn small" type="submit" disabled={busy || draft.trim().length < 3}>
          Teach
        </button>
      </form>
    </div>
  );
}
