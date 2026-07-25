'use client';

import { useRef, useState } from 'react';

import { ApiError, api } from '@/lib/api';
import type { KnowledgeSummary, RetrievalResult } from '@/lib/types';

interface Props {
  corpora: KnowledgeSummary[];
  onChanged: () => void;
}

/**
 * Epic 3 in the UI. Upload documents, then check what a retrieval step will
 * actually find before wiring it into a workflow — grounding you cannot inspect
 * is grounding you cannot trust.
 */
export function KnowledgePanel({ corpora, onChanged }: Props) {
  const fileInput = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [probe, setProbe] = useState<{ handle: string; query: string } | null>(null);
  const [preview, setPreview] = useState<RetrievalResult | null>(null);

  async function upload(file: File) {
    setBusy(true);
    setError(null);
    try {
      await api.uploadDocument(file);
      onChanged();
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : 'Upload failed.');
    } finally {
      setBusy(false);
      if (fileInput.current) fileInput.current.value = '';
    }
  }

  async function runPreview() {
    if (!probe) return;
    setBusy(true);
    setError(null);
    try {
      setPreview(await api.previewRetrieval(probe.handle, probe.query));
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : 'Preview failed.');
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="panel-section">
      <p className="panel-title">Knowledge</p>

      {error ? <div className="notice notice--error">{error}</div> : null}

      <input
        ref={fileInput}
        type="file"
        accept=".txt,.md,.markdown,.csv,.json,.yaml,.yml,.rst,.log"
        style={{ display: 'none' }}
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) void upload(file);
        }}
      />
      <button
        className="btn btn--block"
        disabled={busy}
        onClick={() => fileInput.current?.click()}
      >
        Upload a document
      </button>
      <div className="hint" style={{ marginBottom: 12 }}>
        Plain text formats. Each becomes a Knowledge Context you can link to a
        retrieval step.
      </div>

      {corpora.length === 0 ? (
        <p className="muted small">Nothing uploaded yet.</p>
      ) : (
        corpora.map((corpus) => (
          <div
            className={`list-item${probe?.handle === corpus.handle ? ' is-active' : ''}`}
            key={corpus.handle}
          >
            <span style={{ flex: 1 }}>
              <strong>{corpus.title}</strong>
              <div className="muted small">{corpus.chunk_count} passages</div>
            </span>
            <button
              className="btn btn--ghost small"
              title="Test what this returns"
              onClick={() => {
                setPreview(null);
                setProbe({ handle: corpus.handle, query: '' });
              }}
            >
              test
            </button>
            <button
              className="btn btn--ghost small"
              title="Delete"
              onClick={async () => {
                await api.deleteKnowledge(corpus.handle);
                setProbe(null);
                setPreview(null);
                onChanged();
              }}
            >
              ✕
            </button>
          </div>
        ))
      )}

      {probe ? (
        <div style={{ marginTop: 10 }}>
          <div className="field">
            <label htmlFor="probe">Try a question</label>
            <input
              id="probe"
              value={probe.query}
              placeholder="e.g. what is the meal allowance?"
              onChange={(event) => setProbe({ ...probe, query: event.target.value })}
            />
          </div>
          <button
            className="btn btn--block"
            disabled={busy || probe.query.trim().length === 0}
            onClick={runPreview}
          >
            Show what would be retrieved
          </button>

          {preview ? (
            <div className="stack" style={{ marginTop: 10 }}>
              {preview.chunks.length === 0 ? (
                <p className="muted small">
                  Nothing matched. Try wording closer to the document.
                </p>
              ) : null}
              {preview.chunks.map((chunk) => (
                <div className="list-item" key={chunk.chunk_id} style={{ display: 'block' }}>
                  <div className="muted small">score {chunk.score.toFixed(3)}</div>
                  <div className="small">{chunk.text.slice(0, 260)}</div>
                </div>
              ))}
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
