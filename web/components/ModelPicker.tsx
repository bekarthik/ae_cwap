'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';

import { ApiError, api } from '@/lib/api';
import type { DetectedModel, ModelConfiguration, ProviderOption } from '@/lib/types';

interface Props {
  onClose: () => void;
  onSaved: () => void;
}

/**
 * Pick a model backend, from the product.
 *
 * The platform being model-agnostic is only useful if choosing a model is
 * something a person can actually do. Previously this listed what was possible
 * and told you which environment variables to set — which is a documentation
 * page wearing a dialog's clothes. Now it does the thing:
 *
 *   pick a provider → detect what that endpoint actually serves → test it →
 *   save, and the next run uses it.
 *
 * Detection matters because the curated catalogue cannot know what *your* server
 * has loaded. Testing matters because saving a wrong configuration otherwise
 * surfaces as a transport error on someone's first workflow, with no way to tell
 * a bad key from a wrong URL from a model the server does not have.
 *
 * The API never returns a stored key, so the field stays blank and empty means
 * "keep what is saved". Clearing it deliberately is a separate action.
 */
export function ModelPicker({ onClose, onSaved }: Props) {
  const [config, setConfig] = useState<ModelConfiguration | null>(null);
  const [provider, setProvider] = useState('');
  const [model, setModel] = useState('');
  const [baseUrl, setBaseUrl] = useState('');
  const [apiKey, setApiKey] = useState('');
  // Blank means "use the deployment default", which is what the placeholder
  // shows. Kept as text so the field can be empty rather than showing a 0.
  const [wait, setWait] = useState('');

  const [detected, setDetected] = useState<DetectedModel[] | null>(null);
  const [detectError, setDetectError] = useState<string | null>(null);
  const [test, setTest] = useState<{ ok: boolean; message: string } | null>(null);
  const [busy, setBusy] = useState<'detect' | 'test' | 'save' | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const loaded = await api.modelConfig();
      setConfig(loaded);

      // The running provider may not be one of the offered presets — `stub` is
      // the default and is deliberately not in the list, since nobody chooses
      // it. Falling back to the first option keeps the right-hand pane
      // populated instead of opening on an empty column.
      const running = loaded.stored?.provider ?? loaded.active.provider;
      const known = loaded.providers.some((entry) => entry.key === running);
      const initial = known ? running : (loaded.providers[0]?.key ?? '');

      setProvider(initial);
      setModel(
        known
          ? (loaded.stored?.model ?? loaded.active.model ?? '')
          : (loaded.providers[0]?.default_model ?? ''),
      );
      setBaseUrl(known ? (loaded.stored?.base_url ?? '') : '');
      const stored = known ? (loaded.stored?.timeout_seconds ?? 0) : 0;
      setWait(stored > 0 ? String(stored) : '');
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : 'Could not load model settings.');
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const selected: ProviderOption | null = useMemo(
    () => config?.providers.find((entry) => entry.key === provider) ?? null,
    [config, provider],
  );

  /** Detected models when we have them, the curated list until then. */
  const offered: DetectedModel[] = useMemo(() => {
    if (detected) return detected;
    return (selected?.models ?? []).map((card) => ({ ...card, known: true }));
  }, [detected, selected]);

  /** The timeout to send: a number the user typed, or 0 for "the default". */
  function seconds(): number {
    const parsed = Number.parseInt(wait, 10);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : 0;
  }

  function choose(key: string) {
    const next = config?.providers.find((entry) => entry.key === key) ?? null;
    setProvider(key);
    setModel(next?.default_model ?? '');
    setBaseUrl('');
    setDetected(null);
    setDetectError(null);
    setTest(null);
  }

  async function detect() {
    setBusy('detect');
    setDetectError(null);
    try {
      const result = await api.detectModels(provider, baseUrl, apiKey);
      if (result.ok) {
        setDetected(result.models);
        if (result.models.length && !result.models.some((m) => m.id === model)) {
          setModel(result.models[0].id);
        }
      } else {
        setDetected(null);
        setDetectError(result.error ?? 'Could not reach that endpoint.');
      }
    } catch (caught) {
      setDetectError(caught instanceof ApiError ? caught.message : 'Detection failed.');
    } finally {
      setBusy(null);
    }
  }

  async function runTest() {
    setBusy('test');
    setTest(null);
    try {
      setTest(await api.testModel(provider, model, baseUrl, apiKey, seconds()));
    } catch (caught) {
      setTest({
        ok: false,
        message: caught instanceof ApiError ? caught.message : 'Test failed.',
      });
    } finally {
      setBusy(null);
    }
  }

  async function save() {
    setBusy('save');
    setError(null);
    try {
      // An untouched key field means "keep the stored one", which is the only
      // sane reading when the API never gave it back.
      await api.saveModel(provider, model, baseUrl, apiKey || null, seconds());
      await load();
      setApiKey('');
      onSaved();
      onClose();
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : 'Could not save.');
    } finally {
      setBusy(null);
    }
  }

  async function revert() {
    setBusy('save');
    try {
      await api.clearModel();
      await load();
      onSaved();
      onClose();
    } finally {
      setBusy(null);
    }
  }

  if (!config) {
    return (
      <div className="modal-backdrop" onClick={onClose} role="presentation">
        <div className="modal" onClick={(event) => event.stopPropagation()}>
          <p className="muted">{error ?? 'Loading…'}</p>
        </div>
      </div>
    );
  }

  const local = config.providers.filter((entry) => entry.local);
  const hosted = config.providers.filter((entry) => !entry.local);

  return (
    <div className="modal-backdrop" onClick={onClose} role="presentation">
      <div
        className="modal modal--wide"
        onClick={(event) => event.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="Choose a model"
      >
        <h2>Models</h2>
        <p className="lede">
          Everything here runs the same workflows. Changes apply to your next run —
          nothing restarts.
        </p>

        {error ? <div className="notice notice--error">{error}</div> : null}

        <div className="notice notice--info">
          <strong>Running now:</strong>{' '}
          {config.active.configured ? (
            <>
              {config.active.label}
              {config.active.model ? ` · ${config.active.model}` : ''}
              {config.source === 'tenant' ? (
                <span className="chip" style={{ marginLeft: 6 }}>
                  your setting
                </span>
              ) : (
                <span className="chip" style={{ marginLeft: 6 }}>
                  deployment default
                </span>
              )}
            </>
          ) : (
            <span>{config.active.error}</span>
          )}
        </div>

        <div className="picker">
          <div className="picker__providers">
            <p className="panel-title small">On your own hardware</p>
            {local.map((entry) => (
              <ProviderRow
                key={entry.key}
                entry={entry}
                active={entry.key === provider}
                onSelect={() => choose(entry.key)}
              />
            ))}

            <p className="panel-title small" style={{ marginTop: 12 }}>
              Hosted
            </p>
            {hosted.map((entry) => (
              <ProviderRow
                key={entry.key}
                entry={entry}
                active={entry.key === provider}
                onSelect={() => choose(entry.key)}
              />
            ))}
          </div>

          <div className="picker__models">
            {selected ? (
              <>
                <p className="panel-title small">{selected.label}</p>
                {selected.notes ? <p className="small muted">{selected.notes}</p> : null}

                {config.allow_custom_endpoints ? (
                  <div className="field">
                    <label htmlFor="base-url">Endpoint</label>
                    <input
                      id="base-url"
                      value={baseUrl}
                      placeholder={selected.base_url || 'https://your-server/v1'}
                      onChange={(event) => setBaseUrl(event.target.value)}
                    />
                    <div className="hint">
                      Leave blank for the default. Point it anywhere that speaks the
                      same API — your own box, a cluster, a proxy.
                    </div>
                  </div>
                ) : null}

                <div className="field">
                  <label htmlFor="api-key">
                    API key{selected.requires_key ? '' : ' (not needed here)'}
                  </label>
                  <input
                    id="api-key"
                    type="password"
                    value={apiKey}
                    autoComplete="off"
                    placeholder={
                      config.stored?.has_api_key && config.stored.provider === provider
                        ? 'A key is saved — leave blank to keep it'
                        : selected.requires_key
                          ? 'Required by this provider'
                          : 'Optional'
                    }
                    onChange={(event) => setApiKey(event.target.value)}
                  />
                  <div className="hint">
                    Stored encrypted and never sent back to the browser.
                  </div>
                </div>

                <div className="field">
                  <label htmlFor="model-timeout">How long to wait for a reply</label>
                  <input
                    id="model-timeout"
                    inputMode="numeric"
                    value={wait}
                    placeholder={`${config.default_timeout_seconds} seconds (default)`}
                    onChange={(event) =>
                      setWait(event.target.value.replace(/[^0-9]/g, ''))
                    }
                  />
                  <div className="hint">
                    Seconds. A model running on your own machine can take minutes to
                    answer — a reasoning model thinks before it says anything — and a
                    run fails if the wait runs out. Raise it here rather than
                    restarting the server.
                  </div>
                </div>

                <div className="row">
                  <button className="btn" onClick={detect} disabled={busy !== null}>
                    {busy === 'detect' ? 'Looking…' : 'Detect models'}
                  </button>
                  <button
                    className="btn"
                    onClick={runTest}
                    disabled={busy !== null || !model}
                  >
                    {busy === 'test' ? 'Testing…' : 'Test connection'}
                  </button>
                </div>

                {detectError ? (
                  <div className="notice notice--warn">{detectError}</div>
                ) : null}
                {test ? (
                  <div className={`notice ${test.ok ? 'notice--info' : 'notice--error'}`}>
                    {test.message}
                  </div>
                ) : null}

                <p className="panel-title small">
                  {detected
                    ? `${detected.length} model(s) on this endpoint`
                    : 'Commonly used here'}
                </p>

                {offered.length === 0 ? (
                  <p className="small muted">
                    Nothing listed — name the model your server runs below.
                  </p>
                ) : (
                  <div className="picker__list">
                    {offered.map((card) => (
                      <button
                        key={card.id}
                        className={`list-item${card.id === model ? ' is-active' : ''}`}
                        onClick={() => setModel(card.id)}
                      >
                        <span style={{ textAlign: 'left', flex: 1 }}>
                          <strong>{card.label}</strong>
                          {card.open_weights ? (
                            <span className="chip chip--open">open weights</span>
                          ) : null}
                          <div className="muted small">{card.id}</div>
                          {card.notes ? <div className="muted small">{card.notes}</div> : null}
                          <Supports supports={card.supports} />
                        </span>
                      </button>
                    ))}
                  </div>
                )}

                <div className="field" style={{ marginTop: 10 }}>
                  <label htmlFor="model-id">Model</label>
                  <input
                    id="model-id"
                    value={model}
                    placeholder="or type any model your server has"
                    onChange={(event) => setModel(event.target.value)}
                  />
                </div>

                <div className="row">
                  <button
                    className="btn btn--primary"
                    onClick={save}
                    disabled={busy !== null || !provider}
                  >
                    {busy === 'save' ? 'Saving…' : 'Use this model'}
                  </button>
                  {config.source === 'tenant' ? (
                    <button className="btn" onClick={revert} disabled={busy !== null}>
                      Back to the default
                    </button>
                  ) : null}
                  <button className="btn btn--ghost" onClick={onClose}>
                    Cancel
                  </button>
                </div>
              </>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
}

function ProviderRow({
  entry,
  active,
  onSelect,
}: {
  entry: ProviderOption;
  active: boolean;
  onSelect: () => void;
}) {
  return (
    <button className={`list-item${active ? ' is-active' : ''}`} onClick={onSelect}>
      <span style={{ textAlign: 'left' }}>
        <strong>{entry.label}</strong>
        <div className="muted small">
          {entry.requires_key ? 'needs an API key' : 'no key needed'}
        </div>
      </span>
    </button>
  );
}

function Supports({
  supports,
}: {
  supports: { tools: boolean; vision: boolean; thinking: boolean };
}) {
  const badges = [
    supports.tools ? 'tools' : null,
    supports.vision ? 'vision' : null,
    supports.thinking ? 'thinking' : null,
  ].filter(Boolean) as string[];

  return (
    <span className="row" style={{ gap: 4, flexWrap: 'wrap', marginTop: 4 }}>
      {badges.length === 0 ? (
        <span className="chip">text only</span>
      ) : (
        badges.map((badge) => (
          <span className="chip chip--cap" key={badge}>
            {badge}
          </span>
        ))
      )}
    </span>
  );
}
