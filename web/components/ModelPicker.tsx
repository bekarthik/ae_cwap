'use client';

import { useState } from 'react';

import type { RuntimeInfo } from '@/lib/types';

interface Props {
  runtime: RuntimeInfo;
  onClose: () => void;
}

/**
 * Which model backend this deployment runs, and what else it could run.
 *
 * The platform is model-agnostic, which is only useful if choosing a model is
 * something a person can actually do. So instead of a documentation page saying
 * "set CWAP_LLM_PROVIDER", this lists the backends people actually run — a model
 * on your own laptop, a self-hosted server, a hosted API — with the models each
 * one commonly serves and what each of those can do.
 *
 * It shows the exact settings to apply rather than applying them itself:
 * the provider is process-wide server configuration, and a running deployment
 * switching models underneath other people's in-flight runs is not something a
 * browser session should be able to do.
 */
export function ModelPicker({ runtime, onClose }: Props) {
  const [selected, setSelected] = useState<string>(runtime.llm.provider);
  const provider =
    runtime.available_providers.find((entry) => entry.key === selected) ?? null;
  const [model, setModel] = useState<string>(runtime.llm.model ?? '');

  const local = runtime.available_providers.filter((entry) => entry.local);
  const hosted = runtime.available_providers.filter((entry) => !entry.local);

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
          Everything here runs the same workflows. Agents work on models without
          native tool calling too — the platform falls back to a prompted protocol
          automatically.
        </p>

        <div className="notice notice--info">
          <strong>Running now:</strong>{' '}
          {runtime.llm.configured ? (
            <>
              {runtime.llm.label}
              {runtime.llm.model ? ` · ${runtime.llm.model}` : ''}
              <Supports supports={runtime.llm.supports} />
            </>
          ) : (
            <span>{runtime.llm.error}</span>
          )}
        </div>

        <div className="picker">
          <div className="picker__providers">
            <p className="panel-title small">On your own hardware</p>
            {local.map((entry) => (
              <button
                key={entry.key}
                className={`list-item${entry.key === selected ? ' is-active' : ''}`}
                onClick={() => {
                  setSelected(entry.key);
                  setModel(entry.default_model);
                }}
              >
                <span style={{ textAlign: 'left' }}>
                  <strong>{entry.label}</strong>
                  <div className="muted small">no key needed</div>
                </span>
              </button>
            ))}

            <p className="panel-title small" style={{ marginTop: 12 }}>
              Hosted
            </p>
            {hosted.map((entry) => (
              <button
                key={entry.key}
                className={`list-item${entry.key === selected ? ' is-active' : ''}`}
                onClick={() => {
                  setSelected(entry.key);
                  setModel(entry.default_model);
                }}
              >
                <span style={{ textAlign: 'left' }}>
                  <strong>{entry.label}</strong>
                  <div className="muted small">
                    {entry.requires_key ? 'needs an API key' : 'no key needed'}
                  </div>
                </span>
              </button>
            ))}
          </div>

          <div className="picker__models">
            {provider ? (
              <>
                <p className="panel-title small">{provider.label}</p>
                {provider.notes ? <p className="small muted">{provider.notes}</p> : null}

                {provider.models.length === 0 ? (
                  <p className="small muted">
                    This server runs whatever model you loaded into it, so name it
                    yourself below.
                  </p>
                ) : (
                  provider.models.map((card) => (
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
                  ))
                )}

                <div className="field" style={{ marginTop: 10 }}>
                  <label htmlFor="model-id">Model id</label>
                  <input
                    id="model-id"
                    value={model}
                    placeholder="or type any model your server has"
                    onChange={(event) => setModel(event.target.value)}
                  />
                </div>

                <p className="panel-title small">To switch, set these and restart</p>
                <pre className="config">
                  {[
                    `CWAP_LLM_PROVIDER=${provider.key}`,
                    model ? `CWAP_LLM_MODEL=${model}` : null,
                    provider.requires_key ? 'CWAP_LLM_API_KEY=<your key>' : null,
                    provider.base_url ? null : 'CWAP_LLM_BASE_URL=<your endpoint>',
                  ]
                    .filter(Boolean)
                    .join('\n')}
                </pre>
                <p className="hint">
                  The provider is server configuration, so it is not switched from the
                  browser — a running deployment changing model underneath someone
                  else&apos;s in-flight run is not something a page should be able to do.
                </p>
              </>
            ) : null}
          </div>
        </div>

        <button className="btn btn--block" onClick={onClose}>
          Close
        </button>
      </div>
    </div>
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
