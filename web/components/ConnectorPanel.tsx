'use client';

import { useCallback, useEffect, useState } from 'react';

import { ApiError, api } from '@/lib/api';
import type {
  MCPDirectoryEntry,
  MCPPolicy,
  MCPServerRecord,
  MCPTransport,
} from '@/lib/types';

interface Props {
  onChanged: () => Promise<void> | void;
}

/**
 * Connecting the platform to systems it did not write a connector for.
 *
 * An MCP server's tools become **skills**, so nothing else in the product has to
 * learn a new concept: an agent that can open a pull request is an agent holding
 * a skill, exactly like one that can summarise text. That is why this panel sits
 * next to Agents and Skills rather than in a settings page.
 *
 * The two transports are presented differently on purpose. An HTTP server is a
 * URL, which most people can supply. A stdio server launches a process on the
 * worker, so it is only offered when an administrator has allow-listed commands
 * — saying so up front beats letting someone fill in a form that will be refused.
 *
 * The dialog opens on the servers the platform already knows how to reach rather
 * than on an empty URL field. "Which of these?" is answerable; "paste an MCP
 * endpoint" is a question most people cannot answer on the spot — and the ones
 * who can would still have hit an empty outbound allow-list.
 */
export function ConnectorPanel({ onChanged }: Props) {
  const [servers, setServers] = useState<MCPServerRecord[]>([]);
  const [policy, setPolicy] = useState<MCPPolicy | null>(null);
  const [known, setKnown] = useState<MCPDirectoryEntry[]>([]);
  const [adding, setAdding] = useState(false);
  const [openId, setOpenId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const [body, listed] = await Promise.all([api.listMcpServers(), api.mcpDirectory()]);
      setServers(body.servers);
      setPolicy(body.policy);
      setKnown(listed.servers);
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : 'Could not load connectors.');
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  async function refresh() {
    await load();
    await onChanged();
  }

  async function sync(id: string) {
    setNotice(null);
    try {
      const result = await api.syncMcpServer(id);
      setNotice(`${result.skills} tool(s) available as skills.`);
      await refresh();
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : 'Sync failed.');
    }
  }

  async function toggle(server: MCPServerRecord) {
    await api.setMcpServerEnabled(server.id, !server.enabled);
    await refresh();
  }

  async function remove(id: string) {
    await api.disconnectMcpServer(id);
    await refresh();
  }

  return (
    <div className="panel-section">
      <p className="panel-title">Connected systems</p>

      {error ? <div className="notice notice--error">{error}</div> : null}
      {notice ? <div className="notice notice--info">{notice}</div> : null}

      {servers.length === 0 ? (
        <p className="muted small">
          Nothing connected. Connect a system and its tools become skills your
          agents can use — a repository, a filesystem, an internal service.
          {known.some((entry) => entry.available)
            ? ` ${known.filter((entry) => entry.available).length} are ready to connect.`
            : ''}
        </p>
      ) : (
        servers.map((server) => (
          <div className="roster-item" key={server.id}>
            <button
              className="roster-item__head"
              onClick={() => setOpenId(openId === server.id ? null : server.id)}
            >
              <span style={{ flex: 1, textAlign: 'left' }}>
                <strong>{server.name}</strong>
                {!server.enabled ? <span className="chip">off</span> : null}
                {server.last_error ? <span className="chip chip--bad">error</span> : null}
                <div className="muted small">
                  {server.config.transport} · {server.tools.length} tool(s)
                </div>
              </span>
              <span className="muted small">{openId === server.id ? '−' : '+'}</span>
            </button>

            {openId === server.id ? (
              <div className="roster-item__body">
                {server.description ? (
                  <p className="small">{server.description}</p>
                ) : null}
                {server.last_error ? (
                  <div className="notice notice--error">{server.last_error}</div>
                ) : null}

                <p className="panel-title small">Tools</p>
                <ul className="memory__list">
                  {server.tools.map((tool) => (
                    <li className="memory__item" key={tool.name}>
                      <span className="memory__kind">
                        {tool.read_only ? 'read' : 'write'}
                      </span>
                      <span className="memory__text">
                        <strong>{tool.name}</strong>
                        <div className="muted small">{tool.description}</div>
                      </span>
                    </li>
                  ))}
                </ul>

                <div className="row" style={{ marginTop: 8, flexWrap: 'wrap' }}>
                  <button className="btn small" onClick={() => void sync(server.id)}>
                    Re-read tools
                  </button>
                  <button className="btn small" onClick={() => void toggle(server)}>
                    {server.enabled ? 'Switch off' : 'Switch on'}
                  </button>
                  <button
                    className="btn btn--danger small"
                    onClick={() => void remove(server.id)}
                  >
                    Disconnect
                  </button>
                </div>
              </div>
            ) : null}
          </div>
        ))
      )}

      <button
        className="btn btn--block"
        style={{ marginTop: 6 }}
        onClick={() => setAdding(true)}
      >
        Connect a system
      </button>

      {adding && policy ? (
        <ConnectDialog
          policy={policy}
          known={known}
          onClose={() => setAdding(false)}
          onConnected={async (message) => {
            setNotice(message);
            setAdding(false);
            await refresh();
          }}
        />
      ) : null}
    </div>
  );
}

function ConnectDialog({
  policy,
  known,
  onClose,
  onConnected,
}: {
  policy: MCPPolicy;
  known: MCPDirectoryEntry[];
  onClose: () => void;
  onConnected: (message: string) => Promise<void> | void;
}) {
  const [transport, setTransport] = useState<MCPTransport>('http');
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [url, setUrl] = useState('');
  const [command, setCommand] = useState('');
  const [args, setArgs] = useState('');
  const [secretName, setSecretName] = useState('');
  const [secretValue, setSecretValue] = useState('');
  // The listed entry this form was filled from, so the credential can be
  // labelled with where to get it and sent with the prefix that server wants.
  const [chosen, setChosen] = useState<MCPDirectoryEntry | null>(null);
  const [manual, setManual] = useState(false);

  function pick(entry: MCPDirectoryEntry) {
    setChosen(entry);
    setManual(true);
    setResult(null);
    setTransport(entry.transport);
    setName(entry.key.replace(/-/g, '_'));
    setDescription(entry.description);
    setUrl(entry.url);
    setCommand(entry.command);
    setArgs(entry.args.join(' ') + (entry.argument_hint ? ' ' : ''));
    setSecretName(entry.credentials[0]?.name ?? '');
    setSecretValue('');
  }

  function startBlank() {
    setChosen(null);
    setManual(true);
    setTransport(policy.stdio_enabled ? 'stdio' : 'http');
  }

  /** The credential as the server wants it — "Bearer <token>", not "<token>". */
  function secret(): Record<string, string> {
    if (!secretName || !secretValue) return {};
    const prefix = chosen?.credentials.find((entry) => entry.name === secretName)?.prefix ?? '';
    const value =
      prefix && !secretValue.startsWith(prefix) ? `${prefix}${secretValue}` : secretValue;
    return { [secretName]: value };
  }

  const [busy, setBusy] = useState<'test' | 'connect' | null>(null);
  const [result, setResult] = useState<{ ok: boolean; message: string } | null>(null);

  function payload() {
    return {
      name,
      description,
      config:
        transport === 'http'
          ? { transport, url, command: '', args: [], env: {} }
          : {
              transport,
              command,
              args: args.split(/\s+/).filter(Boolean),
              env: {},
              url: '',
            },
      credentials: secret(),
    };
  }

  async function test() {
    setBusy('test');
    setResult(null);
    try {
      const body = payload();
      setResult(await api.testMcpServer(body.config, body.credentials));
    } catch (caught) {
      setResult({
        ok: false,
        message: caught instanceof ApiError ? caught.message : 'Test failed.',
      });
    } finally {
      setBusy(null);
    }
  }

  async function connect() {
    setBusy('connect');
    setResult(null);
    try {
      const body = await api.connectMcpServer(payload());
      await onConnected(body.message);
    } catch (caught) {
      setResult({
        ok: false,
        message: caught instanceof ApiError ? caught.message : 'Could not connect.',
      });
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="modal-backdrop" onClick={onClose} role="presentation">
      <div
        className="modal"
        onClick={(event) => event.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="Connect a system"
      >
        <h2>Connect a system</h2>
        <p className="lede">
          Any MCP server. Its tools become skills your agents can be given —
          nothing else about a workflow changes.
        </p>

        {!manual ? (
          <Catalogue known={known} onPick={pick} onBlank={startBlank} />
        ) : null}

        {manual && chosen ? (
          <div className="notice notice--info">
            <strong>{chosen.label}</strong> — {chosen.summary}{' '}
            <a href={chosen.docs_url} target="_blank" rel="noreferrer">
              docs
            </a>
            . Everything below is editable; if the endpoint has moved, correct it
            here.{' '}
            <button className="btn--link" onClick={() => setManual(false)}>
              Pick a different one
            </button>
          </div>
        ) : null}

        <div className="tabs" hidden={!manual || chosen !== null}>
          <button
            className={`tab${transport === 'http' ? ' is-active' : ''}`}
            onClick={() => setTransport('http')}
          >
            A URL
          </button>
          <button
            className={`tab${transport === 'stdio' ? ' is-active' : ''}`}
            onClick={() => setTransport('stdio')}
            disabled={!policy.stdio_enabled}
            title={
              policy.stdio_enabled
                ? undefined
                : 'An administrator has not allowed any commands on this deployment'
            }
          >
            A local command
          </button>
        </div>

        <div className="field" hidden={!manual}>
          <label htmlFor="mcp-name">Name</label>
          <input
            id="mcp-name"
            value={name}
            placeholder="github"
            onChange={(event) => setName(event.target.value.toLowerCase())}
          />
          <div className="hint">
            Lower case. Its tools become skills named{' '}
            <code>{name || 'name'}_&lt;tool&gt;</code>.
          </div>
        </div>

        <div className="field" hidden={!manual}>
          <label htmlFor="mcp-description">What it is for</label>
          <input
            id="mcp-description"
            value={description}
            placeholder="Read and write source code"
            onChange={(event) => setDescription(event.target.value)}
          />
        </div>

        {manual && transport === 'http' ? (
          <div className="field">
            <label htmlFor="mcp-url">Server URL</label>
            <input
              id="mcp-url"
              value={url}
              placeholder="https://mcp.example.com/mcp"
              onChange={(event) => setUrl(event.target.value)}
            />
            <div className="hint">
              {chosen
                ? 'This server is one this platform ships with, so it needs no allow-listing.'
                : policy.allowed_hosts.length
                  ? `Any other host must be allow-listed. Allowed here: ${policy.allowed_hosts.join(', ')}`
                  : 'A host that is not on the list above must be added to CWAP_HTTP_ALLOWLIST by an administrator.'}
            </div>
          </div>
        ) : manual ? (
          <>
            <div className="field">
              <label htmlFor="mcp-command">Command</label>
              <input
                id="mcp-command"
                value={command}
                placeholder={policy.allowed_commands[0] ?? 'npx'}
                onChange={(event) => setCommand(event.target.value)}
              />
              <div className="hint">
                Allowed on this deployment: {policy.allowed_commands.join(', ') || 'none'}
              </div>
            </div>
            <div className="field">
              <label htmlFor="mcp-args">Arguments</label>
              <input
                id="mcp-args"
                value={args}
                placeholder="-y @modelcontextprotocol/server-github"
                onChange={(event) => setArgs(event.target.value)}
              />
              {chosen?.argument_hint ? (
                <div className="hint">{chosen.argument_hint}</div>
              ) : null}
            </div>
          </>
        ) : null}

        <div className="field" hidden={!manual}>
          <label htmlFor="mcp-secret-name">
            {chosen?.credentials[0]?.label ??
              `Credential ${transport === 'http' ? '(header)' : '(environment variable)'}`}
            {chosen?.credentials[0]?.required === false ? ' (optional)' : ''}
          </label>
          <div className="row">
            <input
              id="mcp-secret-name"
              value={secretName}
              placeholder={transport === 'http' ? 'Authorization' : 'GITHUB_TOKEN'}
              onChange={(event) => setSecretName(event.target.value)}
            />
            <input
              type="password"
              value={secretValue}
              placeholder={
                chosen?.credentials[0]?.required === false
                  ? 'Optional'
                  : chosen?.credentials.length
                    ? 'Required by this server'
                    : 'Optional'
              }
              autoComplete="off"
              onChange={(event) => setSecretValue(event.target.value)}
            />
          </div>
          <div className="hint">
            {chosen?.credentials[0]?.how ? `${chosen.credentials[0].how} ` : ''}
            Stored encrypted and never sent back to the browser.
          </div>
        </div>

        {result ? (
          <div className={`notice ${result.ok ? 'notice--info' : 'notice--error'}`}>
            {result.message}
          </div>
        ) : null}

        <div className="row" hidden={!manual}>
          <button className="btn" onClick={test} disabled={busy !== null}>
            {busy === 'test' ? 'Trying…' : 'Test'}
          </button>
          <button
            className="btn btn--primary"
            onClick={connect}
            disabled={busy !== null || !name || (transport === 'http' ? !url : !command)}
          >
            {busy === 'connect' ? 'Connecting…' : 'Connect'}
          </button>
          <button className="btn btn--ghost" onClick={onClose}>
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}

/**
 * The servers this deployment can reach, grouped so the list reads as choices
 * rather than as inventory.
 *
 * An entry that cannot be connected here is still shown, with the reason. The
 * alternative — hiding it — leaves someone wondering whether the platform
 * supports their system at all, when the honest answer is "it does, and an
 * administrator has to allow it first".
 */
function Catalogue({
  known,
  onPick,
  onBlank,
}: {
  known: MCPDirectoryEntry[];
  onPick: (entry: MCPDirectoryEntry) => void;
  onBlank: () => void;
}) {
  const categories = known.reduce<Record<string, MCPDirectoryEntry[]>>((groups, entry) => {
    (groups[entry.category] ??= []).push(entry);
    return groups;
  }, {});

  return (
    <>
      {Object.entries(categories).map(([category, entries]) => (
        <div key={category}>
          <p className="panel-title small">{category}</p>
          {entries.map((entry) => (
            <button
              key={entry.key}
              className="list-item"
              style={{
                width: '100%',
                textAlign: 'left',
                opacity: entry.available ? 1 : 0.55,
                cursor: entry.available ? 'pointer' : 'not-allowed',
              }}
              disabled={!entry.available}
              title={entry.blocked_reason || undefined}
              onClick={() => onPick(entry)}
            >
              <span style={{ flex: 1 }}>
                <strong>{entry.label}</strong>
                {entry.read_only ? <span className="chip">read only</span> : null}
                {entry.credentials.some((credential) => credential.required) ? (
                  <span className="chip">needs a token</span>
                ) : null}
                <div className="muted small">
                  {entry.available ? entry.summary : entry.blocked_reason}
                </div>
              </span>
            </button>
          ))}
        </div>
      ))}

      <button className="btn btn--block" style={{ marginTop: 8 }} onClick={onBlank}>
        Something else — I have the details
      </button>
    </>
  );
}
