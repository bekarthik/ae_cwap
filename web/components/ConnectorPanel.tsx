'use client';

import { useCallback, useEffect, useState } from 'react';

import { ApiError, api } from '@/lib/api';
import type { MCPPolicy, MCPServerRecord, MCPTransport } from '@/lib/types';

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
 */
export function ConnectorPanel({ onChanged }: Props) {
  const [servers, setServers] = useState<MCPServerRecord[]>([]);
  const [policy, setPolicy] = useState<MCPPolicy | null>(null);
  const [adding, setAdding] = useState(false);
  const [openId, setOpenId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const body = await api.listMcpServers();
      setServers(body.servers);
      setPolicy(body.policy);
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
          Nothing connected. Add an MCP server and its tools become skills your
          agents can use — a repository, a filesystem, an internal service.
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
  onClose,
  onConnected,
}: {
  policy: MCPPolicy;
  onClose: () => void;
  onConnected: (message: string) => Promise<void> | void;
}) {
  const [transport, setTransport] = useState<MCPTransport>(
    policy.stdio_enabled ? 'stdio' : 'http',
  );
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [url, setUrl] = useState('');
  const [command, setCommand] = useState('');
  const [args, setArgs] = useState('');
  const [secretName, setSecretName] = useState('');
  const [secretValue, setSecretValue] = useState('');

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
      credentials: secretName && secretValue ? { [secretName]: secretValue } : {},
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

        <div className="tabs">
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

        <div className="field">
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

        <div className="field">
          <label htmlFor="mcp-description">What it is for</label>
          <input
            id="mcp-description"
            value={description}
            placeholder="Read and write source code"
            onChange={(event) => setDescription(event.target.value)}
          />
        </div>

        {transport === 'http' ? (
          <div className="field">
            <label htmlFor="mcp-url">Server URL</label>
            <input
              id="mcp-url"
              value={url}
              placeholder="https://mcp.example.com/mcp"
              onChange={(event) => setUrl(event.target.value)}
            />
            <div className="hint">
              {policy.allowed_hosts.length
                ? `Allowed hosts: ${policy.allowed_hosts.join(', ')}`
                : 'No outbound hosts are allow-listed yet — an administrator sets CWAP_HTTP_ALLOWLIST.'}
            </div>
          </div>
        ) : (
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
            </div>
          </>
        )}

        <div className="field">
          <label htmlFor="mcp-secret-name">
            Credential {transport === 'http' ? '(header)' : '(environment variable)'}
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
              placeholder="Optional"
              autoComplete="off"
              onChange={(event) => setSecretValue(event.target.value)}
            />
          </div>
          <div className="hint">Stored encrypted and never sent back to the browser.</div>
        </div>

        {result ? (
          <div className={`notice ${result.ok ? 'notice--info' : 'notice--error'}`}>
            {result.message}
          </div>
        ) : null}

        <div className="row">
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
