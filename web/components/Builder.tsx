'use client';

import {
  Background,
  Controls,
  MiniMap,
  ReactFlow,
  ReactFlowProvider,
  addEdge,
  useEdgesState,
  useNodesState,
  type Connection,
  type NodeTypes,
} from '@xyflow/react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { ApiError, api, openLogSocket } from '@/lib/api';
import {
  NODE_KINDS,
  defaultBindings,
  describeProblems,
  emptyGraph,
  kindFor,
  newId,
  toCanvas,
  toGraph,
  type CanvasEdge,
  type CanvasNode,
} from '@/lib/graph';
import type {
  AgentDefinition,
  KnowledgeSummary,
  LiveOutput,
  LogEvent,
  NodeType,
  RunReport,
  RunStatus,
  RuntimeInfo,
  Session,
  SkillDefinition,
  WorkflowGraph,
  WorkflowSummary,
} from '@/lib/types';

import { DesignModal } from './DesignModal';
import { Inspector } from './Inspector';
import { ConnectorPanel } from './ConnectorPanel';
import { Launcher } from './Launcher';
import { ModelPicker } from './ModelPicker';
import { KnowledgePanel } from './KnowledgePanel';
import { RosterPanel } from './RosterPanel';
import { RunPanel } from './RunPanel';
import { WorkflowNodeCard, type NodeRunState } from './WorkflowNodeCard';

const nodeTypes: NodeTypes = { workflowNode: WorkflowNodeCard };

interface Props {
  session: Session;
  onSignOut: () => void;
}

export function Builder(props: Props) {
  return (
    <ReactFlowProvider>
      <BuilderInner {...props} />
    </ReactFlowProvider>
  );
}

/**
 * Which model this deployment is wired to, stated where the user can see it —
 * and clickable, because "works with any model" is only a real claim if the
 * alternatives are discoverable from the product rather than the README.
 */
function BackendChip({ runtime, onOpen }: { runtime: RuntimeInfo; onOpen: () => void }) {
  const { llm, embeddings } = runtime;
  if (!llm.configured) {
    return (
      <button className="status-pill" data-status="FAILED" title={llm.error} onClick={onOpen}>
        model not configured
      </button>
    );
  }
  const detail = [
    llm.base_url ? `endpoint ${llm.base_url}` : null,
    embeddings.identity ? `embeddings ${embeddings.identity}` : null,
    llm.supports.tools ? 'native tools' : 'prompted tool protocol',
    llm.notes || null,
  ]
    .filter(Boolean)
    .join(' · ');

  return (
    <button className="status-pill" title={detail} onClick={onOpen}>
      {llm.label}
      {llm.model ? ` · ${llm.model}` : ''}
    </button>
  );
}

function BuilderInner({ session, onSignOut }: Props) {
  const initial = useMemo(() => emptyGraph(), []);
  const [meta, setMeta] = useState({
    id: initial.id,
    name: initial.name,
    version: initial.version,
    memory_scope_id: initial.memory_scope_id,
  });

  const canvas = useMemo(() => toCanvas(initial), [initial]);
  const [nodes, setNodes, onNodesChange] = useNodesState<CanvasNode>(canvas.nodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState<CanvasEdge>(canvas.edges);

  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);

  const [workflows, setWorkflows] = useState<WorkflowSummary[]>([]);
  const [corpora, setCorpora] = useState<KnowledgeSummary[]>([]);
  const [runtime, setRuntime] = useState<RuntimeInfo | null>(null);
  const [agents, setAgents] = useState<AgentDefinition[]>([]);
  const [skills, setSkills] = useState<SkillDefinition[]>([]);
  const [goalOpen, setGoalOpen] = useState(false);
  const [modelsOpen, setModelsOpen] = useState(false);
  // Shown on arrival, because an empty canvas answers none of the questions a
  // new user has. Dismissed for the session once they have chosen a way in.
  const [launcherOpen, setLauncherOpen] = useState(true);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const [runStatus, setRunStatus] = useState<RunStatus | null>(null);
  const [logs, setLogs] = useState<LogEvent[]>([]);
  // What the model is writing right now. Held apart from `logs` because these
  // events are never persisted: they are a live view of a step in progress, and
  // the finished text arrives on the step itself a moment later.
  const [live, setLive] = useState<LiveOutput | null>(null);
  const [report, setReport] = useState<RunReport | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const graph = useMemo(() => toGraph(meta, nodes, edges), [meta, nodes, edges]);
  const problems = useMemo(() => describeProblems(graph), [graph]);

  const refreshLists = useCallback(async () => {
    try {
      const [workflowList, corpusList, runtimeInfo, agentList, skillList] =
        await Promise.all([
          api.listWorkflows(),
          api.listKnowledge(),
          api.runtime(),
          api.listAgents(),
          api.listSkills(),
        ]);
      setWorkflows(workflowList);
      setCorpora(corpusList);
      setRuntime(runtimeInfo);
      setAgents(agentList);
      setSkills(skillList);
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : 'Could not load your workspace.');
    }
  }, []);

  useEffect(() => {
    void refreshLists();
  }, [refreshLists]);

  // Close the log stream and stop polling when the builder unmounts, so a
  // navigated-away run leaves nothing running.
  useEffect(
    () => () => {
      socketRef.current?.close();
      if (pollRef.current) clearInterval(pollRef.current);
    },
    [],
  );

  /**
   * Ask the server what actually happened, until the run is over.
   *
   * The WebSocket is how a run feels live; it is not how the canvas knows what
   * a run did. Deriving status from the stream alone means any delivery problem
   * — a worker in another container, a proxy dropping the upgrade, a dead
   * connection — shows a finished run as PENDING forever, which is what a user
   * reported. The run report is the source of truth and is always reachable.
   */
  const watchRun = useCallback(
    (runId: string) => {
      if (pollRef.current) clearInterval(pollRef.current);

      const check = async () => {
        try {
          const report = await api.getRun(runId);
          setRunStatus(report.run.status);
          if (report.run.status === 'SUCCEEDED' || report.run.status === 'FAILED') {
            if (pollRef.current) clearInterval(pollRef.current);
            pollRef.current = null;
            setReport(report);
            if (report.run.error) setRunError(report.run.error);
            // The stream may have missed events; the persisted feed has all of
            // them, so the panel ends up complete either way.
            setLogs(report.logs);
          }
        } catch {
          // A transient failure here is not worth surfacing — the next tick
          // will try again, and the stream may well deliver first.
        }
      };

      pollRef.current = setInterval(check, 2000);
      void check();
    },
    [],
  );

  // ---- canvas editing -------------------------------------------------

  const setRunState = useCallback(
    (nodeId: string | null, state: NodeRunState) => {
      setNodes((current) =>
        current.map((node) =>
          nodeId === null || node.id === nodeId
            ? { ...node, data: { ...node.data, runState: state } }
            : node,
        ),
      );
    },
    [setNodes],
  );

  const addNode = useCallback(
    (type: NodeType) => {
      const kind = kindFor(type);
      const id = `${type}_${newId('n').slice(2, 8)}`;
      setNodes((current) => [
        ...current,
        {
          id,
          type: 'workflowNode',
          position: {
            x: 120 + ((current.length * 60) % 420),
            y: 40 + ((current.length * 90) % 320),
          },
          data: {
            label: kind.label,
            nodeType: type,
            params: structuredClone(kind.defaultParams),
            knowledgeHandle: null,
            agentId: null,
          },
        },
      ]);
      setSelectedNodeId(id);
      setSelectedEdgeId(null);
    },
    [setNodes],
  );

  const onConnect = useCallback(
    (connection: Connection) => {
      const source = nodes.find((node) => node.id === connection.source);
      const target = nodes.find((node) => node.id === connection.target);
      if (!source || !target) return;

      setEdges((current) =>
        addEdge(
          {
            ...connection,
            id: newId('e'),
            label:
              connection.sourceHandle === 'true'
                ? 'yes'
                : connection.sourceHandle === 'false'
                  ? 'no'
                  : undefined,
            data: {
              // Pre-fill a binding that matches the node types being joined, so
              // a connection is useful the moment it is drawn.
              bindings: defaultBindings(source.data.nodeType, target.data.nodeType),
              condition:
                connection.sourceHandle === 'true'
                  ? true
                  : connection.sourceHandle === 'false'
                    ? false
                    : null,
            },
          },
          current,
        ),
      );
    },
    [nodes, setEdges],
  );

  const patchNode = useCallback(
    (id: string, patch: Partial<CanvasNode['data']>) => {
      setNodes((current) =>
        current.map((node) =>
          node.id === id ? { ...node, data: { ...node.data, ...patch } } : node,
        ),
      );
    },
    [setNodes],
  );

  const patchParams = useCallback(
    (id: string, params: Record<string, unknown>) => patchNode(id, { params }),
    [patchNode],
  );

  const patchEdgeBindings = useCallback(
    (id: string, bindings: Record<string, string>) => {
      setEdges((current) =>
        current.map((edge) =>
          edge.id === id ? { ...edge, data: { ...(edge.data ?? {}), bindings } } : edge,
        ),
      );
    },
    [setEdges],
  );

  const deleteNode = useCallback(
    (id: string) => {
      setNodes((current) => current.filter((node) => node.id !== id));
      setEdges((current) =>
        current.filter((edge) => edge.source !== id && edge.target !== id),
      );
      setSelectedNodeId(null);
    },
    [setEdges, setNodes],
  );

  const deleteEdge = useCallback(
    (id: string) => {
      setEdges((current) => current.filter((edge) => edge.id !== id));
      setSelectedEdgeId(null);
    },
    [setEdges],
  );

  // ---- workflow persistence ------------------------------------------

  const applyGraph = useCallback(
    (next: WorkflowGraph) => {
      const converted = toCanvas(next);
      setMeta({
        id: next.id,
        name: next.name,
        version: next.version,
        memory_scope_id: next.memory_scope_id,
      });
      setNodes(converted.nodes);
      setEdges(converted.edges);
      setSelectedNodeId(null);
      setSelectedEdgeId(null);
      setReport(null);
      setLogs([]);
      setLive(null);
      setRunStatus(null);
    },
    [setEdges, setNodes],
  );

  async function save() {
    setSaving(true);
    setError(null);
    setNotice(null);
    try {
      const saved = await api.saveWorkflow(graph);
      setMeta((current) => ({ ...current, version: saved.version }));
      setNotice(`Saved “${graph.name}” (version ${saved.version}).`);
      await refreshLists();
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : 'Save failed.');
    } finally {
      setSaving(false);
    }
  }

  async function open(workflowId: string) {
    setError(null);
    try {
      const loaded = await api.getWorkflow(workflowId);
      applyGraph(loaded.graph);
      setNotice(`Opened “${loaded.name}”.`);
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : 'Could not open workflow.');
    }
  }

  // ---- running --------------------------------------------------------

  async function run() {
    setRunError(null);
    setError(null);
    setNotice(null);
    setLogs([]);
    setReport(null);
    setRunState(null, 'idle');

    if (problems.length) {
      setRunError(problems[0]);
      return;
    }

    let runId: string;
    try {
      const started = await api.startRun(graph.id, {}, graph);
      runId = started.run_id;
      setRunStatus(started.status);
    } catch (caught) {
      setRunStatus('FAILED');
      setRunError(caught instanceof ApiError ? caught.message : 'Could not start the run.');
      return;
    }

    // Started before the socket, and independent of it: this is what makes a
    // run observable even where the live feed cannot reach.
    watchRun(runId);

    socketRef.current?.close();
    const socket = openLogSocket(runId);
    if (!socket) return;
    socketRef.current = socket;

    socket.onmessage = (message) => {
      const event = JSON.parse(message.data as string) as LogEvent & { event: string };
      if (event.event === 'keepalive') return;

      // A fragment of the answer being written. Appended to the live view
      // rather than to the log, which would otherwise be thousands of lines of
      // a single sentence arriving a few words at a time.
      if (event.data?.transient) {
        const kind = (event.data?.kind as string) ?? 'content';
        const fragment = (event.data?.text as string) ?? '';
        setLive((current) =>
          current && current.node === event.node && current.kind === kind
            ? { ...current, text: current.text + fragment }
            : { node: event.node ?? null, kind, text: fragment },
        );
        return;
      }

      setLogs((current) => [...current, event]);

      if (event.event === 'step.started' && event.node) {
        setRunStatus('RUNNING');
        setRunState(event.node, 'running');
        setLive(null);
      } else if (event.event === 'step.completed' && event.node) {
        setRunState(event.node, 'done');
        // The step's real output is now in the report; the preview has served
        // its purpose and would only be a stale duplicate of it.
        setLive(null);
      } else if (event.event === 'run.failed') {
        if (event.node) setRunState(event.node, 'error');
        setRunStatus('FAILED');
        setRunError(event.message);
      } else if (event.event === 'run.succeeded') {
        setRunStatus('SUCCEEDED');
      }

      if (event.event === 'run.succeeded' || event.event === 'run.failed') {
        void api
          .getRun(runId)
          .then(setReport)
          .catch(() => undefined);
      }
    };

    // Not an error the user needs to act on: polling above keeps the run
    // observable, so a dropped stream costs the live tail and nothing else.
    socket.onerror = () => undefined;
  }

  const selectedNode = nodes.find((node) => node.id === selectedNodeId) ?? null;
  const selectedEdge = edges.find((edge) => edge.id === selectedEdgeId) ?? null;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          Cognitive Workflows <small>no-code agent builder</small>
        </div>

        <input
          aria-label="Workflow name"
          value={meta.name}
          onChange={(event) => setMeta({ ...meta, name: event.target.value })}
          style={{ width: 260 }}
        />
        <span className="muted small">v{meta.version}</span>

        <div className="spacer" />

        {runtime ? (
          <BackendChip runtime={runtime} onOpen={() => setModelsOpen(true)} />
        ) : null}

        <button className="btn" onClick={() => setLauncherOpen(true)}>
          New workflow
        </button>
        <button className="btn" onClick={() => setGoalOpen(true)}>
          Describe a goal
        </button>
        <button className="btn" onClick={save} disabled={saving}>
          {saving ? 'Saving…' : 'Save'}
        </button>
        <button className="btn btn--primary" onClick={run} disabled={problems.length > 0}>
          Run
        </button>
        <button className="btn btn--ghost" onClick={onSignOut} title={session.user_id}>
          Sign out
        </button>
      </header>

      <div className="workspace">
        <aside className="panel">
          <div className="panel-section">
            <p className="panel-title">Add a step</p>
            {NODE_KINDS.map((kind) => (
              <button
                className="palette-item"
                key={kind.type}
                onClick={() => addNode(kind.type)}
              >
                <span className="palette-icon" style={{ background: kind.accent }}>
                  {kind.icon}
                </span>
                <span>
                  <div className="palette-label">{kind.label}</div>
                  <div className="palette-blurb">{kind.blurb}</div>
                </span>
              </button>
            ))}
          </div>

          <RosterPanel agents={agents} skills={skills} onChanged={refreshLists} />

          <ConnectorPanel onChanged={refreshLists} />

          <KnowledgePanel corpora={corpora} onChanged={refreshLists} />

          <div className="panel-section">
            <p className="panel-title">Saved workflows</p>
            {workflows.length === 0 ? (
              <p className="muted small">Nothing saved yet.</p>
            ) : (
              workflows.map((workflow) => (
                <button
                  className={`list-item${workflow.id === meta.id ? ' is-active' : ''}`}
                  key={workflow.id}
                  style={{ width: '100%', textAlign: 'left', background: 'transparent' }}
                  onClick={() => open(workflow.id)}
                >
                  <span style={{ flex: 1 }}>
                    <strong>{workflow.name}</strong>
                    <div className="muted small">
                      {workflow.node_count} steps · v{workflow.version}
                    </div>
                  </span>
                </button>
              ))
            )}
            <button
              className="btn btn--block"
              style={{ marginTop: 6 }}
              onClick={() => applyGraph(emptyGraph())}
            >
              Blank canvas
            </button>
          </div>
        </aside>

        <main className="canvas-wrap">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={(_event, node) => {
              setSelectedNodeId(node.id);
              setSelectedEdgeId(null);
            }}
            onEdgeClick={(_event, edge) => {
              setSelectedEdgeId(edge.id);
              setSelectedNodeId(null);
            }}
            onPaneClick={() => {
              setSelectedNodeId(null);
              setSelectedEdgeId(null);
            }}
            fitView
            proOptions={{ hideAttribution: true }}
          >
            <Background gap={18} size={1} />
            <Controls />
            <MiniMap pannable zoomable />
          </ReactFlow>

          {error || notice || problems.length > 0 ? (
            <div
              style={{
                position: 'absolute',
                left: 12,
                bottom: 12,
                right: 12,
                pointerEvents: 'none',
              }}
            >
              {error ? <div className="notice notice--error">{error}</div> : null}
              {notice ? <div className="notice notice--info">{notice}</div> : null}
              {problems.length ? (
                <div className="notice notice--warn">
                  <strong>Not ready to run:</strong>
                  <ul style={{ margin: '4px 0 0', paddingLeft: 18 }}>
                    {problems.map((problem) => (
                      <li key={problem}>{problem}</li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </div>
          ) : null}
        </main>

        <aside className="panel panel--right">
          <Inspector
            node={selectedNode}
            edge={selectedEdge}
            corpora={corpora}
            runtime={runtime}
            agents={agents}
            skills={skills}
            onPatchNode={patchNode}
            onPatchParams={patchParams}
            onPatchEdgeBindings={patchEdgeBindings}
            onDeleteNode={deleteNode}
            onDeleteEdge={deleteEdge}
          />
          <RunPanel
            status={runStatus}
            logs={logs}
            live={live}
            report={report}
            error={runError}
          />
        </aside>
      </div>

      {launcherOpen ? (
        <Launcher
          onDescribeGoal={() => {
            setLauncherOpen(false);
            setGoalOpen(true);
          }}
          onBlankCanvas={() => {
            applyGraph(emptyGraph());
            setLauncherOpen(false);
            setNotice('Empty canvas. Add a step from the left, then draw connections.');
          }}
          onUseTemplate={async (key) => {
            const instantiated = await api.useTemplate(key);
            applyGraph(instantiated.graph);
            setLauncherOpen(false);
            // The template just created agents and possibly skills; the roster
            // and the Inspector's picker have to see them immediately.
            void refreshLists();
            setNotice(
              `“${instantiated.graph.name}” is yours to edit — ${instantiated.notes[0] ?? ''}`,
            );
          }}
          onDismiss={() => setLauncherOpen(false)}
        />
      ) : null}

      {modelsOpen ? (
        <ModelPicker
          onClose={() => setModelsOpen(false)}
          // A model change alters what the canvas should offer — an effort
          // selector against Claude, a temperature slider against Llama — so the
          // runtime capabilities are re-read rather than left stale.
          onSaved={() => void refreshLists()}
        />
      ) : null}

      {goalOpen ? (
        <DesignModal
          onClose={() => setGoalOpen(false)}
          onAccept={(design) => {
            if (!design.graph) return;
            applyGraph(design.graph);
            setGoalOpen(false);
            // The agents and any skills built for them exist now, so the roster
            // and the Inspector's agent picker must reflect that immediately.
            void refreshLists();
            setNotice(
              `${design.agents.length} agent(s) placed on the canvas — review each one, then run.`,
            );
          }}
        />
      ) : null}
    </div>
  );
}
