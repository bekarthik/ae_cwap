import type { WorkspaceMap } from './types';

/**
 * What a workspace looks like once it has been used.
 *
 * Shown when yours is empty, which is the one moment the map would otherwise be
 * a blank shell — the least useful version of the most explanatory screen. It is
 * plainly labelled as an example wherever it appears, because a demo presented
 * as your own data is a lie the product tells on its first screen.
 */
export const SAMPLE_MAP: WorkspaceMap = {
  nodes: [
    { id: 'wf1', kind: 'workflow', label: 'Competitor briefing', detail: '5 steps', size: 1.7, memories: 4 },
    { id: 'wf2', kind: 'workflow', label: 'Release notes', detail: '4 steps', size: 1.5, memories: 2 },
    { id: 'a1', kind: 'agent', label: 'Researcher', detail: 'Checks claims before repeating them', size: 1.8, memories: 6 },
    { id: 'a2', kind: 'agent', label: 'Analyst', detail: 'Weighs options against what matters', size: 1.6, memories: 3 },
    { id: 'a3', kind: 'agent', label: 'Writer', detail: 'Turns findings into the deliverable', size: 1.5, memories: 5 },
    { id: 'a4', kind: 'agent', label: 'Reviewer', detail: 'Demanding; checks work against its objective', size: 1.4, memories: 7 },
    { id: 'a5', kind: 'agent', label: 'Engineer', detail: 'Writes and lands the change', size: 1.6, memories: 2 },
    { id: 's1', kind: 'skill', label: 'summarise', detail: 'Condense text into its key points', size: 1.9, memories: 3 },
    { id: 's2', kind: 'skill', label: 'search my documents', detail: 'Semantic search over your uploads', size: 1.5, memories: 1 },
    { id: 's3', kind: 'skill', label: 'plan steps', detail: 'Break an objective into ordered work', size: 1.3, memories: 0 },
    { id: 's4', kind: 'skill', label: 'github search code', detail: 'From a connected server', size: 1.4, memories: 2 },
    { id: 's5', kind: 'skill', label: 'github create pull request', detail: 'From a connected server', size: 1.2, memories: 1 },
    { id: 's6', kind: 'skill', label: 'compare options', detail: 'Score alternatives against criteria', size: 1.3, memories: 2 },
    { id: 'm1', kind: 'server', label: 'github', detail: 'Read and write repositories', size: 1.6, memories: 0 },
    { id: 'p1', kind: 'provider', label: 'anthropic', detail: 'model backend', size: 1, memories: 0 },
    { id: 'p2', kind: 'provider', label: 'ollama', detail: 'model backend', size: 1, memories: 0 },
  ],
  links: [
    { source: 'wf1', target: 'a1', kind: 'staffs' },
    { source: 'wf1', target: 'a2', kind: 'staffs' },
    { source: 'wf1', target: 'a3', kind: 'staffs' },
    { source: 'wf1', target: 'a4', kind: 'reviews' },
    { source: 'wf2', target: 'a5', kind: 'staffs' },
    { source: 'wf2', target: 'a3', kind: 'staffs' },
    { source: 'a1', target: 's1', kind: 'holds' },
    { source: 'a1', target: 's2', kind: 'holds' },
    { source: 'a1', target: 's4', kind: 'holds' },
    { source: 'a2', target: 's6', kind: 'holds' },
    { source: 'a2', target: 's1', kind: 'holds' },
    { source: 'a3', target: 's1', kind: 'holds' },
    { source: 'a4', target: 's6', kind: 'holds' },
    { source: 'a5', target: 's4', kind: 'holds' },
    { source: 'a5', target: 's5', kind: 'holds' },
    { source: 'a5', target: 's3', kind: 'holds' },
    { source: 's4', target: 'm1', kind: 'from' },
    { source: 's5', target: 'm1', kind: 'from' },
    { source: 'a1', target: 'p2', kind: 'runs_on' },
    { source: 'a2', target: 'p1', kind: 'runs_on' },
    { source: 'a3', target: 'p1', kind: 'runs_on' },
    { source: 'a5', target: 'p1', kind: 'runs_on' },
  ],
  counts: { agents: 5, skills: 6, servers: 1, workflows: 2, memories: 38 },
};
