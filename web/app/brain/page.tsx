'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';

import { AuthGate } from '@/components/AuthGate';
import { Brain } from '@/components/Brain';
import { SAMPLE_MAP } from '@/lib/sample-map';
import { api } from '@/lib/api';
import type { WorkspaceMap } from '@/lib/types';

/**
 * The workspace, full screen.
 *
 * A separate route rather than a panel because it answers a different question
 * from the canvas. The canvas is "what does this workflow do"; this is "what
 * have I built, and what is connected to what" — and that question is worth the
 * whole window.
 */
export default function BrainPage() {
  return (
    <AuthGate>
      {() => <BrainScreen />}
    </AuthGate>
  );
}

function BrainScreen() {
  const [map, setMap] = useState<WorkspaceMap | null>(null);

  useEffect(() => {
    void api
      .workspaceMap()
      .then(setMap)
      .catch(() => setMap(SAMPLE_MAP));
  }, []);

  const empty = map !== null && map.nodes.length === 0;
  const drawn = empty ? SAMPLE_MAP : map;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <strong>Your workspace</strong>
          <small>every agent, skill, workflow and connection</small>
        </div>
        <div className="spacer" />
        {map ? (
          <span className="muted small">
            {map.counts.agents} agents · {map.counts.skills} skills ·{' '}
            {map.counts.workflows} workflows · {map.counts.memories} things learned
          </span>
        ) : null}
        <Link className="btn" href="/">
          Back to the canvas
        </Link>
      </header>

      {drawn ? (
        <div style={{ position: 'relative', minHeight: 0 }}>
          {empty ? (
            <div className="brain__empty">
              Nothing here yet — this is what one looks like once you have built
              something. <Link href="/">Describe a goal</Link> and it fills in.
            </div>
          ) : null}
          <Brain map={drawn} />
        </div>
      ) : (
        <p className="muted small" style={{ padding: 20 }}>
          Loading…
        </p>
      )}
    </div>
  );
}
