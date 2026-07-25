'use client';

import { useEffect, useState } from 'react';

import { ApiError, api, clearSession, loadSession, storeSession } from '@/lib/api';
import type { Session } from '@/lib/types';

interface Props {
  children: (session: Session, signOut: () => void) => React.ReactNode;
}

export function AuthGate({ children }: Props) {
  const [session, setSession] = useState<Session | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    setSession(loadSession());
    setReady(true);
  }, []);

  if (!ready) return null;

  if (!session) {
    return <SignIn onSignedIn={setSession} />;
  }

  return (
    <>
      {children(session, () => {
        clearSession();
        setSession(null);
      })}
    </>
  );
}

function SignIn({ onSignedIn }: { onSignedIn: (session: Session) => void }) {
  const [mode, setMode] = useState<'login' | 'register'>('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const session =
        mode === 'login'
          ? await api.login(email, password)
          : await api.register(email, password);
      storeSession(session);
      onSignedIn(session);
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : 'Something went wrong.');
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="auth-shell">
      <form className="auth-card" onSubmit={submit}>
        <div className="brand" style={{ marginBottom: 4 }}>
          Cognitive Workflows
        </div>
        <p className="muted small" style={{ marginTop: 0, marginBottom: 20 }}>
          Design multi-step AI workflows by connecting steps on a canvas.
        </p>

        {error ? <div className="notice notice--error">{error}</div> : null}

        <div className="field">
          <label htmlFor="email">Email</label>
          <input
            id="email"
            type="email"
            value={email}
            autoComplete="username"
            onChange={(event) => setEmail(event.target.value)}
            required
          />
        </div>

        <div className="field">
          <label htmlFor="password">Password</label>
          <input
            id="password"
            type="password"
            value={password}
            autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
            onChange={(event) => setPassword(event.target.value)}
            required
            minLength={mode === 'register' ? 12 : undefined}
          />
          {mode === 'register' ? (
            <div className="hint">At least 12 characters.</div>
          ) : null}
        </div>

        <button className="btn btn--primary btn--block" type="submit" disabled={busy}>
          {busy ? 'Working…' : mode === 'login' ? 'Sign in' : 'Create account'}
        </button>

        <button
          type="button"
          className="btn btn--ghost btn--block"
          style={{ marginTop: 8 }}
          onClick={() => {
            setMode(mode === 'login' ? 'register' : 'login');
            setError(null);
          }}
        >
          {mode === 'login' ? 'Need an account? Register' : 'Have an account? Sign in'}
        </button>
      </form>
    </div>
  );
}
