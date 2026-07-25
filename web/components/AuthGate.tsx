'use client';

import { useEffect, useState } from 'react';

import {
  ApiError,
  api,
  clearSession,
  loadSession,
  setSessionExpiredHandler,
  storeSession,
} from '@/lib/api';
import type { Session } from '@/lib/types';

interface Props {
  children: (session: Session, signOut: () => void) => React.ReactNode;
}

export function AuthGate({ children }: Props) {
  const [session, setSession] = useState<Session | null>(null);
  const [ready, setReady] = useState(false);
  const [expired, setExpired] = useState(false);

  useEffect(() => {
    setSession(loadSession());
    setReady(true);
  }, []);

  // The API module drops a session the gateway has rejected; this is how that
  // reaches the screen. Without it the stored token is gone but the user is
  // still looking at a canvas where nothing works.
  useEffect(() => {
    setSessionExpiredHandler(() => {
      setSession(null);
      setExpired(true);
    });
    return () => setSessionExpiredHandler(null);
  }, []);

  if (!ready) return null;

  if (!session) {
    return <SignIn onSignedIn={setSession} expired={expired} />;
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

function SignIn({
  onSignedIn,
  expired = false,
}: {
  onSignedIn: (session: Session) => void;
  expired?: boolean;
}) {
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

        {expired && !error ? (
          <div className="notice notice--warn">
            Your session is no longer valid — sign in again. If this deployment&apos;s
            database was recreated, your old account went with it.
          </div>
        ) : null}
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
