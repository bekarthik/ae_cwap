'use client';

import { AuthGate } from '@/components/AuthGate';
import { Builder } from '@/components/Builder';

export default function Home() {
  return (
    <AuthGate>
      {(session, signOut) => <Builder session={session} onSignOut={signOut} />}
    </AuthGate>
  );
}
