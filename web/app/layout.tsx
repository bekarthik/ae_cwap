import type { Metadata } from 'next';

import '@xyflow/react/dist/style.css';
import './globals.css';

export const metadata: Metadata = {
  title: 'AI Cognitive Workflow Platform',
  description:
    'Design multi-step AI agent workflows visually — no code, no API calls to write.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
