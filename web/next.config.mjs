/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // The gateway URL is read at build time so the browser bundle has a single,
  // explicit place it talks to. Nothing else in the app hardcodes a host.
  env: {
    NEXT_PUBLIC_API_BASE: process.env.NEXT_PUBLIC_API_BASE ?? 'http://localhost:8000',
  },
};

export default nextConfig;
