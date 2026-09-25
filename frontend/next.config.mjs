import { withSentryConfig } from '@sentry/nextjs'

const API_BASE = process.env.API_BASE_URL || process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000'

/** @type {import('next').NextConfig} */
const nextConfig = {
  turbopack: {},
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  async rewrites() {
    return [
      { source: '/score', destination: `${API_BASE}/score` },
      { source: '/export', destination: `${API_BASE}/export` },
      { source: '/sites', destination: `${API_BASE}/sites` },
      { source: '/sites/:path*', destination: `${API_BASE}/sites/:path*` },
      { source: '/scenarios', destination: `${API_BASE}/scenarios` },
      { source: '/scenarios/:path*', destination: `${API_BASE}/scenarios/:path*` },
      { source: '/sensitivity/:path*', destination: `${API_BASE}/sensitivity/:path*` },
      { source: '/elicitation/:path*', destination: `${API_BASE}/elicitation/:path*` },
    ]
  },
}

export default withSentryConfig(nextConfig, {
  // Sentry configuration options
  org: process.env.SENTRY_ORG,
  project: process.env.SENTRY_PROJECT,
  silent: !process.env.CI,
  widenClientFileUpload: true,
  hideSourceMaps: true,
  disableLogger: true,
})

