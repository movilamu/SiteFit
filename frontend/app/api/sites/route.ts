import { NextResponse } from 'next/server'
import * as Sentry from '@sentry/nextjs'
import { DEMO_SITES_RESPONSE } from '@/lib/demo-sites'

const API_BASE =
  process.env.API_BASE_URL || process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000'

// Render's free tier spins the backend down after inactivity, and a cold
// start can take 30-50s. For a demo we'd rather fall back to bundled sample
// data quickly than leave the UI stuck on "Loading candidate sites...".
const UPSTREAM_TIMEOUT_MS = 6000

function demoResponse() {
  return NextResponse.json(DEMO_SITES_RESPONSE, {
    status: 200,
    headers: { 'X-Data-Source': 'demo-fallback' },
  })
}

export async function GET(request: Request) {
  const incoming = new URL(request.url)
  const upstream = new URL('/sites', API_BASE)
  incoming.searchParams.forEach((value, key) => {
    upstream.searchParams.set(key, value)
  })

  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS)

  try {
    const response = await fetch(upstream, {
      cache: 'no-store',
      headers: { Accept: 'application/json' },
      signal: controller.signal,
    })
    clearTimeout(timeout)

    if (!response.ok) {
      Sentry.captureMessage(`Upstream /sites returned HTTP ${response.status}`, {
        level: 'warning',
        extra: { status: response.status, url: upstream.toString() },
      })
      return demoResponse()
    }

    const body = await response.text()
    let parsed: { sites?: unknown[] } | null = null
    try {
      parsed = JSON.parse(body)
    } catch {
      // not JSON, fall through to demo below
    }

    if (!parsed || !Array.isArray(parsed.sites) || parsed.sites.length === 0) {
      // Backend reachable but has no seeded data yet - still show a demo.
      return demoResponse()
    }

    return new NextResponse(body, {
      status: response.status,
      headers: {
        'Content-Type': response.headers.get('Content-Type') || 'application/json',
        'X-Data-Source': 'live',
      },
    })
  } catch (error) {
    clearTimeout(timeout)
    Sentry.captureException(error, {
      extra: { upstream: upstream.toString() },
    })
    return demoResponse()
  }
}

