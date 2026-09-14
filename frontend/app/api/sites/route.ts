import { NextResponse } from 'next/server'
import * as Sentry from '@sentry/nextjs'

const API_BASE =
  process.env.API_BASE_URL || process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000'

export async function GET(request: Request) {
  const incoming = new URL(request.url)
  const upstream = new URL('/sites', API_BASE)
  incoming.searchParams.forEach((value, key) => {
    upstream.searchParams.set(key, value)
  })

  try {
    const response = await fetch(upstream, {
      cache: 'no-store',
      headers: { Accept: 'application/json' },
    })
    if (!response.ok) {
      Sentry.captureMessage(`Upstream /sites returned HTTP ${response.status}`, {
        level: 'warning',
        extra: { status: response.status, url: upstream.toString() },
      })
    }
    const body = await response.text()
    return new NextResponse(body, {
      status: response.status,
      headers: {
        'Content-Type': response.headers.get('Content-Type') || 'application/json',
      },
    })
  } catch (error) {
    Sentry.captureException(error, {
      extra: { upstream: upstream.toString() },
    })
    const message = error instanceof Error ? error.message : 'Unknown error'
    return NextResponse.json(
      {
        detail: `Could not reach File 43 GET /sites at ${upstream.origin}. ${message}`,
        count: 0,
        filters_applied: {},
        sites: [],
      },
      { status: 503 },
    )
  }
}

