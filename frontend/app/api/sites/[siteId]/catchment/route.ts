import { NextResponse } from 'next/server'
import * as Sentry from '@sentry/nextjs'

const API_BASE =
  process.env.API_BASE_URL || process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000'

function asGeometry(value: unknown): GeoJSON.Geometry | null {
  if (!value || typeof value !== 'object') return null
  const record = value as Record<string, unknown>
  if (record.type === 'Feature') return asGeometry(record.geometry)
  if (record.type === 'FeatureCollection') {
    const features = Array.isArray(record.features) ? record.features : []
    for (const feature of features) {
      const geometry = asGeometry(feature)
      if (geometry) return geometry
    }
    return null
  }
  if (record.type === 'Polygon' || record.type === 'MultiPolygon') {
    return record as unknown as GeoJSON.Geometry
  }
  if ('geometry' in record) return asGeometry(record.geometry)
  if ('isochrone_10min' in record) return asGeometry(record.isochrone_10min)
  if ('isochrone_15min' in record) return asGeometry(record.isochrone_15min)
  return null
}

async function tryFetch(url: string): Promise<unknown | null> {
  try {
    const response = await fetch(url, {
      cache: 'no-store',
      headers: { Accept: 'application/json' },
    })
    if (!response.ok) return null
    return await response.json()
  } catch (error) {
    Sentry.captureException(error, { extra: { url } })
    return null
  }
}


export async function GET(
  _request: Request,
  context: { params: Promise<{ siteId: string }> },
) {
  const { siteId } = await context.params
  const encoded = encodeURIComponent(siteId)
  const candidates = [
    `${API_BASE}/sites/${encoded}/catchment`,
    `${API_BASE}/catchments?candidate_site_id=${encoded}`,
    `${API_BASE}/candidate_catchments?candidate_site_id=eq.${encoded}`,
  ]

  for (const url of candidates) {
    const payload = await tryFetch(url)
    if (!payload) continue
    const row = Array.isArray(payload) ? payload[0] : payload
    const geometry = asGeometry(row)
    if (geometry) {
      return NextResponse.json({ geometry, source: 'api' })
    }
  }

  return NextResponse.json({
    geometry: null,
    source: null,
    enhancement:
      'Catchment overlay is available when File 22 isochrones (isochrone_10min / isochrone_15min) are joined onto the site or exposed by a catchment API.',
  })
}
