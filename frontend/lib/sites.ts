import { apiUrl, readApiError } from './api'
import { scoreRequestBody } from './weights'

export type SiteCoordinates = {
  latitude: number
  longitude: number
}

export type CandidateSite = {
  site_id: string | number
  name: string | null
  coordinates: SiteCoordinates
  unit_size: number | null
  rent: number | null
  tenure: string | null
  status: string | null
  attractiveness: number | null
  attributes: Record<string, unknown>
}

export type SitesListResponse = {
  count: number
  filters_applied: Record<string, unknown>
  sites: CandidateSite[]
  excluded?: unknown[]
}

export type CatchmentResponse = {
  geometry: GeoJSON.Geometry | null
  source: 'site-attributes' | 'api' | null
  enhancement?: string
}

function siteId(site: Pick<CandidateSite, 'site_id'>): string {
  return String(site.site_id)
}

export { siteId }

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
  return null
}

/** File 22 stores drive-time isochrones on candidate_catchments; they may also
 *  appear on a site's extra attributes if the API joined them. */
export function extractCatchmentFromSite(site: CandidateSite): GeoJSON.Geometry | null {
  const attrs = site.attributes ?? {}
  const candidates = [
    attrs.isochrone_10min,
    attrs.isochrone_15min,
    attrs.catchment,
    attrs.gravity_catchment,
  ]
  for (const candidate of candidates) {
    const geometry = asGeometry(candidate)
    if (geometry) return geometry
  }
  return null
}

export async function fetchSites(query = ''): Promise<CandidateSite[]> {
  const path = query ? `/api/sites?${query}` : '/api/sites'
  const response = await fetch(path, { cache: 'no-store' })
  if (!response.ok) {
    const detail = await response.text().catch(() => '')
    throw new Error(detail || `GET /sites failed (${response.status})`)
  }
  const payload = (await response.json()) as SitesListResponse
  return Array.isArray(payload.sites) ? payload.sites : []
}

export type RankedScore = {
  rank: number
  site_id: string | number
  score: number
}

export type ScoreResponse = {
  method: string
  sites: RankedScore[]
}

/** File 44 POST /score — body is { weights, method }. */
export async function fetchScores(
  weights: Record<string, number>,
  method = 'weighted_sum',
): Promise<ScoreResponse> {
  const response = await fetch(apiUrl('/score'), {
    method: 'POST',
    cache: 'no-store',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify(scoreRequestBody(weights, method)),
  })
  if (!response.ok) {
    throw new Error(await readApiError(response, 'POST /score failed'))
  }
  return (await response.json()) as ScoreResponse
}

export async function fetchCatchment(site: CandidateSite): Promise<CatchmentResponse> {
  const fromAttributes = extractCatchmentFromSite(site)
  if (fromAttributes) {
    return { geometry: fromAttributes, source: 'site-attributes' }
  }

  const response = await fetch(`/api/sites/${encodeURIComponent(siteId(site))}/catchment`, {
    cache: 'no-store',
  })
  if (!response.ok) {
    return {
      geometry: null,
      source: null,
      enhancement:
        'Catchment overlay needs File 22 isochrones exposed on the site payload or a catchment endpoint.',
    }
  }
  const payload = (await response.json()) as CatchmentResponse
  return payload
}

export function displayName(site: CandidateSite): string {
  return site.name?.trim() || `Site ${site.site_id}`
}

export function displayPlace(site: CandidateSite): string {
  const attrs = site.attributes ?? {}
  const city =
    (typeof attrs.city === 'string' && attrs.city) ||
    (typeof attrs.locality === 'string' && attrs.locality) ||
    (typeof attrs.district === 'string' && attrs.district)
  if (city) return `${city}, Tamil Nadu`
  return 'Chennai, Tamil Nadu'
}

export function displayScore(site: CandidateSite): number | null {
  if (typeof site.attractiveness === 'number' && Number.isFinite(site.attractiveness)) {
    return site.attractiveness
  }
  const raw = site.attributes?.score
  if (typeof raw === 'number' && Number.isFinite(raw)) return raw
  return null
}
