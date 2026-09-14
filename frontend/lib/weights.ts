/**
 * Weight-passing convention used by scoring surfaces:
 *
 * File 44 POST /score          → JSON body { weights, method }
 * File 45 GET  /decomposition   → query     weights=<json>&method=
 * File 41 GET  /sensitivity/*   → query     weights=<json>&method=
 * File 46/47 scenarios/export   → JSON body { weight_set, aggregation_method }
 *
 * `weights` / `weight_set` is always Record<criterion, number> summing to ~1.0.
 */

export const DEFAULT_METHOD = 'weighted_sum'

export const DEFAULT_WEIGHTS: Record<string, number> = {
  population: 0.125,
  demographics: 0.125,
  spending: 0.125,
  accessibility: 0.125,
  competition: 0.125,
  cannibalisation: 0.125,
  site: 0.125,
  risk: 0.125,
}

export type ScoreRequestBody = {
  weights: Record<string, number>
  method: string
}

/** File 44 / File 54 scoring payload. */
export function scoreRequestBody(
  weights: Record<string, number>,
  method: string = DEFAULT_METHOD,
): ScoreRequestBody {
  return { weights: normalizeWeights(weights), method }
}

/** File 45 decomposition and File 41 / File 56 sensitivity query string. */
export function weightsQueryParams(
  weights: Record<string, number>,
  method: string = DEFAULT_METHOD,
): URLSearchParams {
  return new URLSearchParams({
    weights: JSON.stringify(normalizeWeights(weights)),
    method,
  })
}

export function weightSum(weights: Record<string, number>): number {
  return Object.values(weights).reduce((sum, value) => sum + (Number(value) || 0), 0)
}

export function normalizeWeights(weights: Record<string, number>): Record<string, number> {
  const total = weightSum(weights)
  if (total <= 0) return weights
  const next: Record<string, number> = {}
  for (const [key, value] of Object.entries(weights)) {
    next[key] = (Number(value) || 0) / total
  }
  return next
}
