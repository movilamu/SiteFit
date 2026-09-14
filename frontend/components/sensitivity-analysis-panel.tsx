'use client'

import { useEffect, useMemo, useState } from 'react'
import { Button } from '@/components/ui/button'
import { apiUrl, readApiError } from '@/lib/api'
import { weightsQueryParams } from '@/lib/weights'

// ---------------------------------------------------------------------------
// Types (mirroring File 41's /sensitivity/* response shapes)
// ---------------------------------------------------------------------------

type ThresholdRow = {
  criterion: string
  current_weight: number
  current_top_site: string
  direction: 'increase' | 'decrease' | null
  threshold_weight: number | null
  delta: number | null
  new_top_site: string | null
  method: string
}

type StabilityRow = {
  site_id: string
  current_rank: number
  stability_percentage: number
  stability_criterion: 'exact_rank' | 'top_N' | string
  top_n?: number
}

type NearTieCluster = {
  cluster_rank: number
  site_ids: string[]
  scores: number[]
  score_range: number
}

type CrossMethodAgreement = {
  ranks: Record<string, number[]>
  pairwise_spearman: Record<string, number>
  overall_agreement: number
  flagged_sites: {
    site_id: string
    ranks: Record<string, number>
    rank_spread: number
  }[]
}

type SensitivityData = {
  thresholds: ThresholdRow[]
  stability: StabilityRow[]
  nearTies: NearTieCluster[]
  crossMethod: CrossMethodAgreement
}

type SensitivityAnalysisPanelProps = {
  weights: Record<string, number>
  method?: string
  apiBaseUrl?: string
  /** Optional site_id -> display name lookup, for friendlier labels than raw ids. */
  siteNames?: Record<string, string>
  className?: string
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

function formatCriterionName(value: string) {
  return value
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase())
}

function formatWeight(value: number | null) {
  return value === null || !Number.isFinite(value) ? '—' : value.toFixed(3)
}

function formatPercent(value: number) {
  return Number.isFinite(value) ? `${value.toFixed(0)}%` : '—'
}

function formatCorrelation(value: number) {
  return Number.isFinite(value) ? value.toFixed(2) : '—'
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function SensitivityAnalysisPanel({
  weights,
  method = 'weighted_sum',
  apiBaseUrl = '',
  siteNames = {},
  className = '',
}: SensitivityAnalysisPanelProps) {
  const [data, setData] = useState<SensitivityData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [retry, setRetry] = useState(0)

  const siteLabel = (siteId: string) => siteNames[siteId] ?? siteId

  useEffect(() => {
    const controller = new AbortController()
    const baseParams = weightsQueryParams(weights, method)

    async function fetchJson(path: string, extra?: Record<string, string>) {
      const params = new URLSearchParams(baseParams)
      if (extra) {
        for (const [key, value] of Object.entries(extra)) params.set(key, value)
      }
      const url = apiBaseUrl
        ? `${apiBaseUrl.replace(/\/+$/, '')}${path}?${params.toString()}`
        : apiUrl(`${path}?${params.toString()}`)
      const response = await fetch(url, {
        method: 'GET',
        headers: { Accept: 'application/json' },
        signal: controller.signal,
      })
      if (!response.ok) {
        throw new Error(await readApiError(response, `Request to ${path} failed (${response.status})`))
      }
      return response.json()
    }

    async function loadAll() {
      setLoading(true)
      setError(null)
      try {
        const [thresholds, stability, nearTies, crossMethod] = await Promise.all([
          fetchJson('/sensitivity/thresholds') as Promise<ThresholdRow[]>,
          fetchJson('/sensitivity/stability') as Promise<StabilityRow[]>,
          fetchJson('/sensitivity/near-ties') as Promise<NearTieCluster[]>,
          fetchJson('/sensitivity/cross-method') as Promise<CrossMethodAgreement>,
        ])
        if (!controller.signal.aborted) {
          setData({ thresholds, stability, nearTies, crossMethod })
        }
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') return
        setData(null)
        setError(err instanceof Error ? err.message : 'Unable to load sensitivity analysis.')
      } finally {
        if (!controller.signal.aborted) setLoading(false)
      }
    }

    loadAll()
    return () => controller.abort()
  }, [apiBaseUrl, method, retry, weights])

  return (
    <section className={`rounded-xl border border-border bg-background p-4 ${className}`}>
      <div className="mb-4">
        <h2 className="text-sm font-semibold text-foreground">Sensitivity analysis</h2>
        <p className="mt-1 text-xs text-muted-foreground">
          How much the current ranking depends on the chosen weights and aggregation method.
        </p>
      </div>

      {loading && !data && (
        <div className="space-y-3" aria-live="polite" aria-busy="true">
          {[0, 1, 2].map((item) => (
            <div key={item} className="animate-pulse">
              <div className="mb-2 h-4 w-40 rounded bg-muted" />
              <div className="h-16 rounded bg-muted" />
            </div>
          ))}
        </div>
      )}

      {!loading && error && (
        <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-4" role="alert">
          <p className="text-sm font-medium text-destructive">Couldn&apos;t load the sensitivity analysis.</p>
          <p className="mt-1 text-xs text-destructive/80">{error}</p>
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="mt-3"
            onClick={() => setRetry((value) => value + 1)}
          >
            Try again
          </Button>
        </div>
      )}

      {!loading && !error && data && (
        <div className="space-y-6">
          <ThresholdsSection rows={data.thresholds} siteLabel={siteLabel} />
          <StabilitySection rows={data.stability} siteLabel={siteLabel} />
          <NearTiesSection clusters={data.nearTies} siteLabel={siteLabel} />
          <CrossMethodSection agreement={data.crossMethod} siteLabel={siteLabel} />
        </div>
      )}
    </section>
  )
}

// ---------------------------------------------------------------------------
// 1. Critical weight thresholds
// ---------------------------------------------------------------------------

function ThresholdsSection({
  rows,
  siteLabel,
}: {
  rows: ThresholdRow[]
  siteLabel: (id: string) => string
}) {
  return (
    <div>
      <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        Critical weight thresholds
      </h3>
      {rows.length === 0 ? (
        <p className="mt-2 text-sm text-muted-foreground">No threshold data available.</p>
      ) : (
        <ul className="mt-2 space-y-2">
          {rows.map((row) => (
            <li
              key={row.criterion}
              className="rounded-lg border border-border bg-muted/20 px-3 py-2 text-sm"
            >
              <div className="font-medium text-foreground">{formatCriterionName(row.criterion)}</div>
              {row.threshold_weight === null || row.delta === null ? (
                <p className="mt-0.5 text-xs text-muted-foreground">
                  No weight change within range flips the top site ({siteLabel(row.current_top_site)}).
                </p>
              ) : (
                <p className="mt-0.5 text-xs text-muted-foreground">
                  Can move{' '}
                  <span className="font-medium text-foreground">
                    {row.delta >= 0 ? '+' : ''}
                    {formatWeight(row.delta)}
                  </span>{' '}
                  (to {formatWeight(row.threshold_weight)}) before the top site changes from{' '}
                  <span className="font-medium text-foreground">{siteLabel(row.current_top_site)}</span> to{' '}
                  <span className="font-medium text-foreground">
                    {row.new_top_site ? siteLabel(row.new_top_site) : '—'}
                  </span>
                  .
                </p>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// 2. Rank stability
// ---------------------------------------------------------------------------

function StabilitySection({
  rows,
  siteLabel,
}: {
  rows: StabilityRow[]
  siteLabel: (id: string) => string
}) {
  const sorted = useMemo(() => [...rows].sort((a, b) => a.current_rank - b.current_rank), [rows])
  const topSite = sorted[0]

  return (
    <div>
      <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        Rank stability
      </h3>
      {topSite && (
        <p className="mt-2 text-sm text-foreground">
          <span className="font-medium">{siteLabel(topSite.site_id)}</span> holds rank{' '}
          {topSite.current_rank} in{' '}
          <span className="font-medium">{formatPercent(topSite.stability_percentage)}</span> of sampled
          weight variations.
        </p>
      )}
      {sorted.length === 0 ? (
        <p className="mt-2 text-sm text-muted-foreground">No stability data available.</p>
      ) : (
        <ul className="mt-3 space-y-1.5">
          {sorted.map((row) => (
            <li key={row.site_id} className="flex items-center gap-3">
              <span className="w-6 shrink-0 text-right text-xs tabular-nums text-muted-foreground">
                #{row.current_rank}
              </span>
              <span className="min-w-0 flex-1 truncate text-sm text-foreground">
                {siteLabel(row.site_id)}
              </span>
              <div className="h-2 w-24 shrink-0 overflow-hidden rounded-full bg-muted">
                <div
                  className="h-full rounded-full bg-primary"
                  style={{ width: `${Math.min(100, Math.max(0, row.stability_percentage))}%` }}
                />
              </div>
              <span className="w-10 shrink-0 text-right text-xs tabular-nums text-muted-foreground">
                {formatPercent(row.stability_percentage)}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// 3. Near-tie groups
// ---------------------------------------------------------------------------

function NearTiesSection({
  clusters,
  siteLabel,
}: {
  clusters: NearTieCluster[]
  siteLabel: (id: string) => string
}) {
  return (
    <div>
      <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        Near-tie groups
      </h3>
      <p className="mt-1 text-xs text-muted-foreground">
        Sites within a cluster are effectively tied and should not be read as strictly ordered.
      </p>
      {clusters.length === 0 ? (
        <p className="mt-2 text-sm text-muted-foreground">No near-tie data available.</p>
      ) : (
        <ul className="mt-2 space-y-2">
          {clusters.map((cluster) => {
            const isTie = cluster.site_ids.length > 1
            return (
              <li
                key={cluster.cluster_rank}
                className={[
                  'rounded-lg border px-3 py-2',
                  isTie ? 'border-amber-300/60 bg-amber-500/10' : 'border-border bg-muted/20',
                ].join(' ')}
              >
                <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
                  <span>Rank {cluster.cluster_rank}</span>
                  {isTie && (
                    <span className="rounded-full bg-amber-500/20 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-amber-700 dark:text-amber-400">
                      Tied &middot; {cluster.site_ids.length} sites
                    </span>
                  )}
                </div>
                <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-sm text-foreground">
                  {cluster.site_ids.map((siteId, idx) => (
                    <span key={siteId}>
                      {siteLabel(siteId)}
                      <span className="ml-1 font-mono text-xs text-muted-foreground">
                        ({cluster.scores[idx].toFixed(4)})
                      </span>
                    </span>
                  ))}
                </div>
                {isTie && (
                  <p className="mt-1 text-[11px] text-muted-foreground">
                    Score spread: {cluster.score_range.toFixed(4)}
                  </p>
                )}
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// 4. Cross-method agreement
// ---------------------------------------------------------------------------

function CrossMethodSection({
  agreement,
  siteLabel,
}: {
  agreement: CrossMethodAgreement
  siteLabel: (id: string) => string
}) {
  const pairwiseEntries = Object.entries(agreement.pairwise_spearman ?? {})

  return (
    <div>
      <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        Cross-method agreement
      </h3>
      <p className="mt-2 text-sm text-foreground">
        Overall agreement across methods:{' '}
        <span className="font-medium">{formatCorrelation(agreement.overall_agreement)}</span>{' '}
        <span className="text-xs text-muted-foreground">(Spearman correlation, 1.0 = identical rankings)</span>
      </p>

      {pairwiseEntries.length > 0 && (
        <ul className="mt-2 flex flex-wrap gap-2">
          {pairwiseEntries.map(([pair, corr]) => (
            <li
              key={pair}
              className="rounded-full border border-border bg-muted/20 px-2.5 py-1 text-xs text-muted-foreground"
            >
              {pair.split('__vs__').map(formatCriterionName).join(' vs ')}:{' '}
              <span className="font-medium text-foreground">{formatCorrelation(corr)}</span>
            </li>
          ))}
        </ul>
      )}

      <div className="mt-3">
        {agreement.flagged_sites.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            No sites are sensitive to the choice of aggregation method.
          </p>
        ) : (
          <>
            <p className="text-xs font-medium text-foreground">
              {agreement.flagged_sites.length} site{agreement.flagged_sites.length === 1 ? '' : 's'} sensitive
              to the aggregation method:
            </p>
            <ul className="mt-2 space-y-2">
              {agreement.flagged_sites.map((site) => (
                <li
                  key={site.site_id}
                  className="rounded-lg border border-destructive/30 bg-destructive/5 px-3 py-2 text-sm"
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-medium text-foreground">{siteLabel(site.site_id)}</span>
                    <span className="text-xs text-muted-foreground">
                      rank spread: <span className="font-medium text-foreground">{site.rank_spread}</span>
                    </span>
                  </div>
                  <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-xs text-muted-foreground">
                    {Object.entries(site.ranks).map(([methodName, rank]) => (
                      <span key={methodName}>
                        {formatCriterionName(methodName)}: <span className="text-foreground">#{rank}</span>
                      </span>
                    ))}
                  </div>
                </li>
              ))}
            </ul>
          </>
        )}
      </div>
    </div>
  )
}
