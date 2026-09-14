'use client'

import { useEffect, useMemo, useState } from 'react'
import { Button } from '@/components/ui/button'
import { ConfidenceIndicator } from './confidence-indicator'
import { apiUrl, readApiError } from '@/lib/api'
import { weightsQueryParams } from '@/lib/weights'

export type CriterionDecomposition = {
  criterion: string
  normalized_value: number
  weight: number
  contribution: number
  /** Confidence in [0, 1] for this criterion's underlying value, e.g. from
   *  File 16's areal-interpolation confidence score. Optional since some
   *  criteria are sourced directly (no interpolation) and may omit it. */
  data_confidence?: number | null
  /** Optional server-provided rationale for the confidence score. */
  confidence_explanation?: string | null
}

export type SiteDecompositionResponse = {
  site_id: string
  method: string
  total_score: number
  criteria: CriterionDecomposition[]
}

type ScoreDecompositionPanelProps = {
  siteId: string | null
  weights: Record<string, number>
  method?: string
  apiBaseUrl?: string
  className?: string
}

function formatCriterionName(value: string) {
  return value
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase())
}

function formatNumber(value: number) {
  return Number.isFinite(value) ? value.toFixed(4) : '—'
}

export default function ScoreDecompositionPanel({
  siteId,
  weights,
  method = 'weighted_sum',
  apiBaseUrl = '',
  className = '',
}: ScoreDecompositionPanelProps) {
  const [data, setData] = useState<SiteDecompositionResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [retry, setRetry] = useState(0)

  useEffect(() => {
    if (!siteId) {
      setData(null)
      setError(null)
      setLoading(false)
      return
    }

    const controller = new AbortController()
    const params = weightsQueryParams(weights, method)

    async function loadDecomposition() {
      setLoading(true)
      setError(null)

      try {
        const url = apiBaseUrl
          ? `${apiBaseUrl.replace(/\/+$/, '')}/sites/${encodeURIComponent(siteId!)}/decomposition?${params.toString()}`
          : apiUrl(`/sites/${encodeURIComponent(siteId!)}/decomposition?${params.toString()}`)
        const response = await fetch(url, {
          method: 'GET',
          headers: { Accept: 'application/json' },
          signal: controller.signal,
        })

        if (!response.ok) {
          throw new Error(await readApiError(response, `Unable to load score decomposition (${response.status})`))
        }

        const decomposition = (await response.json()) as SiteDecompositionResponse
        setData(decomposition)
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') return
        setData(null)
        setError(err instanceof Error ? err.message : 'Unable to load score decomposition.')
      } finally {
        if (!controller.signal.aborted) setLoading(false)
      }
    }

    loadDecomposition()
    return () => controller.abort()
  }, [apiBaseUrl, method, retry, siteId, weights])

  const maxContribution = Math.max(
    ...(data?.criteria.map((criterion) => Math.max(criterion.contribution, 0)) ?? [0]),
    0,
  )

  return (
    <section className={`rounded-xl border border-border bg-background p-4 ${className}`}>
      <div className="mb-4 flex items-start justify-between gap-4">
        <div>
          <h2 className="text-sm font-semibold text-foreground">Score decomposition</h2>
          <p className="mt-1 text-xs text-muted-foreground">
            Each bar shows how much that criterion contributes to this site&apos;s score.
          </p>
        </div>
        {data && (
          <div className="shrink-0 text-right">
            <div className="text-xs text-muted-foreground">Total score</div>
            <div className="text-lg font-semibold tabular-nums text-foreground">
              {formatNumber(data.total_score)}
            </div>
          </div>
        )}
      </div>

      {!siteId && (
        <div className="rounded-lg border border-dashed border-border px-4 py-8 text-center text-sm text-muted-foreground">
          Select a site to see why it scored the way it did.
        </div>
      )}

      {siteId && loading && (
        <div className="space-y-3" aria-live="polite" aria-busy="true">
          {[0, 1, 2, 3].map((item) => (
            <div key={item} className="animate-pulse">
              <div className="mb-2 h-4 w-32 rounded bg-muted" />
              <div className="h-7 rounded bg-muted" />
            </div>
          ))}
        </div>
      )}

      {siteId && !loading && error && (
        <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-4" role="alert">
          <p className="text-sm font-medium text-destructive">Couldn&apos;t load the decomposition.</p>
          <p className="mt-1 text-xs text-destructive/80">{error}</p>
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="mt-3"
            onClick={() => {
              setRetry((value) => value + 1)
            }}
          >
            Try again
          </Button>
        </div>
      )}

      {siteId && !loading && !error && data && (
        <div className="space-y-4" aria-label="Score contribution breakdown">
          {data.criteria.map((criterion) => {
            const contribution = Math.max(criterion.contribution, 0)
            const width = maxContribution > 0 ? (contribution / maxContribution) * 100 : 0

            return (
              <div key={criterion.criterion}>
                <div className="mb-1.5 flex items-baseline justify-between gap-3">
                  <span className="flex min-w-0 items-center gap-1.5 truncate text-sm font-medium text-foreground">
                    <span className="min-w-0 truncate">{formatCriterionName(criterion.criterion)}</span>
                    <ConfidenceIndicator
                      score={criterion.data_confidence}
                      explanation={criterion.confidence_explanation}
                      criterionLabel={formatCriterionName(criterion.criterion)}
                    />
                  </span>
                  <span className="shrink-0 font-mono text-xs tabular-nums text-muted-foreground">
                    +{formatNumber(criterion.contribution)}
                  </span>
                </div>
                <div
                  className="h-7 overflow-hidden rounded-md bg-muted"
                  role="img"
                  aria-label={`${formatCriterionName(criterion.criterion)} contributes ${formatNumber(criterion.contribution)} to the total score`}
                >
                  <div
                    className="h-full rounded-md bg-primary transition-[width] duration-500 ease-out"
                    style={{ width: `${width}%`, minWidth: contribution > 0 ? '2px' : undefined }}
                  />
                </div>
              </div>
            )
          })}

          <div className="mt-5 grid grid-cols-2 gap-3 border-t border-border pt-4 text-xs">
            <div>
              <div className="text-muted-foreground">Aggregation</div>
              <div className="mt-0.5 font-medium text-foreground">{formatCriterionName(data.method)}</div>
            </div>
            <div className="text-right">
              <div className="text-muted-foreground">Criteria</div>
              <div className="mt-0.5 font-medium tabular-nums text-foreground">{data.criteria.length}</div>
            </div>
          </div>
        </div>
      )}
    </section>
  )
}
