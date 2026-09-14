'use client'

import { useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { apiUrl, readApiError } from '@/lib/api'
import { scoreRequestBody, weightsQueryParams } from '@/lib/weights'
import { ConfidenceIndicator } from './confidence-indicator'

type SiteComparisonProps = {
  sites: CandidateSite[]
  weights: Record<string, number>
  method?: string
  selectedSiteIds?: string[]
  onSelectionChange?: (siteIds: string[]) => void
}

type CandidateSite = {
  id?: string
  site_id?: string
  name?: string
  site_name?: string
  [key: string]: unknown
}

type CriterionDecomposition = {
  criterion: string
  normalized_value: number
  weight: number
  contribution: number
  data_confidence?: number | null
  confidence_explanation?: string | null
}

type DecompositionResponse = {
  site_id: string
  method: string
  total_score: number
  criteria: CriterionDecomposition[]
}

type RankedSite = {
  rank: number
  site_id: string | number
  score: number
}

function getSiteId(site: CandidateSite): string {
  return String(site.id ?? site.site_id ?? '')
}

function getSiteName(site: CandidateSite): string {
  return String(site.name ?? site.site_name ?? site.id ?? site.site_id ?? 'Unnamed site')
}

function formatNumber(value: number): string {
  return value.toFixed(4)
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}

export default function SiteComparison({
  sites,
  weights,
  method = 'weighted_sum',
  selectedSiteIds: controlledSelectedIds,
  onSelectionChange,
}: SiteComparisonProps) {
  const [internalSelectedIds, setInternalSelectedIds] = useState<string[]>([])
  const [data, setData] = useState<Record<string, DecompositionResponse>>({})
  const [ranks, setRanks] = useState<Record<string, RankedSite>>({})
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [retry, setRetry] = useState(0)

  const selectedIds = controlledSelectedIds ?? internalSelectedIds

  const selectedSites = useMemo(
    () =>
      selectedIds
        .map((id) => sites.find((site) => getSiteId(site) === id))
        .filter((site): site is CandidateSite => Boolean(site)),
    [selectedIds, sites],
  )

  function updateSelection(nextIds: string[]) {
    if (controlledSelectedIds === undefined) {
      setInternalSelectedIds(nextIds)
    }
    onSelectionChange?.(nextIds)
  }

  function toggleSite(siteId: string) {
    if (selectedIds.includes(siteId)) {
      updateSelection(selectedIds.filter((id) => id !== siteId))
      return
    }
    if (selectedIds.length >= 4) return
    updateSelection([...selectedIds, siteId])
  }

  useEffect(() => {
    if (selectedIds.length === 0) {
      setData({})
      setRanks({})
      setError(null)
      return
    }

    const controller = new AbortController()

    async function loadComparisons() {
      setLoading(true)
      setError(null)

      try {
        const query = weightsQueryParams(weights, method)
        const results = await Promise.all(
          selectedIds.map(async (id) => {
            const response = await fetch(
              apiUrl(`/sites/${encodeURIComponent(id)}/decomposition?${query.toString()}`),
              { signal: controller.signal, headers: { Accept: 'application/json' } },
            )
            if (!response.ok) {
              throw new Error(await readApiError(response, `Unable to load decomposition for site ${id}`))
            }
            return (await response.json()) as DecompositionResponse
          }),
        )

        const scoreResponse = await fetch(apiUrl('/score'), {
          method: 'POST',
          signal: controller.signal,
          headers: {
            'Content-Type': 'application/json',
            Accept: 'application/json',
          },
          body: JSON.stringify(scoreRequestBody(weights, method)),
        })
        if (!scoreResponse.ok) {
          throw new Error(await readApiError(scoreResponse, 'Unable to load POST /score rankings'))
        }
        const scored = (await scoreResponse.json()) as { sites?: RankedSite[] }

        if (controller.signal.aborted) return

        const nextData: Record<string, DecompositionResponse> = {}
        for (const result of results) {
          nextData[String(result.site_id)] = result
        }
        const nextRanks: Record<string, RankedSite> = {}
        for (const row of scored.sites ?? []) {
          nextRanks[String(row.site_id)] = row
        }
        setData(nextData)
        setRanks(nextRanks)
      } catch (err) {
        if (controller.signal.aborted) return
        if (err instanceof DOMException && err.name === 'AbortError') return
        setError(err instanceof Error ? err.message : 'Unable to load site comparison.')
      } finally {
        if (!controller.signal.aborted) setLoading(false)
      }
    }

    loadComparisons()
    return () => controller.abort()
  }, [selectedIds, weights, method, retry])

  const criteria = useMemo(() => {
    const seen = new Set<string>()
    for (const siteId of selectedIds) {
      for (const criterion of data[siteId]?.criteria ?? []) {
        seen.add(criterion.criterion)
      }
    }
    return Array.from(seen)
  }, [selectedIds, data])

  if (sites.length === 0) {
    return (
      <section className="rounded-xl border bg-background p-5">
        <h2 className="text-base font-semibold">Site comparison</h2>
        <p className="mt-1 text-sm text-muted-foreground">No shortlisted sites are available.</p>
      </section>
    )
  }

  return (
    <section className="rounded-xl border bg-background">
      <div className="border-b p-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-base font-semibold">Compare shortlisted sites</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Select 2–4 sites to compare File 44 ranks and File 45 criterion contributions.
            </p>
          </div>
          {selectedIds.length > 0 && (
            <Button variant="outline" size="sm" onClick={() => updateSelection([])}>
              Clear
            </Button>
          )}
        </div>

        <div className="mt-4 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
          {sites.map((site) => {
            const id = getSiteId(site)
            const checked = selectedIds.includes(id)
            const disabled = !checked && selectedIds.length >= 4
            return (
              <label
                key={id}
                className={[
                  'flex cursor-pointer items-center gap-2 rounded-lg border p-3 text-sm transition-colors',
                  checked ? 'border-primary bg-primary/5' : 'hover:bg-muted/50',
                  disabled ? 'cursor-not-allowed opacity-50' : '',
                ].join(' ')}
              >
                <input
                  type="checkbox"
                  checked={checked}
                  disabled={disabled}
                  onChange={() => toggleSite(id)}
                  className="size-4"
                />
                <span className="min-w-0 truncate font-medium">{getSiteName(site)}</span>
              </label>
            )
          })}
        </div>

        <p className="mt-2 text-xs text-muted-foreground">
          {selectedIds.length} of 4 sites selected
          {selectedIds.length < 2 ? ' — select at least 2 to compare.' : ''}
        </p>
      </div>

      {selectedIds.length < 2 ? (
        <div className="p-8 text-center text-sm text-muted-foreground">
          Select at least two shortlisted sites to see the comparison.
        </div>
      ) : loading && Object.keys(data).length === 0 ? (
        <div className="p-8 text-center text-sm text-muted-foreground" aria-busy="true">
          Loading comparison…
        </div>
      ) : error ? (
        <div className="p-5">
          <div className="rounded-lg border border-destructive/30 bg-destructive/5 p-4" role="alert">
            <p className="text-sm text-destructive">{error}</p>
            <Button className="mt-3" variant="outline" size="sm" onClick={() => setRetry((value) => value + 1)}>
              Retry
            </Button>
          </div>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[760px] border-collapse text-sm">
            <thead>
              <tr className="border-b bg-muted/30">
                <th className="sticky left-0 z-10 min-w-[220px] bg-muted/30 px-5 py-3 text-left font-medium">
                  Criterion
                </th>
                {selectedSites.map((site) => {
                  const id = getSiteId(site)
                  const result = data[id]
                  const ranked = ranks[id]
                  return (
                    <th key={id} className="min-w-[180px] px-4 py-3 text-left align-top">
                      <div className="font-semibold">{getSiteName(site)}</div>
                      {ranked && (
                        <div className="mt-1 text-xs font-normal text-muted-foreground">
                          Rank #{ranked.rank} · score {formatNumber(ranked.score)}
                        </div>
                      )}
                      {result && (
                        <div className="mt-1 text-xs font-normal text-muted-foreground">
                          Decomposition total:{' '}
                          <span className="font-medium text-foreground">{formatNumber(result.total_score)}</span>
                        </div>
                      )}
                    </th>
                  )
                })}
              </tr>
            </thead>
            <tbody>
              {criteria.map((criterion) => (
                <tr key={criterion} className="border-b last:border-0">
                  <th className="sticky left-0 bg-background px-5 py-4 text-left font-medium">{criterion}</th>
                  {selectedIds.map((id) => {
                    const criterionData = data[id]?.criteria.find((item) => item.criterion === criterion)
                    if (!criterionData) {
                      return (
                        <td key={id} className="px-4 py-4 text-muted-foreground">
                          —
                        </td>
                      )
                    }
                    return (
                      <td key={id} className="px-4 py-4 align-top">
                        <div className="space-y-2">
                          <div>
                            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                              Normalized value
                              <ConfidenceIndicator
                                score={criterionData.data_confidence}
                                explanation={criterionData.confidence_explanation}
                                criterionLabel={criterion}
                              />
                            </div>
                            <div className="font-medium">
                              {formatNumber(criterionData.normalized_value)}
                              <span className="ml-2 text-xs text-muted-foreground">
                                ({formatPercent(criterionData.normalized_value)})
                              </span>
                            </div>
                          </div>
                          <div>
                            <div className="text-xs text-muted-foreground">Contribution</div>
                            <div className="font-semibold">{formatNumber(criterionData.contribution)}</div>
                          </div>
                          <div className="h-2 overflow-hidden rounded-full bg-muted">
                            <div
                              className="h-full rounded-full bg-primary transition-all"
                              style={{
                                width: `${Math.min(100, Math.max(0, criterionData.contribution * 100))}%`,
                              }}
                            />
                          </div>
                        </div>
                      </td>
                    )
                  })}
                </tr>
              ))}
              <tr className="border-t bg-muted/20">
                <th className="sticky left-0 bg-muted/20 px-5 py-4 text-left font-semibold">Total score</th>
                {selectedIds.map((id: string) => (
                  <td key={id} className="px-4 py-4 text-lg font-bold">
                    {data[id] ? formatNumber(data[id].total_score) : '—'}
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}
