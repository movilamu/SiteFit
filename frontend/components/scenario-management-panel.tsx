'use client'

import { useCallback, useEffect, useMemo, useState } from 'react'
import { Button } from '@/components/ui/button'
import { apiUrl, readApiError } from '@/lib/api'
import { normalizeWeights } from '@/lib/weights'

// ---------------------------------------------------------------------------
// Types matching File 46 (api/scenario_routes.py)
// ---------------------------------------------------------------------------

export interface SavedScenario {
  scenario_id: number
  name: string
  weight_set: Record<string, number>
  aggregation_method: string
  created_at: string
  updated_at: string
}

export interface ScenarioRankingItem {
  site_id: string | number
  score: number
  rank: number
}

export interface ScenarioWithRanking extends SavedScenario {
  ranking: ScenarioRankingItem[]
}

export interface ScenarioComparisonRankingRow {
  site_id: string | number
  [key: `scenario_${number}_rank`]: number | undefined
  [key: `scenario_${number}_score`]: number | undefined
}

export interface ScenarioComparisonResponse {
  scenarios: SavedScenario[]
  ranking: ScenarioComparisonRankingRow[]
}

export interface ScenarioManagementPanelProps {
  /** The current active weight set (e.g. { grid_distance: 0.4, slope: 0.6 }) */
  currentWeights: Record<string, number>
  /** The current active aggregation method (weighted_sum, weighted_product, distance_to_ideal) */
  currentMethod?: string
  /** Base URL for API requests (default: '') */
  apiBaseUrl?: string
  /** Optional lookup mapping site_id -> display name */
  siteNames?: Record<string, string>
  /** Callback triggered when a saved scenario is selected to load/apply */
  onSelectScenario?: (scenario: SavedScenario) => void
  /** Callback triggered when a new scenario is successfully saved */
  onScenarioSaved?: (scenario: SavedScenario) => void
  /** Additional custom class names */
  className?: string
}

// ---------------------------------------------------------------------------
// Helpers & Formatters
// ---------------------------------------------------------------------------

const SUPPORTED_METHODS: Record<string, string> = {
  weighted_sum: 'Weighted Sum',
  weighted_product: 'Weighted Product',
  distance_to_ideal: 'Distance to Ideal',
}

function formatMethodName(method: string): string {
  const normalized = method.trim().toLowerCase().replace(/[- ]/g, '_')
  return SUPPORTED_METHODS[normalized] ?? method.replace(/[_-]+/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}

function formatScore(value: number | undefined | null): string {
  if (value === undefined || value === null || !Number.isFinite(value)) return '—'
  return value.toFixed(4)
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}

function formatDate(dateStr: string): string {
  try {
    const d = new Date(dateStr)
    if (isNaN(d.getTime())) return dateStr
    return d.toLocaleDateString(undefined, {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  } catch {
    return dateStr
  }
}

// ---------------------------------------------------------------------------
// Self-contained SVG Icons (guarantees zero missing icon dependency errors)
// ---------------------------------------------------------------------------

function BookmarkIcon({ className = 'size-4' }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
    </svg>
  )
}

function GitCompareIcon({ className = 'size-4' }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <circle cx="18" cy="18" r="3" />
      <circle cx="6" cy="6" r="3" />
      <path strokeLinecap="round" strokeLinejoin="round" d="M13 6h3a2 2 0 012 2v7M6 9v12" />
    </svg>
  )
}

function RefreshIcon({ className = 'size-4' }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
    </svg>
  )
}

function PlusIcon({ className = 'size-4' }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
    </svg>
  )
}

function CheckIcon({ className = 'size-4' }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
    </svg>
  )
}

function AlertCircleIcon({ className = 'size-4' }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <circle cx="12" cy="12" r="10" />
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4m0 4h.01" />
    </svg>
  )
}

function ChevronDownIcon({ className = 'size-4' }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
    </svg>
  )
}

function ChevronUpIcon({ className = 'size-4' }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M5 15l7-7 7 7" />
    </svg>
  )
}

function XIcon({ className = 'size-4' }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
    </svg>
  )
}

function ArrowUpDownIcon({ className = 'size-3.5' }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4" />
    </svg>
  )
}

// ---------------------------------------------------------------------------
// Main Scenario Management Component
// ---------------------------------------------------------------------------

export default function ScenarioManagementPanel({
  currentWeights,
  currentMethod = 'weighted_sum',
  apiBaseUrl = '',
  siteNames = {},
  onSelectScenario,
  onScenarioSaved,
  className = '',
}: ScenarioManagementPanelProps) {
  // Scenarios list state
  const [scenarios, setScenarios] = useState<SavedScenario[]>([])
  const [loadingList, setLoadingList] = useState(false)
  const [listError, setListError] = useState<string | null>(null)
  const [activeScenarioId, setActiveScenarioId] = useState<number | null>(null)

  // Save modal state
  const [isSaveOpen, setIsSaveOpen] = useState(false)
  const [scenarioName, setScenarioName] = useState('')
  const [saveLoading, setSaveLoading] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)
  const [saveSuccessMsg, setSaveSuccessMsg] = useState<string | null>(null)

  // Compare selection (selected scenario IDs)
  const [selectedForCompare, setSelectedForCompare] = useState<number[]>([])

  // View Mode: 'list' or 'compare'
  const [viewMode, setViewMode] = useState<'list' | 'compare'>('list')

  // Compare API state
  const [compareData, setCompareData] = useState<ScenarioComparisonResponse | null>(null)
  const [compareLoading, setCompareLoading] = useState(false)
  const [compareError, setCompareError] = useState<string | null>(null)

  // Ranking preview for single scenario inspection (GET /scenarios/{id})
  const [inspectScenarioId, setInspectScenarioId] = useState<number | null>(null)
  const [inspectData, setInspectData] = useState<ScenarioWithRanking | null>(null)
  const [inspectLoading, setInspectLoading] = useState(false)
  const [inspectError, setInspectError] = useState<string | null>(null)

  // Compare table sorting
  const [sortScenarioId, setSortScenarioId] = useState<number | 'site_id'>('site_id')
  const [sortAscending, setSortAscending] = useState(true)

  // Weight sum analysis for current weights (File 46 requires weights to sum to 1.0)
  const currentTotalWeight = useMemo(() => {
    return Object.values(currentWeights).reduce((sum, w) => sum + (Number(w) || 0), 0)
  }, [currentWeights])

  const isWeightSumValid = useMemo(() => {
    return Math.abs(currentTotalWeight - 1.0) < 1e-5
  }, [currentTotalWeight])

  // -------------------------------------------------------------------------
  // Fetch Saved Scenarios (GET /scenarios)
  // -------------------------------------------------------------------------

  const fetchScenarios = useCallback(async () => {
    setLoadingList(true)
    setListError(null)

    try {
      const url = apiBaseUrl ? `${apiBaseUrl.replace(/\/+$/, '')}/scenarios` : apiUrl('/scenarios')
      const res = await fetch(url, {
        method: 'GET',
        headers: { Accept: 'application/json' },
      })

      if (!res.ok) {
        throw new Error(await readApiError(res, `Failed to load scenarios (${res.status})`))
      }

      const data: SavedScenario[] = await res.json()
      setScenarios(data)
    } catch (err) {
      setListError(err instanceof Error ? err.message : 'Unable to load saved scenarios.')
    } finally {
      setLoadingList(false)
    }
  }, [apiBaseUrl])

  useEffect(() => {
    fetchScenarios()
  }, [fetchScenarios])

  // -------------------------------------------------------------------------
  // Save Current Scenario (POST /scenarios)
  // -------------------------------------------------------------------------

  const handleOpenSaveDialog = () => {
    setScenarioName('')
    setSaveError(null)
    setSaveSuccessMsg(null)
    setIsSaveOpen(true)
  }

  const handleSaveScenario = async (e?: React.FormEvent) => {
    if (e) e.preventDefault()

    const trimmedName = scenarioName.trim()
    if (!trimmedName) {
      setSaveError('Scenario name cannot be blank.')
      return
    }

    if (Object.keys(currentWeights).length === 0) {
      setSaveError('Weight set must contain at least one criterion.')
      return
    }

    // Prepare weights, ensuring they normalize to sum 1.0 to fulfill File 46 validation
    let finalWeights = { ...currentWeights }
    if (!isWeightSumValid) {
      if (currentTotalWeight > 0) {
        finalWeights = normalizeWeights(currentWeights)
      } else {
        setSaveError('Weights cannot sum to zero.')
        return
      }
    }

    setSaveLoading(true)
    setSaveError(null)

    try {
      const url = apiBaseUrl ? `${apiBaseUrl.replace(/\/+$/, '')}/scenarios` : apiUrl('/scenarios')
      const res = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json',
        },
        body: JSON.stringify({
          name: trimmedName,
          weight_set: finalWeights,
          aggregation_method: currentMethod,
        }),
      })

      if (!res.ok) {
        throw new Error(await readApiError(res, `Save failed (${res.status})`))
      }

      const savedScenario: SavedScenario = await res.json()
      setSaveSuccessMsg(`Scenario "${savedScenario.name}" saved successfully!`)
      setActiveScenarioId(savedScenario.scenario_id)

      await fetchScenarios()
      onScenarioSaved?.(savedScenario)

      setTimeout(() => {
        setIsSaveOpen(false)
        setSaveSuccessMsg(null)
      }, 1200)
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : 'Error occurred while saving scenario.')
    } finally {
      setSaveLoading(false)
    }
  }

  // -------------------------------------------------------------------------
  // Compare Selection & API Fetch (GET /scenarios/compare?ids=...)
  // -------------------------------------------------------------------------

  const toggleCompareSelect = (id: number) => {
    setSelectedForCompare((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]
    )
  }

  const selectAllForCompare = () => {
    setSelectedForCompare(scenarios.map((s) => s.scenario_id))
  }

  const clearCompareSelection = () => {
    setSelectedForCompare([])
    setCompareData(null)
  }

  const fetchComparison = useCallback(
    async (scenarioIds: number[]) => {
      if (scenarioIds.length < 2) {
        setCompareData(null)
        return
      }

      setCompareLoading(true)
      setCompareError(null)

      try {
        const idsParam = scenarioIds.join(',')
        const url = apiBaseUrl
          ? `${apiBaseUrl.replace(/\/+$/, '')}/scenarios/compare?ids=${encodeURIComponent(idsParam)}`
          : apiUrl(`/scenarios/compare?ids=${encodeURIComponent(idsParam)}`)
        const res = await fetch(url, {
          method: 'GET',
          headers: { Accept: 'application/json' },
        })

        if (!res.ok) {
          throw new Error(await readApiError(res, `Failed to compare scenarios (${res.status})`))
        }

        const data: ScenarioComparisonResponse = await res.json()
        setCompareData(data)
      } catch (err) {
        setCompareError(err instanceof Error ? err.message : 'Unable to compare selected scenarios.')
      } finally {
        setCompareLoading(false)
      }
    },
    [apiBaseUrl]
  )

  useEffect(() => {
    if (viewMode === 'compare' && selectedForCompare.length >= 2) {
      fetchComparison(selectedForCompare)
    }
  }, [viewMode, selectedForCompare, fetchComparison])

  // -------------------------------------------------------------------------
  // Single Scenario Ranking Inspection (GET /scenarios/{scenario_id})
  // -------------------------------------------------------------------------

  const inspectScenario = async (id: number) => {
    if (inspectScenarioId === id) {
      setInspectScenarioId(null)
      setInspectData(null)
      setInspectError(null)
      return
    }

    setInspectScenarioId(id)
    setInspectLoading(true)
    setInspectData(null)
    setInspectError(null)

    try {
      const url = apiBaseUrl
        ? `${apiBaseUrl.replace(/\/+$/, '')}/scenarios/${id}`
        : apiUrl(`/scenarios/${id}`)
      const res = await fetch(url, {
        headers: { Accept: 'application/json' },
      })
      if (!res.ok) throw new Error(await readApiError(res, `Failed to load ranking for scenario #${id}`))
      const data: ScenarioWithRanking = await res.json()
      setInspectData(data)
    } catch (err) {
      setInspectData(null)
      setInspectError(err instanceof Error ? err.message : 'Unable to inspect scenario ranking.')
    } finally {
      setInspectLoading(false)
    }
  }

  // -------------------------------------------------------------------------
  // Comparison Table Sorting
  // -------------------------------------------------------------------------

  const sortedRankingRows = useMemo(() => {
    if (!compareData?.ranking) return []
    const rows = [...compareData.ranking]

    rows.sort((a, b) => {
      if (sortScenarioId === 'site_id') {
        const idA = String(a.site_id)
        const idB = String(b.site_id)
        return sortAscending ? idA.localeCompare(idB) : idB.localeCompare(idA)
      }

      const rankA = a[`scenario_${sortScenarioId}_rank`] ?? 999999
      const rankB = b[`scenario_${sortScenarioId}_rank`] ?? 999999
      return sortAscending ? rankA - rankB : rankB - rankA
    })

    return rows
  }, [compareData, sortScenarioId, sortAscending])

  const toggleSort = (col: number | 'site_id') => {
    if (sortScenarioId === col) {
      setSortAscending((prev) => !prev)
    } else {
      setSortScenarioId(col)
      setSortAscending(true)
    }
  }

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

  return (
    <div className={`space-y-4 rounded-xl border border-border bg-background p-5 shadow-xs ${className}`}>
      {/* Top Header & Mode Navigation */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border pb-4">
        <div>
          <div className="flex items-center gap-2">
            <BookmarkIcon className="size-5 text-primary" />
            <h2 className="text-lg font-semibold text-foreground">MCA Scenarios</h2>
          </div>
          <p className="mt-0.5 text-xs text-muted-foreground">
            Save criteria weighting scenarios, review recomputed rankings, and compare outcomes side-by-side.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {/* View Mode Toggle */}
          <div className="inline-flex rounded-lg border border-border bg-muted/30 p-0.5 text-xs">
            <button
              type="button"
              onClick={() => setViewMode('list')}
              className={`rounded-md px-3 py-1.5 font-medium transition-all ${
                viewMode === 'list'
                  ? 'bg-background text-foreground shadow-xs'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              Saved List ({scenarios.length})
            </button>
            <button
              type="button"
              onClick={() => setViewMode('compare')}
              className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 font-medium transition-all ${
                viewMode === 'compare'
                  ? 'bg-background text-foreground shadow-xs'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              <GitCompareIcon className="size-3.5" />
              Compare
              {selectedForCompare.length > 0 && (
                <span className="ml-0.5 rounded-full bg-primary/15 px-1.5 py-0.2 text-[10px] font-semibold text-primary">
                  {selectedForCompare.length}
                </span>
              )}
            </button>
          </div>

          {/* 1. Save Current Scenario Button */}
          <Button
            type="button"
            variant="default"
            size="sm"
            onClick={handleOpenSaveDialog}
            className="flex items-center gap-1.5"
          >
            <PlusIcon className="size-4" />
            <span>Save current scenario</span>
          </Button>
        </div>
      </div>

      {/* Save Scenario Modal Dialog */}
      {isSaveOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4 backdrop-blur-xs"
          role="dialog"
          aria-modal="true"
        >
          <div className="w-full max-w-md rounded-xl border border-border bg-background p-6 shadow-xl">
            <div className="flex items-center justify-between pb-3 border-b border-border">
              <h3 className="text-base font-semibold text-foreground flex items-center gap-2">
                <BookmarkIcon className="size-4 text-primary" />
                Save Current Scenario
              </h3>
              <button
                type="button"
                onClick={() => setIsSaveOpen(false)}
                className="text-muted-foreground hover:text-foreground p-1 rounded-md"
              >
                <XIcon className="size-4" />
              </button>
            </div>

            <form onSubmit={handleSaveScenario} className="mt-4 space-y-4">
              <div>
                <label htmlFor="scenario-name-input" className="block text-xs font-medium text-foreground">
                  Scenario Name <span className="text-destructive">*</span>
                </label>
                <input
                  id="scenario-name-input"
                  type="text"
                  value={scenarioName}
                  onChange={(e) => setScenarioName(e.target.value)}
                  placeholder="e.g. Green Transition 2026, Fast-Track Grid"
                  maxLength={255}
                  required
                  autoFocus
                  className="mt-1.5 w-full rounded-lg border border-border bg-background px-3 py-2 text-sm text-foreground outline-none focus:border-primary focus:ring-2 focus:ring-primary/20"
                />
                <div className="mt-1 flex justify-between text-[11px] text-muted-foreground">
                  <span>Name this MCA configuration for future comparison</span>
                  <span>{scenarioName.length}/255</span>
                </div>
              </div>

              {/* Live Parameters Preview */}
              <div className="rounded-lg border border-border bg-muted/30 p-3 text-xs space-y-2">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Aggregation Method:</span>
                  <span className="font-semibold text-foreground">{formatMethodName(currentMethod)}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Criteria Count:</span>
                  <span className="font-medium text-foreground">{Object.keys(currentWeights).length} criteria</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Weights Sum:</span>
                  <span
                    className={`font-semibold tabular-nums ${
                      isWeightSumValid ? 'text-green-600 dark:text-green-400' : 'text-amber-600 dark:text-amber-400'
                    }`}
                  >
                    {currentTotalWeight.toFixed(4)} {isWeightSumValid ? '(Valid 1.0)' : '(Auto-normalized to 1.0)'}
                  </span>
                </div>

                <div className="mt-2 pt-2 border-t border-border/50">
                  <span className="text-[11px] text-muted-foreground block mb-1.5">Weight breakdown:</span>
                  <div className="flex flex-wrap gap-1.5 max-h-24 overflow-y-auto">
                    {Object.entries(currentWeights).map(([crit, w]) => (
                      <span
                        key={crit}
                        className="inline-flex items-center gap-1 rounded bg-background px-2 py-0.5 text-[11px] border border-border"
                      >
                        <span className="font-medium text-foreground">{crit}:</span>
                        <span className="text-muted-foreground tabular-nums">{formatPercent(w)}</span>
                      </span>
                    ))}
                  </div>
                </div>
              </div>

              {saveError && (
                <div className="flex items-start gap-2 rounded-lg border border-destructive/30 bg-destructive/10 p-3 text-xs text-destructive">
                  <AlertCircleIcon className="size-4 shrink-0 mt-0.5" />
                  <span>{saveError}</span>
                </div>
              )}

              {saveSuccessMsg && (
                <div className="flex items-center gap-2 rounded-lg border border-green-500/30 bg-green-500/10 p-3 text-xs text-green-700 dark:text-green-300">
                  <CheckIcon className="size-4 shrink-0" />
                  <span>{saveSuccessMsg}</span>
                </div>
              )}

              <div className="flex justify-end gap-2 pt-2">
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => setIsSaveOpen(false)}
                  disabled={saveLoading}
                >
                  Cancel
                </Button>
                <Button
                  type="submit"
                  variant="default"
                  size="sm"
                  disabled={saveLoading || !scenarioName.trim()}
                  className="min-w-24"
                >
                  {saveLoading ? 'Saving…' : 'Save Scenario'}
                </Button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* =================================================================== */}
      {/* 2. LIST OF SAVED SCENARIOS */}
      {/* =================================================================== */}
      {viewMode === 'list' && (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3 text-xs">
            <div className="text-muted-foreground">
              {scenarios.length === 0 ? 'No scenarios saved yet.' : `${scenarios.length} saved scenarios`}
              {selectedForCompare.length > 0 && (
                <span className="ml-2 font-medium text-foreground">
                  ({selectedForCompare.length} selected for compare)
                </span>
              )}
            </div>

            <div className="flex items-center gap-2">
              {scenarios.length > 0 && (
                <>
                  <Button
                    type="button"
                    variant="ghost"
                    size="xs"
                    onClick={selectedForCompare.length === scenarios.length ? clearCompareSelection : selectAllForCompare}
                    className="text-xs"
                  >
                    {selectedForCompare.length === scenarios.length ? 'Deselect All' : 'Select All'}
                  </Button>

                  {selectedForCompare.length >= 2 && (
                    <Button
                      type="button"
                      variant="secondary"
                      size="xs"
                      onClick={() => setViewMode('compare')}
                      className="text-xs font-semibold flex items-center gap-1 text-primary"
                    >
                      <GitCompareIcon className="size-3.5" />
                      Compare Selected ({selectedForCompare.length})
                    </Button>
                  )}
                </>
              )}

              <Button
                type="button"
                variant="ghost"
                size="icon-xs"
                onClick={fetchScenarios}
                title="Refresh scenarios list"
                aria-label="Refresh scenarios"
                disabled={loadingList}
              >
                <RefreshIcon className={`size-3.5 ${loadingList ? 'animate-spin' : ''}`} />
              </Button>
            </div>
          </div>

          {/* Loading Skeleton */}
          {loadingList && scenarios.length === 0 && (
            <div className="space-y-2 py-4">
              {[0, 1, 2].map((i) => (
                <div key={i} className="h-16 rounded-lg border border-border bg-muted/20 animate-pulse p-3" />
              ))}
            </div>
          )}

          {/* Error Message */}
          {listError && (
            <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-4 text-xs text-destructive flex items-start justify-between gap-3">
              <div className="flex items-start gap-2">
                <AlertCircleIcon className="size-4 shrink-0 mt-0.5" />
                <span>{listError}</span>
              </div>
              <Button type="button" variant="outline" size="xs" onClick={fetchScenarios}>
                Retry
              </Button>
            </div>
          )}

          {/* Empty State */}
          {!loadingList && scenarios.length === 0 && !listError && (
            <div className="rounded-lg border border-dashed border-border py-12 text-center">
              <BookmarkIcon className="mx-auto size-8 text-muted-foreground/50" />
              <h4 className="mt-2 text-sm font-semibold text-foreground">No saved scenarios yet</h4>
              <p className="mt-1 text-xs text-muted-foreground max-w-sm mx-auto">
                Save your active weights and aggregation method to recall or compare alternative decision criteria.
              </p>
              <Button
                type="button"
                variant="default"
                size="sm"
                className="mt-4"
                onClick={handleOpenSaveDialog}
              >
                <PlusIcon className="size-3.5 mr-1" />
                Save Current Scenario
              </Button>
            </div>
          )}

          {/* Scenario Cards */}
          <div className="grid gap-2.5">
            {scenarios.map((scenario) => {
              const isSelectedForCompare = selectedForCompare.includes(scenario.scenario_id)
              const isActive = activeScenarioId === scenario.scenario_id
              const isInspecting = inspectScenarioId === scenario.scenario_id
              const criteriaList = Object.entries(scenario.weight_set || {})

              return (
                <div
                  key={scenario.scenario_id}
                  className={`rounded-lg border transition-all ${
                    isActive
                      ? 'border-primary/80 bg-primary/5 ring-1 ring-primary/30'
                      : isSelectedForCompare
                      ? 'border-primary/50 bg-muted/20'
                      : 'border-border bg-background hover:border-border/80 hover:bg-muted/10'
                  }`}
                >
                  <div className="p-3.5 flex flex-col md:flex-row md:items-center justify-between gap-3">
                    {/* Checkbox + Details */}
                    <div className="flex items-start gap-3 min-w-0">
                      <label className="flex items-center cursor-pointer mt-0.5">
                        <input
                          type="checkbox"
                          checked={isSelectedForCompare}
                          onChange={() => toggleCompareSelect(scenario.scenario_id)}
                          className="size-4 rounded border-border text-primary focus:ring-primary"
                          title="Select for compare view"
                        />
                      </label>

                      <div className="min-w-0">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="font-semibold text-sm text-foreground truncate">
                            {scenario.name}
                          </span>
                          <span className="rounded bg-muted px-2 py-0.5 text-[10px] font-medium text-muted-foreground">
                            {formatMethodName(scenario.aggregation_method)}
                          </span>
                          <span className="text-[11px] text-muted-foreground">
                            #{scenario.scenario_id}
                          </span>
                        </div>

                        <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
                          <span>{criteriaList.length} criteria</span>
                          <span>•</span>
                          <span>{formatDate(scenario.created_at)}</span>
                        </div>
                      </div>
                    </div>

                    {/* Actions */}
                    <div className="flex items-center gap-2 self-end md:self-auto shrink-0">
                      <Button
                        type="button"
                        variant="outline"
                        size="xs"
                        onClick={() => inspectScenario(scenario.scenario_id)}
                        className="text-xs flex items-center gap-1"
                        title="Recompute and inspect rankings with File 32"
                      >
                        <span>Rankings</span>
                        {isInspecting ? (
                          <ChevronUpIcon className="size-3" />
                        ) : (
                          <ChevronDownIcon className="size-3" />
                        )}
                      </Button>

                      <Button
                        type="button"
                        variant={isActive ? 'secondary' : 'default'}
                        size="xs"
                        onClick={() => {
                          setActiveScenarioId(scenario.scenario_id)
                          onSelectScenario?.(scenario)
                        }}
                        className="text-xs"
                      >
                        {isActive ? (
                          <span className="flex items-center gap-1 text-primary">
                            <CheckIcon className="size-3" /> Loaded
                          </span>
                        ) : (
                          'Load Weights'
                        )}
                      </Button>
                    </div>
                  </div>

                  {/* Weights Breakdown Chips */}
                  <div className="px-3.5 pb-3 flex flex-wrap gap-1.5 border-t border-border/40 pt-2">
                    {criteriaList.slice(0, 5).map(([criterion, weight]) => (
                      <span
                        key={criterion}
                        className="inline-flex items-center gap-1 rounded bg-muted/50 px-2 py-0.5 text-[11px] text-muted-foreground"
                      >
                        <span className="font-medium text-foreground">{criterion}:</span>
                        <span className="tabular-nums font-mono">{formatPercent(Number(weight))}</span>
                      </span>
                    ))}
                    {criteriaList.length > 5 && (
                      <span className="rounded bg-muted/30 px-2 py-0.5 text-[11px] text-muted-foreground">
                        +{criteriaList.length - 5} more
                      </span>
                    )}
                  </div>

                  {/* Expandable Individual Ranking Preview (GET /scenarios/{id}) */}
                  {isInspecting && (
                    <div className="border-t border-border bg-muted/20 p-3 text-xs">
                      {inspectLoading && (
                        <div className="py-4 text-center text-muted-foreground">
                          Recomputing site scores via File 32 scoring…
                        </div>
                      )}
                      {!inspectLoading && inspectError && (
                        <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-3 text-destructive flex items-center justify-between gap-3">
                          <div className="flex items-center gap-2">
                            <AlertCircleIcon className="size-4 shrink-0" />
                            <span>{inspectError}</span>
                          </div>
                          <Button
                            type="button"
                            variant="outline"
                            size="xs"
                            onClick={() => inspectScenario(scenario.scenario_id)}
                          >
                            Retry
                          </Button>
                        </div>
                      )}
                      {!inspectLoading && !inspectError && inspectData && (
                        <div>
                          <div className="flex items-center justify-between pb-2 mb-2 border-b border-border">
                            <span className="font-semibold text-foreground">Top Ranked Sites</span>
                            <span className="text-[11px] text-muted-foreground">
                              {inspectData.ranking.length} sites evaluated
                            </span>
                          </div>
                          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2 max-h-56 overflow-y-auto pr-1">
                            {inspectData.ranking.slice(0, 9).map((site) => (
                              <div
                                key={String(site.site_id)}
                                className="flex items-center justify-between rounded border border-border bg-background p-2"
                              >
                                <div className="flex items-center gap-2 min-w-0">
                                  <span
                                    className={`flex size-5 shrink-0 items-center justify-center rounded-full text-[10px] font-bold ${
                                      site.rank === 1
                                        ? 'bg-amber-500/20 text-amber-600 dark:text-amber-400'
                                        : 'bg-muted text-muted-foreground'
                                    }`}
                                  >
                                    {site.rank}
                                  </span>
                                  <span className="truncate font-medium text-foreground">
                                    {siteNames[String(site.site_id)] ?? `Site ${site.site_id}`}
                                  </span>
                                </div>
                                <span className="font-mono text-muted-foreground tabular-nums pl-2">
                                  {formatScore(site.score)}
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* =================================================================== */}
      {/* 3. SIDE-BY-SIDE COMPARE VIEW */}
      {/* =================================================================== */}
      {viewMode === 'compare' && (
        <div className="space-y-4">
          {/* Comparison Header Bar */}
          <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-border bg-muted/20 p-3 text-xs">
            <div className="flex items-center gap-2">
              <GitCompareIcon className="size-4 text-primary" />
              <span className="font-semibold text-foreground">
                Comparing {selectedForCompare.length} Scenario{selectedForCompare.length === 1 ? '' : 's'}
              </span>
              {selectedForCompare.length < 2 && (
                <span className="text-amber-600 dark:text-amber-400 font-medium">
                  (Select at least 2 scenarios to compute side-by-side rankings)
                </span>
              )}
            </div>

            <div className="flex items-center gap-2">
              <Button
                type="button"
                variant="outline"
                size="xs"
                onClick={clearCompareSelection}
              >
                Clear Selection
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="xs"
                onClick={() => fetchComparison(selectedForCompare)}
                disabled={selectedForCompare.length < 2 || compareLoading}
              >
                <RefreshIcon className={`size-3.5 mr-1 ${compareLoading ? 'animate-spin' : ''}`} />
                Re-score
              </Button>
            </div>
          </div>

          {/* Quick Scenario Toggles */}
          <div className="flex flex-wrap items-center gap-2 pb-1 text-xs">
            <span className="text-muted-foreground">Scenarios to compare:</span>
            {scenarios.map((s) => {
              const selected = selectedForCompare.includes(s.scenario_id)
              return (
                <button
                  key={s.scenario_id}
                  type="button"
                  onClick={() => toggleCompareSelect(s.scenario_id)}
                  className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-1 transition-colors ${
                    selected
                      ? 'border-primary bg-primary/10 font-semibold text-primary'
                      : 'border-border bg-background text-muted-foreground hover:bg-muted'
                  }`}
                >
                  <span>{s.name}</span>
                  {selected ? <CheckIcon className="size-3" /> : <PlusIcon className="size-3" />}
                </button>
              )
            })}
          </div>

          {/* Prompt if < 2 selected */}
          {selectedForCompare.length < 2 && (
            <div className="rounded-lg border border-dashed border-border p-8 text-center text-xs text-muted-foreground">
              Select at least 2 saved scenarios from above or the list view to see their rankings side-by-side.
            </div>
          )}

          {/* Loading Indicator */}
          {compareLoading && (
            <div className="rounded-lg border border-border p-8 text-center text-xs text-muted-foreground animate-pulse">
              Re-scoring and comparing scenarios through File 32…
            </div>
          )}

          {/* Error Message */}
          {compareError && (
            <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-4 text-xs text-destructive flex items-center justify-between">
              <span>{compareError}</span>
              <Button
                type="button"
                variant="outline"
                size="xs"
                onClick={() => fetchComparison(selectedForCompare)}
              >
                Retry
              </Button>
            </div>
          )}

          {/* Side-by-Side Comparison Table */}
          {!compareLoading && !compareError && compareData && compareData.scenarios.length >= 2 && (
            <div className="overflow-x-auto rounded-lg border border-border">
              <table className="w-full min-w-[700px] border-collapse text-left text-xs">
                <thead>
                  {/* Scenario Headers */}
                  <tr className="border-b border-border bg-muted/40">
                    <th className="sticky left-0 z-20 bg-muted/80 backdrop-blur-xs p-3 font-semibold text-foreground min-w-[180px]">
                      <button
                        type="button"
                        onClick={() => toggleSort('site_id')}
                        className="flex items-center gap-1.5 font-semibold text-foreground hover:underline"
                      >
                        <span>Candidate Site</span>
                        <ArrowUpDownIcon className="size-3 text-muted-foreground" />
                      </button>
                    </th>

                    {compareData.scenarios.map((sc) => (
                      <th
                        key={sc.scenario_id}
                        colSpan={2}
                        className="border-l border-border p-3 align-top min-w-[200px]"
                      >
                        <div className="font-semibold text-sm text-foreground flex items-center justify-between">
                          <span>{sc.name}</span>
                          <span className="text-[10px] font-normal text-muted-foreground">
                            #{sc.scenario_id}
                          </span>
                        </div>
                        <div className="mt-1 flex items-center gap-2">
                          <span className="rounded bg-background px-1.5 py-0.5 text-[10px] font-medium border border-border text-foreground">
                            {formatMethodName(sc.aggregation_method)}
                          </span>
                          <span className="text-[10px] text-muted-foreground">
                            {Object.keys(sc.weight_set || {}).length} criteria
                          </span>
                        </div>
                      </th>
                    ))}
                  </tr>

                  {/* Sub-headers: Rank & Score per Scenario */}
                  <tr className="border-b border-border bg-muted/20 text-[11px] text-muted-foreground">
                    <th className="sticky left-0 z-20 bg-muted/70 backdrop-blur-xs px-3 py-2 font-medium">
                      Site ID
                    </th>
                    {compareData.scenarios.map((sc) => (
                      <th
                        key={`sub-${sc.scenario_id}`}
                        colSpan={2}
                        className="border-l border-border px-3 py-2 font-medium"
                      >
                        <div className="flex items-center justify-between">
                          <button
                            type="button"
                            onClick={() => toggleSort(sc.scenario_id)}
                            className="flex items-center gap-1 hover:text-foreground font-semibold"
                          >
                            <span>Rank</span>
                            <ArrowUpDownIcon className="size-2.5" />
                          </button>
                          <span>Score</span>
                        </div>
                      </th>
                    ))}
                  </tr>
                </thead>

                <tbody className="divide-y divide-border">
                  {sortedRankingRows.map((row) => {
                    const sid = String(row.site_id)
                    const sname = siteNames[sid] ?? `Site ${sid}`

                    return (
                      <tr key={sid} className="hover:bg-muted/15 transition-colors">
                        {/* Sticky Candidate Site Name & ID */}
                        <td className="sticky left-0 z-10 bg-background/95 backdrop-blur-xs p-3 font-medium text-foreground">
                          <div className="truncate max-w-[160px] font-medium">{sname}</div>
                          <div className="text-[10px] text-muted-foreground font-mono">{sid}</div>
                        </td>

                        {/* Scenario Columns */}
                        {compareData.scenarios.map((sc) => {
                          const rank = row[`scenario_${sc.scenario_id}_rank`]
                          const score = row[`scenario_${sc.scenario_id}_score`]
                          const isTop = rank === 1

                          return (
                            <td
                              key={`cell-${sc.scenario_id}`}
                              colSpan={2}
                              className={`border-l border-border p-3 tabular-nums ${
                                isTop ? 'bg-amber-500/5' : ''
                              }`}
                            >
                              <div className="flex items-center justify-between">
                                <div className="flex items-center gap-1.5">
                                  <span
                                    className={`inline-flex size-5 items-center justify-center rounded-full text-[10px] font-bold ${
                                      isTop
                                        ? 'bg-amber-500/20 text-amber-700 dark:text-amber-400 ring-1 ring-amber-500/40'
                                        : rank && rank <= 3
                                        ? 'bg-primary/10 text-primary font-semibold'
                                        : 'bg-muted text-muted-foreground'
                                    }`}
                                  >
                                    {rank ?? '—'}
                                  </span>
                                  {isTop && (
                                    <span className="text-[9px] font-bold uppercase tracking-wider text-amber-600 dark:text-amber-400">
                                      Top 1
                                    </span>
                                  )}
                                </div>

                                <span className="font-mono text-xs text-foreground font-medium">
                                  {formatScore(score)}
                                </span>
                              </div>
                            </td>
                          )
                        })}
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  )
}