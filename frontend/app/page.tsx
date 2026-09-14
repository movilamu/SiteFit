'use client'

import dynamic from 'next/dynamic'
import { useEffect, useMemo, useRef, useState } from 'react'

import {
  displayName,
  displayPlace,
  displayScore,
  fetchCatchment,
  fetchScores,
  fetchSites,
  siteId,
  type CandidateSite,
} from '@/lib/sites'
import {
  DEFAULT_METHOD,
  DEFAULT_WEIGHTS,
  normalizeWeights,
  weightSum,
} from '@/lib/weights'
import ScoreDecompositionPanel from '@/components/score-decomposition-panel'
import SiteComparison from '@/components/site-comparison'
import SensitivityAnalysisPanel from '@/components/sensitivity-analysis-panel'
import ScenarioManagementPanel from '@/components/scenario-management-panel'
import ExportPanel from '@/components/export-panel'
import ConstraintFilterPanel, { type SitesApiResponse } from '@/components/constraint-filter-panel'

const SitesMap = dynamic(() => import('@/components/sites-map'), {
  ssr: false,
  loading: () => (
    <div className="absolute inset-0 flex items-center justify-center text-xs text-slate-500">
      Loading map…
    </div>
  ),
})

const CRITERIA_DEFINITIONS = [
  { id: 'population', label: 'Population', color: '#38bdf8' },
  { id: 'demographics', label: 'Demographics', color: '#a78bfa' },
  { id: 'spending', label: 'Spending Power', color: '#f472b6' },
  { id: 'accessibility', label: 'Accessibility', color: '#fb923c' },
  { id: 'competition', label: 'Competition', color: '#facc15' },
  { id: 'cannibalisation', label: 'Cannibalisation', color: '#4ade80' },
  { id: 'site', label: 'Site & Property', color: '#2dd4bf' },
  { id: 'risk', label: 'Market Risk', color: '#f87171' },
]

export default function Page() {
  const [sites, setSites] = useState<CandidateSite[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [sitesError, setSitesError] = useState<string | null>(null)
  const [sitesLoading, setSitesLoading] = useState(true)

  const [weights, setWeights] = useState<Record<string, number>>(DEFAULT_WEIGHTS)
  const [method, setMethod] = useState<string>(DEFAULT_METHOD)
  const [scoringLoading, setScoringLoading] = useState(false)
  const [scoringError, setScoringError] = useState<string | null>(null)
  const [scoresMap, setScoresMap] = useState<Record<string, { rank: number; score: number }>>({})

  const [catchment, setCatchment] = useState<GeoJSON.Geometry | null>(null)
  const [catchmentNote, setCatchmentNote] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'Weights' | 'Comparison' | 'Sensitivity' | 'Scenarios' | 'Export'>('Weights')
  const [showConstraintFilter, setShowConstraintFilter] = useState(false)

  const listRef = useRef<HTMLDivElement | null>(null)

  // Load candidate sites from GET /sites
  useEffect(() => {
    let cancelled = false
    setSitesLoading(true)
    fetchSites()
      .then((rows) => {
        if (cancelled) return
        setSites(rows)
        setSitesError(null)
        if (rows[0]) {
          setSelectedId(siteId(rows[0]))
        }
      })
      .catch((error: unknown) => {
        if (cancelled) return
        setSites([])
        setSitesError(error instanceof Error ? error.message : 'Failed to load GET /sites')
      })
      .finally(() => {
        if (!cancelled) setSitesLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  // Call POST /score (File 44) when weights or method change to update ranking
  useEffect(() => {
    let cancelled = false
    setScoringLoading(true)
    setScoringError(null)

    fetchScores(weights, method)
      .then((res) => {
        if (cancelled) return
        const map: Record<string, { rank: number; score: number }> = {}
        for (const item of res.sites) {
          map[String(item.site_id)] = { rank: item.rank, score: item.score }
        }
        setScoresMap(map)
        setScoringError(null)
      })
      .catch((err: unknown) => {
        if (cancelled) return
        // Keep previous scores if available, show error banner
        setScoringError(err instanceof Error ? err.message : 'Failed to calculate scores from POST /score')
      })
      .finally(() => {
        if (!cancelled) setScoringLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [weights, method])

  // Ranked sites sorted according to File 44 scores (fallback to site attractiveness)
  const sortedSites = useMemo(() => {
    const list = [...sites]
    list.sort((a, b) => {
      const idA = siteId(a)
      const idB = siteId(b)
      const scoreA = scoresMap[idA]?.score ?? displayScore(a) ?? -Infinity
      const scoreB = scoresMap[idB]?.score ?? displayScore(b) ?? -Infinity
      return scoreB - scoreA
    })
    return list
  }, [sites, scoresMap])

  const selectedSite = useMemo(
    () => sortedSites.find((site) => siteId(site) === selectedId) ?? sortedSites[0] ?? null,
    [sortedSites, selectedId],
  )
  const selectedIndex = selectedSite ? sortedSites.findIndex((site) => siteId(site) === siteId(selectedSite)) : -1

  const siteNameMap = useMemo(() => {
    const map: Record<string, string> = {}
    for (const site of sites) {
      map[siteId(site)] = displayName(site)
    }
    return map
  }, [sites])

  // Fetch catchment for selected site
  useEffect(() => {
    if (!selectedSite) {
      setCatchment(null)
      setCatchmentNote(null)
      return
    }
    let cancelled = false
    fetchCatchment(selectedSite).then((result) => {
      if (cancelled) return
      setCatchment(result.geometry)
      setCatchmentNote(result.geometry ? null : result.enhancement ?? null)
    })
    return () => {
      cancelled = true
    }
  }, [selectedSite])

  useEffect(() => {
    if (!selectedId || !listRef.current) return
    const row = listRef.current.querySelector(`[data-site-id="${CSS.escape(selectedId)}"]`)
    row?.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
  }, [selectedId])

  function handleWeightChange(criterionId: string, val: number) {
    setWeights((prev) => ({
      ...prev,
      [criterionId]: Math.max(0, val),
    }))
  }

  function handleAutoNormalize() {
    setWeights((prev) => normalizeWeights(prev))
  }

  function handleFilterApplied(filteredData: SitesApiResponse) {
    if (Array.isArray(filteredData.sites)) {
      const mapped: CandidateSite[] = filteredData.sites.map((s) => ({
        site_id: s.site_id,
        name: s.name ?? `Site ${s.site_id}`,
        coordinates: s.coordinates ?? { latitude: 13.0827, longitude: 80.2707 },
        unit_size: s.unit_size ?? null,
        rent: s.rent ?? null,
        tenure: s.tenure ?? null,
        status: s.status ?? null,
        attractiveness: s.attractiveness ?? null,
        attributes: {},
      }))
      setSites(mapped)
      if (mapped[0]) {
        setSelectedId(siteId(mapped[0]))
      }
    }
  }

  function handleFilterReset() {
    setSitesLoading(true)
    fetchSites()
      .then((rows) => {
        setSites(rows)
        setSitesError(null)
        if (rows[0]) setSelectedId(siteId(rows[0]))
      })
      .catch((err: unknown) => {
        setSitesError(err instanceof Error ? err.message : 'Failed to reload sites')
      })
      .finally(() => {
        setSitesLoading(false)
      })
  }

  const currentTotalWeight = useMemo(() => weightSum(weights), [weights])
  const isSumValid = Math.abs(currentTotalWeight - 1.0) <= 0.01

  return (
    <main className="min-h-screen bg-[#0b0d12] text-slate-100">
      <header className="flex h-[72px] items-center justify-between border-b border-white/[0.08] bg-[#10131a] px-7">
        <div className="flex items-center gap-8">
          <div className="flex items-center gap-3">
            <div className="flex size-8 items-center justify-center rounded-lg bg-cyan-400 text-sm font-black text-[#081016]">R</div>
            <div>
              <p className="text-sm font-semibold tracking-tight">Retail Atlas</p>
              <p className="text-[10px] uppercase tracking-[0.18em] text-slate-500">Site ranking workspace</p>
            </div>
          </div>
          <div className="hidden h-7 w-px bg-white/10 md:block" />
          <nav className="flex items-center gap-1" aria-label="Dashboard sections">
            {(['Weights', 'Comparison', 'Sensitivity', 'Scenarios', 'Export'] as const).map((tab) => (
              <button
                key={tab}
                type="button"
                onClick={() => setActiveTab(tab)}
                className={`rounded-md px-3 py-2 text-xs font-medium transition-colors ${activeTab === tab ? 'bg-cyan-400/15 text-cyan-300 ring-1 ring-cyan-400/30' : 'text-slate-400 hover:bg-white/5 hover:text-slate-200'}`}
              >
                {tab}
              </button>
            ))}
          </nav>
        </div>
        <div className="flex items-center gap-3 text-xs text-slate-500">
          <span className="hidden sm:inline">Tamil Nadu pilot · Greater Chennai</span>
          <button
            type="button"
            onClick={() => setShowConstraintFilter((prev) => !prev)}
            className={`rounded-md border px-3 py-1.5 font-medium transition-colors ${showConstraintFilter ? 'border-cyan-400 bg-cyan-400/15 text-cyan-300' : 'border-white/10 text-slate-300 hover:bg-white/5'}`}
          >
            {showConstraintFilter ? 'Hide Constraints' : 'Filter Constraints'}
          </button>
        </div>
      </header>

      {/* Main Tab Views */}
      {activeTab === 'Comparison' && (
        <div className="p-6 md:p-8">
          <SiteComparison sites={sites} weights={weights} method={method} />
        </div>
      )}

      {activeTab === 'Sensitivity' && (
        <div className="p-6 md:p-8">
          <SensitivityAnalysisPanel weights={weights} method={method} siteNames={siteNameMap} />
        </div>
      )}

      {activeTab === 'Scenarios' && (
        <div className="p-6 md:p-8">
          <ScenarioManagementPanel
            currentWeights={weights}
            currentMethod={method}
            siteNames={siteNameMap}
            onSelectScenario={(sc) => {
              setWeights(sc.weight_set)
              if (sc.aggregation_method) setMethod(sc.aggregation_method)
            }}
          />
        </div>
      )}

      {activeTab === 'Export' && (
        <div className="p-6 md:p-8">
          <ExportPanel weights={weights} method={method} />
        </div>
      )}

      {activeTab === 'Weights' && (
        <div className="flex min-h-[calc(100vh-72px)] flex-col lg:flex-row">
          {/* Left Sidebar: Model Weights Controls */}
          <aside className="w-full shrink-0 border-b border-white/[0.08] bg-[#0e1117] p-6 lg:w-[320px] lg:border-b-0 lg:border-r">
            <div className="mb-6 flex items-center justify-between">
              <div>
                <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">Model controls</p>
                <h1 className="mt-1 text-lg font-semibold tracking-tight">Criteria Weights</h1>
              </div>
              <span className={`rounded-full px-2.5 py-0.5 text-[10px] font-semibold ${isSumValid ? 'bg-emerald-400/10 text-emerald-400 border border-emerald-400/20' : 'bg-amber-400/10 text-amber-300 border border-amber-400/20'}`}>
                Sum: {currentTotalWeight.toFixed(2)}
              </span>
            </div>

            {/* Method selection */}
            <div className="mb-5 space-y-1.5">
              <label className="text-xs font-medium text-slate-400">Aggregation Method</label>
              <select
                aria-label="Aggregation Method"
                value={method}
                onChange={(e) => setMethod(e.target.value)}
                className="w-full rounded-md border border-white/10 bg-white/5 px-3 py-2 text-xs font-medium text-slate-200 focus:border-cyan-400 focus:outline-none"
              >
                <option value="weighted_sum" className="bg-[#11151d] text-slate-200">Weighted Sum (SAW)</option>
                <option value="weighted_product" className="bg-[#11151d] text-slate-200">Weighted Product (WPM)</option>
                <option value="distance_to_ideal" className="bg-[#11151d] text-slate-200">Distance to Ideal (TOPSIS)</option>
              </select>
            </div>

            {/* Weight sliders */}
            <div className="space-y-4 max-h-[calc(100vh-380px)] overflow-y-auto pr-1">
              {CRITERIA_DEFINITIONS.map((crit) => {
                const val = weights[crit.id] ?? 0
                return (
                  <div key={crit.id} className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-3">
                    <div className="mb-1.5 flex items-center justify-between text-xs">
                      <span className="font-medium text-slate-300">{crit.label}</span>
                      <span className="font-mono tabular-nums text-cyan-300">{(val * 100).toFixed(1)}%</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.01"
                      value={val}
                      onChange={(e) => handleWeightChange(crit.id, parseFloat(e.target.value))}
                      className="w-full accent-cyan-400"
                      aria-label={`${crit.label} weight slider`}
                    />
                  </div>
                )
              })}
            </div>

            {/* Normalize button & stats */}
            <div className="mt-5 border-t border-white/[0.07] pt-4 space-y-3">
              <button
                type="button"
                onClick={handleAutoNormalize}
                className="w-full rounded-md border border-cyan-400/30 bg-cyan-400/10 py-2 text-xs font-semibold text-cyan-300 transition-colors hover:bg-cyan-400/20"
              >
                Auto-Normalize Weights to 1.00
              </button>
              <div className="flex items-center justify-between text-xs text-slate-500">
                <span>Ranked Sites: <strong className="text-slate-300">{sortedSites.length}</strong></span>
                <span>{scoringLoading ? 'Calculating…' : 'Up to date'}</span>
              </div>
            </div>
          </aside>

          {/* Center Workspace */}
          <section className="flex-1 p-5 md:p-7 space-y-5">
            {/* Optional Constraint Filter Dropdown/Panel */}
            {showConstraintFilter && (
              <div className="mb-5">
                <ConstraintFilterPanel
                  onFilterChange={handleFilterApplied}
                  onResetFilters={handleFilterReset}
                />
              </div>
            )}

            <div className="flex flex-wrap items-end justify-between gap-4">
              <div>
                <p className="text-xs font-medium text-cyan-300">TAMIL NADU PILOT / CHENNAI</p>
                <h2 className="mt-1 text-2xl font-semibold tracking-tight">Retail site ranking</h2>
                <p className="mt-1 text-sm text-slate-500">Candidate sites scored via File 44 and mapped over Greater Chennai.</p>
              </div>
              <div className="flex items-center gap-2">
                {scoringLoading && (
                  <span className="flex items-center gap-1 text-xs text-cyan-400">
                    <span className="size-2 rounded-full bg-cyan-400 animate-ping" />
                    Scoring…
                  </span>
                )}
              </div>
            </div>

            {scoringError && (
              <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 p-3 text-xs text-rose-300">
                {scoringError}
              </div>
            )}

            <div className="grid gap-5 xl:grid-cols-[minmax(300px,0.78fr)_minmax(420px,1.22fr)]">
              {/* Site Ranking List */}
              <div className="overflow-hidden rounded-xl border border-white/[0.09] bg-[#11151d]">
                <div className="flex items-center justify-between border-b border-white/[0.08] px-5 py-4">
                  <div>
                    <h3 className="text-sm font-semibold">Ranked sites</h3>
                    <p className="mt-1 text-xs text-slate-500">
                      {sitesLoading ? 'Loading candidate sites…' : `${sortedSites.length} locations ranked`}
                    </p>
                  </div>
                  <span className="text-xs text-slate-500">Score ↓</span>
                </div>
                <div ref={listRef} className="max-h-[390px] overflow-y-auto p-2">
                  {sitesError && (
                    <p className="px-3 py-4 text-xs leading-5 text-rose-300">{sitesError}</p>
                  )}
                  {!sitesLoading && !sitesError && sortedSites.length === 0 && (
                    <p className="px-3 py-4 text-xs text-slate-500">No candidate sites available.</p>
                  )}
                  {sortedSites.map((site, index) => {
                    const id = siteId(site)
                    const scoreInfo = scoresMap[id]
                    const scoreVal = scoreInfo?.score ?? displayScore(site)
                    const isSelected = selectedSite && siteId(selectedSite) === id

                    return (
                      <button
                        key={id}
                        type="button"
                        data-site-id={id}
                        onClick={() => setSelectedId(id)}
                        className={`flex w-full items-center gap-3 rounded-lg px-3 py-3 text-left transition-colors ${isSelected ? 'bg-cyan-400/10 ring-1 ring-inset ring-cyan-300/25' : 'hover:bg-white/[0.04]'}`}
                      >
                        <span className={`w-6 text-center text-xs font-semibold ${index < 3 ? 'text-cyan-300' : 'text-slate-600'}`}>
                          {String(index + 1).padStart(2, '0')}
                        </span>
                        <span className="min-w-0 flex-1">
                          <span className="block truncate text-sm font-medium text-slate-200">{displayName(site)}</span>
                          <span className="mt-0.5 block text-[11px] text-slate-500">{displayPlace(site)}</span>
                        </span>
                        <span className="text-right">
                          <span className="block text-sm font-semibold tabular-nums text-slate-100">
                            {scoreVal === null || scoreVal === undefined ? '—' : Number(scoreVal).toFixed(3)}
                          </span>
                        </span>
                      </button>
                    )
                  })}
                </div>
              </div>

              {/* Maplibre Map View */}
              <div className="relative min-h-[390px] overflow-hidden rounded-xl border border-white/[0.09] bg-[#11151d]">
                <SitesMap
                  sites={sortedSites}
                  selectedSiteId={selectedSite ? siteId(selectedSite) : null}
                  onSelectSite={setSelectedId}
                  catchment={catchment}
                />
                <div className="pointer-events-none absolute inset-x-0 top-0 z-10 flex items-start justify-between p-5">
                  <div>
                    <h3 className="text-sm font-semibold">Market map</h3>
                    <p className="mt-1 text-xs text-slate-500">Greater Chennai pilot · OSM tiles</p>
                  </div>
                  <span className="rounded-md border border-white/10 bg-[#11151d]/80 px-2 py-1 text-[10px] text-slate-500">MAPLIBRE / OSM</span>
                </div>
                <div className="pointer-events-none absolute bottom-4 left-5 right-5 z-10 flex items-center justify-between border-t border-white/[0.07] pt-3 text-[11px] text-slate-500">
                  <span>
                    Selected:{' '}
                    <strong className="font-medium text-slate-300">
                      {selectedSite ? displayName(selectedSite) : 'None'}
                    </strong>
                  </span>
                  <span>{catchment ? '10-min catchment' : catchmentNote ? 'Catchment: enhancement' : 'Pilot zoom'}</span>
                </div>
              </div>
            </div>

            {/* Score Decomposition Panel for Selected Site (File 53 Integration) */}
            <ScoreDecompositionPanel
              siteId={selectedSite ? siteId(selectedSite) : null}
              weights={weights}
              method={method}
              className="mt-5"
            />
          </section>
        </div>
      )}
    </main>
  )
}
