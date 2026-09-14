'use client'

import React, { useState, useMemo } from 'react'
import { Filter, ChevronDown, ChevronUp, AlertCircle, Building2, RotateCcw } from 'lucide-react'
import { apiUrl } from '@/lib/api'

export interface ExcludedSite {
  site_identifier: string | number
  reasons: string[]
}

export interface FilterSite {
  site_id: string | number
  name?: string
  unit_size?: number
  rent?: number
  tenure?: string
  status?: string
  attractiveness?: number
  coordinates?: {
    latitude: number
    longitude: number
  }
}

export interface SitesApiResponse {
  count: number
  filters_applied: Record<string, unknown>
  sites: FilterSite[]
  excluded: ExcludedSite[]
}

export interface ConstraintFilterPanelProps {
  availableTenureTypes?: string[]
  onFilterChange?: (filteredData: SitesApiResponse) => void
  onResetFilters?: () => void
  apiEndpoint?: string
  className?: string
}

export function ConstraintFilterPanel({
  availableTenureTypes = ['lease', 'freehold', 'licence'],
  onFilterChange,
  onResetFilters,
  apiEndpoint = '/sites',
  className = '',
}: ConstraintFilterPanelProps) {
  // Input States
  const [minUnitSize, setMinUnitSize] = useState<string>('')
  const [maxRent, setMaxRent] = useState<string>('')
  const [selectedTenures, setSelectedTenures] = useState<string[]>([])

  // Response / UI States
  const [isLoading, setIsLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)
  const [excludedSites, setExcludedSites] = useState<ExcludedSite[]>([])
  const [showExclusionDetails, setShowExclusionDetails] = useState<boolean>(false)
  const [lastAppliedCount, setLastAppliedCount] = useState<number | null>(null)

  // Toggle tenure checkbox state
  const handleTenureToggle = (tenure: string) => {
    setSelectedTenures((prev) =>
      prev.includes(tenure) ? prev.filter((t) => t !== tenure) : [...prev, tenure]
    )
  }

  // Build GET query string matching File 43 specs
  const queryParams = useMemo(() => {
    const params = new URLSearchParams()
    if (minUnitSize.trim()) params.append('min_unit_size', minUnitSize.trim())
    if (maxRent.trim()) params.append('max_rent', maxRent.trim())
    if (selectedTenures.length > 0) {
      params.append('required_tenure', selectedTenures.join(','))
    }
    return params.toString()
  }, [minUnitSize, maxRent, selectedTenures])

  // Execute Filter GET /sites request
  const handleApplyFilters = async (e?: React.FormEvent) => {
    if (e) e.preventDefault()
    setIsLoading(true)
    setError(null)

    try {
      const endpoint = apiEndpoint.startsWith('/') ? apiEndpoint : `/${apiEndpoint}`
      const url = queryParams ? apiUrl(`${endpoint}?${queryParams}`) : apiUrl(endpoint)
      const response = await fetch(url, {
        method: 'GET',
        headers: {
          Accept: 'application/json',
        },
      })

      if (!response.ok) {
        let errorDetail = `Server returned HTTP ${response.status}`
        try {
          const errorData = await response.json()
          if (errorData.detail) errorDetail = typeof errorData.detail === 'string' ? errorData.detail : JSON.stringify(errorData.detail)
        } catch {
          // keep fallback
        }
        throw new Error(errorDetail)
      }

      const data: SitesApiResponse = await response.json()
      setExcludedSites(data.excluded || [])
      setLastAppliedCount(data.count ?? data.sites?.length ?? 0)
      if (onFilterChange) {
        onFilterChange(data)
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to fetch candidate sites.')
    } finally {
      setIsLoading(false)
    }
  }

  // Reset filter controls
  const handleReset = () => {
    setMinUnitSize('')
    setMaxRent('')
    setSelectedTenures([])
    setError(null)
    setExcludedSites([])
    setLastAppliedCount(null)
    if (onResetFilters) {
      onResetFilters()
    }
  }

  // Compute total reasons summary for File 26 output
  const totalExclusionReasonsCount = useMemo(() => {
    return excludedSites.reduce((acc, curr) => acc + (curr.reasons?.length || 0), 0)
  }, [excludedSites])

  return (
    <div className={`w-full rounded-xl border border-white/[0.09] bg-[#11151d] text-slate-100 shadow-sm ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between border-b border-white/[0.08] px-4 py-3">
        <div className="flex items-center space-x-2">
          <Filter className="size-4 text-cyan-400" />
          <h3 className="text-sm font-semibold tracking-tight text-slate-100">Site Constraints</h3>
        </div>
        <button
          type="button"
          onClick={handleReset}
          className="flex items-center gap-1 text-xs text-slate-400 transition-colors hover:text-slate-200"
        >
          <RotateCcw className="size-3" />
          Reset
        </button>
      </div>

      {/* Filter Form */}
      <form onSubmit={handleApplyFilters} className="space-y-4 p-4">
        {/* Min Unit Size */}
        <div>
          <label className="mb-1 block text-xs font-medium text-slate-400">
            Minimum Unit Size (sq ft)
          </label>
          <input
            type="number"
            min="0"
            placeholder="e.g. 1500"
            value={minUnitSize}
            onChange={(e) => setMinUnitSize(e.target.value)}
            className="w-full rounded-md border border-white/10 bg-white/5 px-3 py-2 text-sm text-slate-100 placeholder-slate-500 focus:border-cyan-400 focus:outline-none focus:ring-1 focus:ring-cyan-400"
          />
        </div>

        {/* Max Rent */}
        <div>
          <label className="mb-1 block text-xs font-medium text-slate-400">
            Maximum Rent Cap (£/yr)
          </label>
          <input
            type="number"
            min="0"
            placeholder="e.g. 250000"
            value={maxRent}
            onChange={(e) => setMaxRent(e.target.value)}
            className="w-full rounded-md border border-white/10 bg-white/5 px-3 py-2 text-sm text-slate-100 placeholder-slate-500 focus:border-cyan-400 focus:outline-none focus:ring-1 focus:ring-cyan-400"
          />
        </div>

        {/* Tenure Multi-Select */}
        <div>
          <label className="mb-1.5 block text-xs font-medium text-slate-400">
            Tenure Types
          </label>
          <div className="space-y-1.5 rounded-md border border-white/10 bg-white/[0.03] p-2.5">
            {availableTenureTypes.map((tenure) => {
              const isChecked = selectedTenures.includes(tenure)
              return (
                <label
                  key={tenure}
                  className="flex cursor-pointer select-none items-center space-x-2 text-xs capitalize text-slate-300 hover:text-white"
                >
                  <input
                    type="checkbox"
                    checked={isChecked}
                    onChange={() => handleTenureToggle(tenure)}
                    className="size-3.5 rounded border-white/20 bg-white/5 text-cyan-500 focus:ring-cyan-400 focus:ring-offset-0"
                  />
                  <span>{tenure}</span>
                </label>
              )
            })}
          </div>
        </div>

        {/* Validation Error Message */}
        {error && (
          <div className="flex items-start space-x-2 rounded-md border border-rose-500/30 bg-rose-500/10 p-2.5 text-xs text-rose-300">
            <AlertCircle className="mt-0.5 size-4 shrink-0 text-rose-400" />
            <span>{error}</span>
          </div>
        )}

        {/* Submit Button */}
        <button
          type="submit"
          disabled={isLoading}
          className="flex w-full items-center justify-center space-x-1.5 rounded-md bg-cyan-500 px-4 py-2 text-xs font-semibold text-slate-950 shadow-sm transition-colors hover:bg-cyan-400 disabled:opacity-50"
        >
          {isLoading ? (
            <span>Applying Filters...</span>
          ) : (
            <span>Apply Constraints {lastAppliedCount !== null ? `(${lastAppliedCount} sites)` : ''}</span>
          )}
        </button>
      </form>

      {/* Exclusion Summary Panel */}
      <div className="rounded-b-xl border-t border-white/[0.08] bg-white/[0.02] p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <span className="inline-flex items-center justify-center rounded-full bg-rose-400/20 px-2 py-0.5 text-xs font-semibold text-rose-300">
              {excludedSites.length}
            </span>
            <span className="text-xs font-medium text-slate-400">
              Sites Excluded
            </span>
          </div>

          {excludedSites.length > 0 && (
            <button
              type="button"
              onClick={() => setShowExclusionDetails(!showExclusionDetails)}
              className="flex items-center space-x-1 text-xs font-medium text-cyan-400 hover:text-cyan-300"
            >
              <span>{showExclusionDetails ? 'Hide' : 'Show Reasons'}</span>
              {showExclusionDetails ? (
                <ChevronUp className="size-3" />
              ) : (
                <ChevronDown className="size-3" />
              )}
            </button>
          )}
        </div>

        {/* Detailed Reasons Breakdown */}
        {showExclusionDetails && excludedSites.length > 0 && (
          <div className="mt-3 max-h-48 space-y-2 overflow-y-auto pr-1">
            <div className="text-[11px] text-slate-400">
              {totalExclusionReasonsCount} restriction violations logged:
            </div>
            {excludedSites.map((site) => (
              <div
                key={site.site_identifier}
                className="space-y-1 rounded border border-white/10 bg-white/[0.03] p-2 text-xs"
              >
                <div className="flex items-center space-x-1 font-semibold text-slate-300">
                  <Building2 className="size-3 text-slate-500" />
                  <span>Site ID: {site.site_identifier}</span>
                </div>
                <ul className="list-disc space-y-0.5 pl-4 text-[11px] text-rose-400">
                  {site.reasons.map((reason, idx) => (
                    <li key={idx}>{reason}</li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

export default ConstraintFilterPanel
