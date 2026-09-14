'use client'

import React, { useCallback, useMemo, useState } from 'react'
import { Button } from '@/components/ui/button'
import { apiUrl, readApiError } from '@/lib/api'
import { normalizeWeights } from '@/lib/weights'

// ---------------------------------------------------------------------------
// Types matching File 47 (/api/export_routes.py) and File 46 (scenario_routes)
// ---------------------------------------------------------------------------

export interface ExportRequest {
  scenario_id?: number | null
  weight_set?: Record<string, number> | null
  aggregation_method: string
}

export interface ExportResponse {
  export_id: string | number
  scenario_id: number | null
  format: string
  storage_path: string
  download_url: string
  weight_set: Record<string, number>
  aggregation_method: string
  row_count: number
}

export interface ExportHistoryItem {
  id: string
  createdAt: string
  source: 'scenario' | 'inline'
  scenarioName?: string
  response: ExportResponse
}

export interface ExportPanelProps {
  /** The current active weight set (e.g. { grid_distance: 0.4, slope: 0.6 }) */
  weights?: Record<string, number>
  /** Active saved scenario ID, if exporting from a saved scenario */
  scenarioId?: number | null
  /** Human-readable scenario name if exporting from a saved scenario */
  scenarioName?: string | null
  /** Active aggregation method (default: 'weighted_sum') */
  method?: string
  /** Base URL for API requests (default: '') */
  apiBaseUrl?: string
  /** Custom endpoint route (default: '/export', fallback checks '/api/export') */
  exportEndpoint?: string
  /** Callback triggered when export succeeds */
  onExportSuccess?: (response: ExportResponse) => void
  /** Callback triggered when export fails */
  onExportError?: (error: Error) => void
  /** Additional custom class names */
  className?: string
}

// ---------------------------------------------------------------------------
// Method formatting & Helpers
// ---------------------------------------------------------------------------

const SUPPORTED_METHODS: Record<string, string> = {
  weighted_sum: 'Weighted Sum (SAW)',
  weighted_product: 'Weighted Product (WPM)',
  distance_to_ideal: 'Distance to Ideal (TOPSIS)',
}

function formatMethodName(method: string): string {
  const normalized = (method || 'weighted_sum').trim().toLowerCase().replace(/[- ]/g, '_')
  return SUPPORTED_METHODS[normalized] ?? method.replace(/[_-]+/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}

function formatCriterionName(key: string): string {
  return key
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase())
}

function formatWeightPercent(val: number): string {
  return `${(val * 100).toFixed(1)}%`
}

function formatWeightDecimal(val: number): string {
  return Number.isFinite(val) ? val.toFixed(4) : '—'
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
      second: '2-digit',
    })
  } catch {
    return dateStr
  }
}

// ---------------------------------------------------------------------------
// Self-Contained SVG Icons (no external icon package dependency)
// ---------------------------------------------------------------------------

function DownloadIcon({ className = 'size-4' }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2" />
      <path strokeLinecap="round" strokeLinejoin="round" d="M7 10l5 5 5-5m-5 5V3" />
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

function CopyIcon({ className = 'size-4' }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
      />
    </svg>
  )
}

function FileJsonIcon({ className = 'size-4' }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
      />
    </svg>
  )
}

function ScaleIcon({ className = 'size-4' }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M3 6l3 1m0 0l-3 9a5.002 5.002 0 006.001 0M6 7l3 9M6 7l6-2m6 2l3-1m-3 1l-3 9a5.002 5.002 0 006.001 0M18 7l3 9m-3-9l-6-2m0-2v2m0 16V5m0 16H9m3 0h3"
      />
    </svg>
  )
}

function ShieldCheckIcon({ className = 'size-4' }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
      />
    </svg>
  )
}

function AlertTriangleIcon({ className = 'size-4' }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
      />
    </svg>
  )
}

function ExternalLinkIcon({ className = 'size-4' }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
    </svg>
  )
}

function RefreshCwIcon({ className = 'size-4' }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
      />
    </svg>
  )
}

function SpinnerIcon({ className = 'size-4' }: { className?: string }) {
  return (
    <svg className={`${className} animate-spin`} fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
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

function HistoryIcon({ className = 'size-4' }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  )
}

// ---------------------------------------------------------------------------
// Main Export Panel Component
// ---------------------------------------------------------------------------

export default function ExportPanel({
  weights = {},
  scenarioId = null,
  scenarioName = null,
  method = 'weighted_sum',
  apiBaseUrl = '',
  exportEndpoint = '/export',
  onExportSuccess,
  onExportError,
  className = '',
}: ExportPanelProps) {
  // Mode selection: 'scenario' (when scenarioId provided) or 'inline'
  const [exportSource, setExportSource] = useState<'scenario' | 'inline'>(
    scenarioId != null ? 'scenario' : 'inline'
  )

  // Request & Status state
  const [isExporting, setIsExporting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [copiedLink, setCopiedLink] = useState(false)
  const [copiedWeights, setCopiedWeights] = useState(false)
  const [showJsonPreview, setShowJsonPreview] = useState(false)
  const [showHistory, setShowHistory] = useState(false)

  // Successful exports state
  const [latestExport, setLatestExport] = useState<ExportResponse | null>(null)
  const [exportHistory, setExportHistory] = useState<ExportHistoryItem[]>([])

  // Local weights state for optional auto-normalization
  const [localWeights, setLocalWeights] = useState<Record<string, number>>(weights)

  // Synchronize when incoming weights change
  React.useEffect(() => {
    setLocalWeights(weights)
  }, [weights])

  // Update export source if scenarioId changes
  React.useEffect(() => {
    if (scenarioId != null) {
      setExportSource('scenario')
    } else {
      setExportSource('inline')
    }
  }, [scenarioId])

  // Clean and validate current inline weights
  const currentWeightEntries = useMemo(() => {
    return Object.entries(localWeights).filter(([k]) => k.trim().length > 0)
  }, [localWeights])

  const currentTotalWeight = useMemo(() => {
    return currentWeightEntries.reduce((sum, [, w]) => sum + (Number(w) || 0), 0)
  }, [currentWeightEntries])

  // File 47 validator requires: abs(total - 1.0) <= 1e-6
  const isInlineSumValid = useMemo(() => {
    return Math.abs(currentTotalWeight - 1.0) <= 1e-5
  }, [currentTotalWeight])

  const hasValidCriteria = currentWeightEntries.length > 0

  // Can the user initiate an export?
  const canExport = useMemo(() => {
    if (isExporting) return false
    if (exportSource === 'scenario') {
      return typeof scenarioId === 'number' && scenarioId > 0
    }
    return hasValidCriteria && isInlineSumValid
  }, [isExporting, exportSource, scenarioId, hasValidCriteria, isInlineSumValid])

  // Helper: auto-normalize inline weights to sum exactly to 1.0
  const handleAutoNormalize = useCallback(() => {
    if (currentTotalWeight <= 0) return
    const normalized: Record<string, number> = {}
    const entries = Object.entries(localWeights)
    let accumulated = 0

    entries.forEach(([key, val], idx) => {
      if (idx === entries.length - 1) {
        // Ensure final entry absorbs any tiny floating point rounding
        normalized[key] = parseFloat((1.0 - accumulated).toFixed(6))
      } else {
        const norm = parseFloat(((Number(val) || 0) / currentTotalWeight).toFixed(6))
        normalized[key] = norm
        accumulated += norm
      }
    })
    setLocalWeights(normalized)
    setError(null)
  }, [currentTotalWeight, localWeights])

  // -------------------------------------------------------------------------
  // Execute POST /export (File 47)
  // -------------------------------------------------------------------------

  const handleExportShortlist = useCallback(async () => {
    setIsExporting(true)
    setError(null)
    setCopiedLink(false)

    // Normalize method to match File 47 supported methods
    const normalizedMethod = (method || 'weighted_sum')
      .trim()
      .toLowerCase()
      .replace(/[- ]/g, '_')

    // Construct strictly compliant payload for File 47 ExportRequest
    // Note: ExportRequest uses extra="forbid" and requires EXACTLY ONE of
    // scenario_id OR weight_set.
    let payload: Record<string, unknown>

    if (exportSource === 'scenario' && scenarioId != null) {
      payload = {
        scenario_id: Number(scenarioId),
        aggregation_method: normalizedMethod,
      }
    } else {
      // Inline weights mapping — normalize to strictly satisfy File 47 validator
      const cleanedWeights: Record<string, number> = {}
      for (const [key, rawVal] of Object.entries(localWeights)) {
        const cleanKey = key.trim()
        if (!cleanKey) continue
        cleanedWeights[cleanKey] = Number(rawVal)
      }

      payload = {
        weight_set: normalizeWeights(cleanedWeights),
        aggregation_method: normalizedMethod,
      }
    }

    // Build URL candidates to accommodate varying base URLs / mounting paths
    const cleanEndpoint = exportEndpoint.startsWith('/') ? exportEndpoint : `/${exportEndpoint}`
    const primaryUrl = apiBaseUrl
      ? `${apiBaseUrl.replace(/\/+$/, '')}${cleanEndpoint}`
      : apiUrl(cleanEndpoint)

    try {
      let response = await fetch(primaryUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json',
        },
        body: JSON.stringify(payload),
      })

      // Fallback: If 404 on '/export', attempt '/api/export' or vice versa
      if (response.status === 404 && !cleanEndpoint.includes('/api/')) {
        const fallbackUrl = apiBaseUrl
          ? `${apiBaseUrl.replace(/\/+$/, '')}/api/export`
          : apiUrl('/api/export')
        const retryRes = await fetch(fallbackUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Accept: 'application/json',
          },
          body: JSON.stringify(payload),
        })
        if (retryRes.ok || retryRes.status !== 404) {
          response = retryRes
        }
      }

      if (!response.ok) {
        throw new Error(await readApiError(response, `Export failed (HTTP ${response.status})`))
      }

      const data = (await response.json()) as ExportResponse
      setLatestExport(data)

      // Add to session history
      const historyEntry: ExportHistoryItem = {
        id: String(data.export_id || Date.now()),
        createdAt: new Date().toISOString(),
        source: exportSource,
        scenarioName: scenarioName ?? (data.scenario_id ? `Scenario #${data.scenario_id}` : undefined),
        response: data,
      }
      setExportHistory((prev) => [historyEntry, ...prev.slice(0, 9)])

      if (onExportSuccess) {
        onExportSuccess(data)
      }
    } catch (err: unknown) {
      const parsedError = err instanceof Error ? err : new Error(String(err))
      setError(parsedError.message)
      if (onExportError) {
        onExportError(parsedError)
      }
    } finally {
      setIsExporting(false)
    }
  }, [
    exportSource,
    scenarioId,
    scenarioName,
    method,
    localWeights,
    apiBaseUrl,
    exportEndpoint,
    onExportSuccess,
    onExportError,
  ])

  // Copy link to clipboard
  const handleCopyLink = useCallback(async (url: string) => {
    try {
      await navigator.clipboard.writeText(url)
      setCopiedLink(true)
      setTimeout(() => setCopiedLink(false), 2500)
    } catch {
      // Fallback
    }
  }, [])

  // Copy weight set JSON to clipboard
  const handleCopyWeightsJson = useCallback(async (weightsObj: Record<string, number>) => {
    try {
      await navigator.clipboard.writeText(JSON.stringify(weightsObj, null, 2))
      setCopiedWeights(true)
      setTimeout(() => setCopiedWeights(false), 2500)
    } catch {
      // Fallback
    }
  }, [])

  // Direct trigger browser download from URL
  const handleDirectDownload = useCallback((url: string, filename?: string) => {
    const link = document.createElement('a')
    link.href = url
    link.target = '_blank'
    link.rel = 'noopener noreferrer'
    if (filename) {
      link.download = filename
    }
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }, [])

  return (
    <div className={`space-y-4 rounded-xl border border-border bg-card p-5 text-card-foreground shadow-sm ${className}`}>
      {/* ------------------------------------------------------------------- */}
      {/* Header & Description */}
      {/* ------------------------------------------------------------------- */}
      <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between border-b border-border/60 pb-3">
        <div className="flex items-center gap-2.5">
          <div className="flex size-9 items-center justify-center rounded-lg bg-primary/10 text-primary">
            <FileJsonIcon className="size-5" />
          </div>
          <div>
            <h3 className="text-base font-semibold leading-none tracking-tight">
              Export Ranked Shortlist
            </h3>
            <p className="mt-1 text-xs text-muted-foreground">
              Generate a self-describing, reproducible JSON artifact via File 47 (<code className="font-mono text-[11px]">POST /export</code>).
            </p>
          </div>
        </div>

        {/* Export History Toggle (if any past exports exist) */}
        {exportHistory.length > 0 && (
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={() => setShowHistory((prev) => !prev)}
            className="text-xs text-muted-foreground hover:text-foreground"
          >
            <HistoryIcon className="size-3.5" />
            <span>History ({exportHistory.length})</span>
            <ChevronDownIcon className={`size-3 transition-transform ${showHistory ? 'rotate-180' : ''}`} />
          </Button>
        )}
      </div>

      {/* ------------------------------------------------------------------- */}
      {/* Configuration & Pre-Export Controls */}
      {/* ------------------------------------------------------------------- */}
      <div className="rounded-lg border border-border/60 bg-muted/20 p-3.5 space-y-3">
        {/* Source Mode Toggle if scenarioId is available */}
        {scenarioId != null && (
          <div className="flex items-center justify-between text-xs border-b border-border/40 pb-2.5">
            <span className="font-medium text-foreground">Export Mode</span>
            <div className="flex items-center gap-1 rounded-md bg-muted p-0.5">
              <button
                type="button"
                onClick={() => setExportSource('scenario')}
                className={`rounded px-2.5 py-1 text-xs font-medium transition-colors ${
                  exportSource === 'scenario'
                    ? 'bg-background text-foreground shadow-xs'
                    : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                Saved Scenario #{scenarioId}
              </button>
              <button
                type="button"
                onClick={() => setExportSource('inline')}
                className={`rounded px-2.5 py-1 text-xs font-medium transition-colors ${
                  exportSource === 'inline'
                    ? 'bg-background text-foreground shadow-xs'
                    : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                Inline Active Weights
              </button>
            </div>
          </div>
        )}

        {/* Selected Parameters Overview */}
        <div className="grid grid-cols-1 gap-2 text-xs sm:grid-cols-3">
          <div className="rounded-md border border-border/40 bg-background/60 p-2">
            <span className="text-muted-foreground">Source:</span>{' '}
            <span className="font-semibold text-foreground">
              {exportSource === 'scenario'
                ? scenarioName
                  ? `${scenarioName} (ID: ${scenarioId})`
                  : `Scenario #${scenarioId}`
                : 'Current Inline Weights'}
            </span>
          </div>
          <div className="rounded-md border border-border/40 bg-background/60 p-2">
            <span className="text-muted-foreground">Method:</span>{' '}
            <span className="font-semibold text-foreground">{formatMethodName(method)}</span>
          </div>
          <div className="rounded-md border border-border/40 bg-background/60 p-2">
            <span className="text-muted-foreground">Weight Sum:</span>{' '}
            <span
              className={`font-semibold ${
                exportSource === 'scenario' || isInlineSumValid
                  ? 'text-emerald-600 dark:text-emerald-400'
                  : 'text-amber-600 dark:text-amber-400'
              }`}
            >
              {exportSource === 'scenario' ? '1.0000 (Persisted)' : `${currentTotalWeight.toFixed(4)} ${isInlineSumValid ? '✓' : '(Sum ≠ 1.0)'}`}
            </span>
          </div>
        </div>

        {/* Sum Warning & Auto-Normalize Helper (when inline weights don't sum to 1.0) */}
        {exportSource === 'inline' && !isInlineSumValid && hasValidCriteria && (
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-800 dark:text-amber-200">
            <div className="flex items-center gap-2">
              <AlertTriangleIcon className="size-4 shrink-0 text-amber-600 dark:text-amber-400" />
              <span>
                File 47 strictly requires weights to sum to <strong>1.0</strong> (current sum is{' '}
                {currentTotalWeight.toFixed(4)}).
              </span>
            </div>
            <Button
              type="button"
              variant="outline"
              size="xs"
              onClick={handleAutoNormalize}
              className="border-amber-500/40 text-amber-900 hover:bg-amber-500/20 dark:text-amber-100 shrink-0"
            >
              <RefreshCwIcon className="size-3" />
              <span>Normalize to 1.0</span>
            </Button>
          </div>
        )}

        {/* Error Notification */}
        {error && (
          <div className="rounded-md border border-destructive/30 bg-destructive/10 p-3 text-xs text-destructive">
            <div className="flex items-start gap-2">
              <AlertTriangleIcon className="size-4 shrink-0 mt-0.5" />
              <div className="flex-1">
                <span className="font-medium">Export generation failed:</span> {error}
              </div>
            </div>
          </div>
        )}

        {/* 1. "Export shortlist" Action Button */}
        <div className="flex items-center justify-end pt-1">
          <Button
            type="button"
            variant="default"
            size="default"
            disabled={!canExport}
            onClick={handleExportShortlist}
            className="w-full sm:w-auto font-medium"
          >
            {isExporting ? (
              <>
                <SpinnerIcon className="size-4" />
                <span>Generating Export...</span>
              </>
            ) : (
              <>
                <DownloadIcon className="size-4" />
                <span>Export shortlist</span>
              </>
            )}
          </Button>
        </div>
      </div>

      {/* ------------------------------------------------------------------- */}
      {/* Session Export History Dropdown (Collapsible) */}
      {/* ------------------------------------------------------------------- */}
      {showHistory && exportHistory.length > 0 && (
        <div className="space-y-2 rounded-lg border border-border bg-muted/10 p-3">
          <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
            Recent Session Exports ({exportHistory.length})
          </div>
          <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
            {exportHistory.map((item) => (
              <div
                key={item.id}
                className="flex items-center justify-between rounded-md border border-border/50 bg-background px-3 py-2 text-xs"
              >
                <div className="flex flex-col">
                  <span className="font-medium text-foreground">
                    {item.scenarioName ?? 'Inline Weight Set'}
                  </span>
                  <span className="text-[11px] text-muted-foreground">
                    {formatDate(item.createdAt)} • {item.response.row_count} sites • {formatMethodName(item.response.aggregation_method)}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    type="button"
                    variant="outline"
                    size="xs"
                    onClick={() => setLatestExport(item.response)}
                  >
                    View
                  </Button>
                  <Button
                    type="button"
                    variant="ghost"
                    size="xs"
                    onClick={() => handleDirectDownload(item.response.download_url)}
                  >
                    <DownloadIcon className="size-3" />
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ------------------------------------------------------------------- */}
      {/* 2 & 3. Export Result with Download Link and Weight Set Display      */}
      {/* ------------------------------------------------------------------- */}
      {latestExport && (
        <div className="space-y-4 rounded-xl border border-emerald-500/30 bg-emerald-500/5 p-4 dark:border-emerald-500/20 dark:bg-emerald-950/10">
          {/* Status Header */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-emerald-500/20 pb-3">
            <div className="flex items-center gap-2">
              <span className="flex size-6 items-center justify-center rounded-full bg-emerald-600 text-white dark:bg-emerald-500">
                <CheckIcon className="size-3.5" />
              </span>
              <div>
                <h4 className="text-sm font-semibold text-foreground">
                  Shortlist Export Ready
                </h4>
                <p className="text-[11px] text-muted-foreground">
                  Export #{latestExport.export_id} registered and archived in Supabase storage.
                </p>
              </div>
            </div>

            {/* Reproducibility Badge */}
            <div className="flex items-center gap-1.5 self-start sm:self-auto rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-0.5 text-[11px] font-medium text-emerald-700 dark:text-emerald-300">
              <ShieldCheckIcon className="size-3.5" />
              <span>Reproducible Artifact</span>
            </div>
          </div>

          {/* 2. Download Button & Link Controls */}
          <div className="flex flex-wrap items-center gap-2.5 pt-1">
            {/* Primary Download Button */}
            <a
              href={latestExport.download_url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex h-8 items-center justify-center gap-1.5 rounded-lg bg-primary px-3 text-xs font-medium text-primary-foreground shadow-xs transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <DownloadIcon className="size-3.5" />
              <span>Download Shortlist JSON</span>
              <ExternalLinkIcon className="size-3 opacity-70" />
            </a>

            {/* Copy Signed URL Button */}
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => handleCopyLink(latestExport.download_url)}
              className="text-xs"
            >
              {copiedLink ? (
                <>
                  <CheckIcon className="size-3.5 text-emerald-600" />
                  <span>URL Copied!</span>
                </>
              ) : (
                <>
                  <CopyIcon className="size-3.5" />
                  <span>Copy Download URL</span>
                </>
              )}
            </Button>

            {/* Direct Trigger Download */}
            <Button
              type="button"
              variant="secondary"
              size="sm"
              onClick={() =>
                handleDirectDownload(
                  latestExport.download_url,
                  `shortlist_export_${latestExport.export_id}.json`
                )
              }
              className="text-xs"
            >
              Direct Save
            </Button>

            {/* Toggle Raw JSON Inspector */}
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() => setShowJsonPreview((prev) => !prev)}
              className="ml-auto text-xs text-muted-foreground"
            >
              <span>{showJsonPreview ? 'Hide Artifact Details' : 'View Artifact Details'}</span>
              <ChevronDownIcon className={`size-3 transition-transform ${showJsonPreview ? 'rotate-180' : ''}`} />
            </Button>
          </div>

          {/* Artifact Metadata Info Bar */}
          <div className="grid grid-cols-2 gap-2 text-[11px] sm:grid-cols-4 rounded-md border border-border/40 bg-background/50 p-2.5">
            <div>
              <span className="text-muted-foreground">Ranked Sites:</span>{' '}
              <span className="font-semibold text-foreground">{latestExport.row_count}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Storage Path:</span>{' '}
              <span className="font-mono text-foreground truncate block" title={latestExport.storage_path}>
                {latestExport.storage_path}
              </span>
            </div>
            <div>
              <span className="text-muted-foreground">Format:</span>{' '}
              <span className="uppercase font-semibold text-foreground">{latestExport.format}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Aggregation:</span>{' '}
              <span className="font-medium text-foreground">{formatMethodName(latestExport.aggregation_method)}</span>
            </div>
          </div>

          {/* --------------------------------------------------------------- */}
          {/* 3. Clearly show the weighting set that produced the export      */}
          {/* --------------------------------------------------------------- */}
          <div className="rounded-lg border border-border bg-background p-3.5 shadow-xs space-y-3">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-border/50 pb-2.5">
              <div className="flex items-center gap-2">
                <ScaleIcon className="size-4 text-primary" />
                <span className="text-xs font-semibold uppercase tracking-wider text-foreground">
                  Exact Weight Set Used for Scoring
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-[11px] text-muted-foreground">
                  Total Weight: <strong className="text-foreground">100% (1.0000)</strong>
                </span>
                <Button
                  type="button"
                  variant="outline"
                  size="xs"
                  onClick={() => handleCopyWeightsJson(latestExport.weight_set)}
                  className="text-[11px] h-6 px-2"
                >
                  {copiedWeights ? (
                    <>
                      <CheckIcon className="size-3 text-emerald-600" />
                      <span>Copied!</span>
                    </>
                  ) : (
                    <>
                      <CopyIcon className="size-3" />
                      <span>Copy Weights</span>
                    </>
                  )}
                </Button>
              </div>
            </div>

            {/* Weights Breakdown List & Visualization */}
            <div className="space-y-2">
              {Object.entries(latestExport.weight_set || {}).map(([criterion, weight]) => {
                const numWeight = Number(weight) || 0
                const percent = Math.min(Math.max(numWeight * 100, 0), 100)

                return (
                  <div key={criterion} className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-medium text-foreground">
                        {formatCriterionName(criterion)}
                      </span>
                      <div className="flex items-center gap-2 font-mono text-[11px]">
                        <span className="text-muted-foreground">
                          {formatWeightDecimal(numWeight)}
                        </span>
                        <span className="rounded bg-primary/10 px-1.5 py-0.5 font-semibold text-primary">
                          {formatWeightPercent(numWeight)}
                        </span>
                      </div>
                    </div>
                    {/* Visual proportion bar */}
                    <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
                      <div
                        className="h-full rounded-full bg-primary transition-all duration-300"
                        style={{ width: `${percent}%` }}
                      />
                    </div>
                  </div>
                )
              })}
            </div>

            <p className="text-[11px] text-muted-foreground italic">
              * This exact weight vector and scoring method are embedded directly in the JSON document. Any analyst loading this file can reproduce the exact rankings via File 32 scoring path.
            </p>
          </div>

          {/* Collapsible Raw JSON Preview */}
          {showJsonPreview && (
            <div className="rounded-lg border border-border/60 bg-muted/30 p-3 space-y-1.5">
              <div className="flex items-center justify-between text-xs font-semibold text-muted-foreground">
                <span>Export Metadata & Structure</span>
                <span className="font-mono text-[10px]">v1 JSON Schema</span>
              </div>
              <pre className="max-h-56 overflow-auto rounded bg-background p-2.5 font-mono text-[11px] text-foreground border border-border/40">
                {JSON.stringify(
                  {
                    export_id: latestExport.export_id,
                    storage_path: latestExport.storage_path,
                    download_url: latestExport.download_url,
                    aggregation_method: latestExport.aggregation_method,
                    scenario_id: latestExport.scenario_id,
                    row_count: latestExport.row_count,
                    weight_set: latestExport.weight_set,
                  },
                  null,
                  2
                )}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
