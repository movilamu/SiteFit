'use client'

import { useId, useState } from 'react'

// ---------------------------------------------------------------------------
// Confidence tiers — mirrors the `confidence` score produced by File 16
// (areal_interpolation.interpolate_to_analysis_unit / _estimate_confidence),
// which is in [0, 1] and is threaded through as `data_confidence` on
// CriterionDecomposition-like objects.
// ---------------------------------------------------------------------------

export type ConfidenceTier = 'high' | 'medium' | 'low' | 'unknown'

export function confidenceTier(score: number | null | undefined): ConfidenceTier {
  if (score === null || score === undefined || Number.isNaN(score)) return 'unknown'
  if (score >= 0.85) return 'high'
  if (score >= 0.5) return 'medium'
  return 'low'
}

const TIER_STYLES: Record<ConfidenceTier, { dot: string; ring: string; label: string }> = {
  high: { dot: 'bg-emerald-500', ring: 'ring-emerald-500/30', label: 'High confidence' },
  medium: { dot: 'bg-amber-500', ring: 'ring-amber-500/30', label: 'Medium confidence' },
  low: { dot: 'bg-rose-500', ring: 'ring-rose-500/30', label: 'Low confidence' },
  unknown: { dot: 'bg-slate-300', ring: 'ring-slate-300/30', label: 'Confidence unknown' },
}

/**
 * Default human-readable rationale, used when the API response doesn't
 * supply a bespoke `confidence_explanation` string. Kept generic since we
 * only have the score on the client, not the interpolation internals
 * (contributing_sources / coverage_ratio / used_population_surface) that
 * produced it in File 16.
 */
function defaultExplanation(tier: ConfidenceTier, score: number | null | undefined): string {
  switch (tier) {
    case 'high':
      return 'This value maps closely to source geography with little or no interpolation required.'
    case 'medium':
      return 'This value required some interpolation from source geography (e.g. blending multiple source polygons or falling back to area-based weighting), so treat it as a reasonable estimate rather than an exact figure.'
    case 'low':
      return 'This value required significant interpolation from source geography — coverage was partial, many source polygons were blended, or no population surface was available. Treat it as a rough estimate.'
    default:
      return score === null || score === undefined
        ? 'No confidence score is available for this value.'
        : 'Confidence could not be categorized for this value.'
  }
}

export type ConfidenceIndicatorProps = {
  /** Confidence score in [0, 1], e.g. from CriterionDecomposition.data_confidence */
  score: number | null | undefined
  /** Optional server-provided rationale; falls back to a generic tier-based message */
  explanation?: string | null
  /** Optional label of the thing this confidence applies to, used for aria text */
  criterionLabel?: string
  className?: string
}

/**
 * Small colored-dot indicator. Click/tap or hover/focus to reveal a brief
 * explanation of why the value carries that confidence level. Works for
 * both mouse (hover) and touch/keyboard (tap/focus) interaction.
 */
export function ConfidenceIndicator({
  score,
  explanation,
  criterionLabel,
  className = '',
}: ConfidenceIndicatorProps) {
  const [open, setOpen] = useState(false)
  const tooltipId = useId()
  const tier = confidenceTier(score)
  const styles = TIER_STYLES[tier]
  const text = explanation?.trim() || defaultExplanation(tier, score)
  const scoreText = score === null || score === undefined || Number.isNaN(score) ? null : `${Math.round(score * 100)}%`

  return (
    <span className={`relative inline-flex ${className}`}>
      <button
        type="button"
        className={`inline-flex size-2.5 shrink-0 rounded-full ${styles.dot} ring-4 ${styles.ring} focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-1 focus-visible:ring-ring`}
        aria-describedby={tooltipId}
        aria-label={`${styles.label}${criterionLabel ? ` for ${criterionLabel}` : ''}${scoreText ? ` (${scoreText})` : ''}`}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onClick={(e) => {
          e.stopPropagation()
          setOpen((v) => !v)
        }}
      />
      {open && (
        <span
          role="tooltip"
          id={tooltipId}
          className="absolute bottom-full left-1/2 z-20 mb-1.5 w-56 -translate-x-1/2 rounded-md border border-border bg-popover px-2.5 py-2 text-xs leading-snug text-popover-foreground shadow-md"
        >
          <span className="mb-1 flex items-center gap-1.5 font-medium">
            <span className={`inline-flex size-2 rounded-full ${styles.dot}`} />
            {styles.label}
            {scoreText && <span className="font-normal text-muted-foreground">· {scoreText}</span>}
          </span>
          {text}
        </span>
      )}
    </span>
  )
}
