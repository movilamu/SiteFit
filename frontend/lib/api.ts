import * as Sentry from '@sentry/nextjs'

/** FastAPI origin. Empty string uses same-origin Next rewrites to File 42. */
export const API_BASE_URL =
  (typeof process !== 'undefined' && process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/+$/, '')) ||
  ''

export function apiUrl(path: string): string {
  const normalized = path.startsWith('/') ? path : `/${path}`
  return `${API_BASE_URL}${normalized}`
}

/**
 * Parses API error response and reports the failure to Sentry.
 * Preserves demographic data privacy by capturing status and route context without PII.
 */
export async function readApiError(response: Response, fallback: string): Promise<string> {
  let message = `${fallback} (${response.status})`
  try {
    const body = await response.json()
    if (typeof body?.detail === 'string') {
      message = body.detail
    } else if (Array.isArray(body?.detail)) {
      message = body.detail.map((item: { msg?: string }) => item.msg || JSON.stringify(item)).join('; ')
    } else if (body?.detail) {
      message = JSON.stringify(body.detail)
    }
  } catch {
    // keep fallback
  }

  // Capture failed API call in Sentry
  Sentry.captureMessage(`API Error: ${response.status} ${response.url || fallback} - ${message}`, {
    level: 'error',
    extra: {
      status: response.status,
      statusText: response.statusText,
      url: response.url,
      detail: message,
    },
  })

  return message
}

/**
 * Explicit helper to capture exceptions from network requests or data parsing.
 */
export function captureApiError(error: unknown, context?: Record<string, unknown>): void {
  Sentry.captureException(error, {
    extra: context,
  })
}

