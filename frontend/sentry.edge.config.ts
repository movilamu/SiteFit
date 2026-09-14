import * as Sentry from '@sentry/nextjs'

Sentry.init({
  dsn: process.env.SENTRY_DSN || process.env.NEXT_PUBLIC_SENTRY_DSN,
  tracesSampleRate: 1.0,
  // Demographic data privacy: do not capture PII by default
  sendDefaultPii: false,
  beforeSend(event) {
    if (event.user) {
      delete event.user.ip_address
      delete event.user.email
      delete event.user.username
    }
    return event
  },
})
