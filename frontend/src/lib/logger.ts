/**
 * Level-gated browser logger.
 *
 * Level is controlled by the VITE_LOG_LEVEL env variable:
 *   debug  → everything
 *   info   → info, warn, error
 *   warn   → warn, error
 *   error  → error only
 *
 * Defaults to "info" in development, "warn" in production.
 */

type Level = 'debug' | 'info' | 'warn' | 'error'

const LEVELS: Record<Level, number> = { debug: 0, info: 1, warn: 2, error: 3 }

function resolveLevel(): Level {
  const fromEnv = (import.meta.env.VITE_LOG_LEVEL ?? '').toLowerCase() as Level
  if (fromEnv in LEVELS) return fromEnv
  return import.meta.env.PROD ? 'warn' : 'info'
}

const activeLevel = LEVELS[resolveLevel()]

function shouldLog(level: Level): boolean {
  return LEVELS[level] >= activeLevel
}

function fmt(level: Level, module: string, message: string): string {
  const ts = new Date().toISOString()
  return `[${ts}] [${level.toUpperCase()}] [${module}] ${message}`
}

export interface Logger {
  debug(message: string, ...args: unknown[]): void
  info(message: string, ...args: unknown[]): void
  warn(message: string, ...args: unknown[]): void
  error(message: string, ...args: unknown[]): void
}

export function getLogger(module: string): Logger {
  return {
    debug(message, ...args) {
      if (shouldLog('debug')) console.debug(fmt('debug', module, message), ...args)
    },
    info(message, ...args) {
      if (shouldLog('info')) console.info(fmt('info', module, message), ...args)
    },
    warn(message, ...args) {
      if (shouldLog('warn')) console.warn(fmt('warn', module, message), ...args)
    },
    error(message, ...args) {
      if (shouldLog('error')) console.error(fmt('error', module, message), ...args)
    },
  }
}
