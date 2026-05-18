import { Component, type ErrorInfo, type ReactNode } from 'react'
import { getLogger } from '@/lib/logger'

const logger = getLogger('ErrorBoundary')

interface Props {
  children: ReactNode
  fallback?: ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
}

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, error: null }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    logger.error('Unhandled render error', { error: error.message, stack: info.componentStack })
  }

  render() {
    if (this.state.hasError) {
      return (
        this.props.fallback ?? (
          <div className="min-h-screen flex items-center justify-center bg-gray-50">
            <div className="text-center max-w-md px-6">
              <h2 className="text-lg font-semibold text-gray-800 mb-2">Something went wrong</h2>
              <p className="text-sm text-gray-500 mb-4">
                An unexpected error occurred. Please refresh the page.
              </p>
              <button
                onClick={() => window.location.reload()}
                className="px-4 py-2 bg-indigo-600 text-white text-sm rounded-lg hover:bg-indigo-700 transition-colors"
              >
                Reload page
              </button>
            </div>
          </div>
        )
      )
    }
    return this.props.children
  }
}
