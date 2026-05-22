import axios, { type AxiosError, type InternalAxiosRequestConfig } from 'axios'
import { getLogger } from '@/lib/logger'

const logger = getLogger('api')

interface TimedRequestConfig extends InternalAxiosRequestConfig {
  _startMs?: number
}

const api = axios.create({ baseURL: import.meta.env.VITE_API_URL ?? 'http://localhost:8000' })

// --- Request interceptor: log every outgoing call ---
api.interceptors.request.use((config: TimedRequestConfig) => {
  const method = config.method?.toUpperCase() ?? 'REQUEST'
  logger.debug(`${method} ${config.url}`, { params: config.params })
  config._startMs = Date.now()
  return config
})

// --- Response interceptor: log success and errors ---
api.interceptors.response.use(
  (response) => {
    const config = response.config as TimedRequestConfig
    const ms = config._startMs ? Date.now() - config._startMs : null
    const method = response.config.method?.toUpperCase() ?? 'REQUEST'
    logger.info(
      `${method} ${response.config.url} → ${response.status}${ms !== null ? ` (${ms}ms)` : ''}`,
    )
    return response
  },
  (error: AxiosError) => {
    const config = (error.config ?? {}) as Record<string, unknown>
    const ms = config._startMs ? Date.now() - (config._startMs as number) : null
    const method = (error.config?.method ?? 'REQUEST').toUpperCase()
    const url = error.config?.url ?? 'unknown'
    const status = error.response?.status ?? 'network error'

    if (error.response) {
      // Server responded with a non-2xx status
      const detail = (error.response.data as Record<string, unknown>)?.detail ?? error.message
      logger.error(
        `${method} ${url} → ${status}${ms !== null ? ` (${ms}ms)` : ''}: ${detail}`,
        error.response.data,
      )
    } else if (error.request) {
      // Request sent but no response received (network failure, timeout)
      logger.error(`${method} ${url} — no response received (network/timeout)`, {
        message: error.message,
      })
    } else {
      logger.error(`${method} ${url} — request setup error: ${error.message}`)
    }

    return Promise.reject(error)
  },
)

// --- Conversation API ---

export interface ConversationSummary {
  id: string
  title: string | null
  created_at: string
  message_count: number
}

export interface ConversationDetail {
  id: string
  title: string | null
  created_at: string
  messages: Array<{ role: string; content: string }>
}

export async function getConversations(): Promise<ConversationSummary[]> {
  const { data } = await api.get('/conversations')
  return data
}

export async function createConversation(title: string): Promise<{ id: string; title: string | null }> {
  const { data } = await api.post('/conversations', { title })
  return data
}

export async function getConversation(id: string): Promise<ConversationDetail> {
  const { data } = await api.get(`/conversations/${id}`)
  return data
}

export async function appendMessage(
  conversationId: string,
  role: 'user' | 'assistant',
  content: string,
): Promise<void> {
  await api.post(`/conversations/${conversationId}/messages`, { role, content })
}

export function setAuthToken(token: string | null) {
  if (token) {
    api.defaults.headers.common['Authorization'] = `Bearer ${token}`
    logger.debug('Auth token set')
  } else {
    delete api.defaults.headers.common['Authorization']
    logger.debug('Auth token cleared')
  }
}

export default api
