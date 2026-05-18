import { useState } from 'react'
import api from '@/lib/api'
import { getLogger } from '@/lib/logger'

const logger = getLogger('useChat')

export interface Citation {
  filename: string
  page: string | number
  chunk_id: string | number
  excerpt: string
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: string[]
  citations?: Citation[]
  searchType?: string
  domain?: string
  rawData?: Record<string, unknown>
}

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function sendMessage(query: string, responseStyle: 'detailed' | 'brief' = 'detailed') {
    setError(null)
    const userMsg: Message = { id: crypto.randomUUID(), role: 'user', content: query }
    setMessages(prev => [...prev, userMsg])
    setLoading(true)

    logger.info('Sending query', { queryLength: query.length, responseStyle })

    try {
      const { data } = await api.post('/query', { user_query: query, response_style: responseStyle })

      logger.info('Query response received', {
        domain: data.domain,
        confidence: data.confidence,
        numResults: data.results?.length ?? 0,
        searchType: data.metadata?.search_type,
      })

      const assistantMsg: Message = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: data.summary ?? 'No response.',
        sources: data.metadata?.source_files ?? [],
        citations: data.citations ?? [],
        searchType: data.metadata?.search_type,
        domain: data.domain,
        rawData: data,
      }
      setMessages(prev => [...prev, assistantMsg])
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Request failed.'
      logger.error('Query failed', { error: message, query: query.slice(0, 80) })
      setError(message)
    } finally {
      setLoading(false)
    }
  }

  function clearMessages() {
    logger.debug('Clearing chat messages')
    setMessages([])
  }

  return { messages, loading, error, sendMessage, clearMessages }
}
