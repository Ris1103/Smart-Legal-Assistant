import { useState } from 'react'
import api from '@/lib/api'

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

    try {
      const { data } = await api.post('/query', { user_query: query, response_style: responseStyle })
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
      setError(err instanceof Error ? err.message : 'Request failed.')
    } finally {
      setLoading(false)
    }
  }

  function clearMessages() { setMessages([]) }

  return { messages, loading, error, sendMessage, clearMessages }
}
