import { useState } from 'react'
import api from '@/lib/api'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: string[]
  searchType?: string
}

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function sendMessage(query: string) {
    setError(null)
    const userMsg: Message = { id: crypto.randomUUID(), role: 'user', content: query }
    setMessages(prev => [...prev, userMsg])
    setLoading(true)

    try {
      const { data } = await api.post('/query', { user_query: query })
      const assistantMsg: Message = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: data.summary ?? data.answer ?? 'No response.',
        sources: data.metadata?.source_files ?? [],
        searchType: data.metadata?.search_type,
      }
      setMessages(prev => [...prev, assistantMsg])
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Request failed.'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  function clearMessages() { setMessages([]) }

  return { messages, loading, error, sendMessage, clearMessages }
}
