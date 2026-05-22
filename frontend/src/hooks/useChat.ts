import { useState } from 'react'
import api, { createConversation, appendMessage, getConversation } from '@/lib/api'
import { getLogger } from '@/lib/logger'

const logger = getLogger('useChat')

export interface Citation {
  filename: string
  page: string | number
  chunk_id: string | number
  excerpt: string
  url?: string
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
  const [conversationId, setConversationId] = useState<string | null>(null)

  async function sendMessage(query: string, responseStyle: 'detailed' | 'brief' = 'detailed') {
    setError(null)
    const userMsg: Message = { id: crypto.randomUUID(), role: 'user', content: query }
    setMessages(prev => [...prev, userMsg])
    setLoading(true)

    logger.info('Sending query', { queryLength: query.length, responseStyle })

    try {
      // Create conversation on first message
      let convId = conversationId
      if (!convId) {
        try {
          const conv = await createConversation(query.slice(0, 60))
          convId = conv.id
          setConversationId(convId)
        } catch (e) {
          logger.warn('Could not create conversation, continuing without persistence', e)
        }
      }

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

      // Persist both turns
      if (convId) {
        try {
          await appendMessage(convId, 'user', query)
          await appendMessage(convId, 'assistant', assistantMsg.content)
        } catch (e) {
          logger.warn('Could not persist messages', e)
        }
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Request failed.'
      logger.error('Query failed', { error: message, query: query.slice(0, 80) })
      setError(message)
    } finally {
      setLoading(false)
    }
  }

  async function loadConversation(id: string) {
    try {
      const conv = await getConversation(id)
      setConversationId(id)
      setMessages(
        conv.messages.map(m => ({
          id: crypto.randomUUID(),
          role: m.role as 'user' | 'assistant',
          content: m.content,
        }))
      )
      setError(null)
    } catch (e) {
      logger.error('Failed to load conversation', e)
      setError('Failed to load conversation.')
    }
  }

  function newChat() {
    setMessages([])
    setConversationId(null)
    setError(null)
  }

  function clearMessages() {
    logger.debug('Clearing chat messages')
    newChat()
  }

  return { messages, loading, error, conversationId, sendMessage, loadConversation, newChat, clearMessages }
}
