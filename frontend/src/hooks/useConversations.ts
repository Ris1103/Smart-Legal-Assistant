import { useState, useEffect, useCallback } from 'react'
import { getConversations, type ConversationSummary } from '@/lib/api'
import { getLogger } from '@/lib/logger'

const logger = getLogger('useConversations')

export function useConversations() {
  const [conversations, setConversations] = useState<ConversationSummary[]>([])
  const [loading, setLoading] = useState(false)

  const refetch = useCallback(async () => {
    setLoading(true)
    try {
      const data = await getConversations()
      setConversations(data)
    } catch (e) {
      logger.warn('Failed to fetch conversations', e)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { refetch() }, [refetch])

  return { conversations, loading, refetch }
}
