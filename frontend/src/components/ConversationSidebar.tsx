import { PlusCircle, MessageSquare } from 'lucide-react'
import { useUser } from '@clerk/clerk-react'
import { cn } from '@/lib/utils'
import { useConversations } from '@/hooks/useConversations'
import type { ConversationSummary } from '@/lib/api'

interface Props {
  activeConversationId: string | null
  onSelect: (id: string) => void
  onNew: () => void
}

function formatDate(iso: string) {
  const d = new Date(iso)
  const now = new Date()
  const diffDays = Math.floor((now.getTime() - d.getTime()) / 86400000)
  if (diffDays === 0) return 'Today'
  if (diffDays === 1) return 'Yesterday'
  if (diffDays < 7) return `${diffDays}d ago`
  return d.toLocaleDateString('en-IN', { day: 'numeric', month: 'short' })
}

export default function ConversationSidebar({ activeConversationId, onSelect, onNew }: Props) {
  const { isSignedIn } = useUser()
  const { conversations, loading } = useConversations()

  if (!isSignedIn) return null

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-3 py-2">
        <span className="text-xs font-semibold text-gray-400 dark:text-gray-500 uppercase tracking-wide">History</span>
        <button
          onClick={onNew}
          className="p-1 rounded-md text-gray-400 hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
          title="New chat"
        >
          <PlusCircle className="w-4 h-4" />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto space-y-0.5 px-1">
        {loading && (
          <p className="text-xs text-gray-400 dark:text-gray-500 px-2 py-2">Loading…</p>
        )}
        {!loading && conversations.length === 0 && (
          <p className="text-xs text-gray-400 dark:text-gray-500 px-2 py-2">No conversations yet.</p>
        )}
        {conversations.map((c: ConversationSummary) => (
          <button
            key={c.id}
            onClick={() => onSelect(c.id)}
            className={cn(
              'w-full text-left px-2 py-2 rounded-lg transition-colors group',
              activeConversationId === c.id
                ? 'bg-indigo-50 dark:bg-indigo-950'
                : 'hover:bg-gray-100 dark:hover:bg-gray-800',
            )}
          >
            <div className="flex items-start gap-2">
              <MessageSquare className="w-3.5 h-3.5 mt-0.5 shrink-0 text-gray-400 dark:text-gray-500" />
              <div className="min-w-0 flex-1">
                <p className={cn(
                  'text-xs font-medium truncate',
                  activeConversationId === c.id
                    ? 'text-indigo-700 dark:text-indigo-300'
                    : 'text-gray-700 dark:text-gray-300',
                )}>
                  {c.title ?? 'Untitled'}
                </p>
                <p className="text-xs text-gray-400 dark:text-gray-500 mt-0.5">
                  {formatDate(c.created_at)} · {c.message_count} msg{c.message_count !== 1 ? 's' : ''}
                </p>
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}
