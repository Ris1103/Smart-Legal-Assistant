import { useState } from 'react'
import { ChevronDown, ChevronUp } from 'lucide-react'
import { cn } from '@/lib/utils'
import type { Message } from '@/hooks/useChat'

export default function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === 'user'
  const [citationsOpen, setCitationsOpen] = useState(false)
  const citations = message.citations ?? []

  return (
    <div className={cn('flex', isUser ? 'justify-end' : 'justify-start')}>
      <div className={cn(
        'max-w-prose rounded-2xl px-4 py-3 text-sm leading-relaxed',
        isUser
          ? 'bg-indigo-600 text-white rounded-br-sm'
          : 'bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-gray-800 dark:text-gray-100 rounded-bl-sm shadow-sm',
      )}>
        <p className="whitespace-pre-wrap">{message.content}</p>

        {!isUser && (citations.length > 0 || (message.sources && message.sources.length > 0)) && (
          <div className="mt-3 pt-2 border-t border-gray-100 dark:border-gray-700">
            {/* Sources chips */}
            {message.sources && message.sources.length > 0 && (
              <div className="flex flex-wrap gap-1 mb-2">
                {message.sources.map((s, i) => (
                  <span key={i} className="text-xs bg-indigo-50 dark:bg-indigo-950 text-indigo-600 dark:text-indigo-300 px-2 py-0.5 rounded-full border border-indigo-100 dark:border-indigo-800">
                    {s}
                  </span>
                ))}
              </div>
            )}

            {/* Citations collapsible */}
            {citations.length > 0 && (
              <div>
                <button
                  onClick={() => setCitationsOpen(o => !o)}
                  className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
                >
                  {citationsOpen ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                  {citations.length} source{citations.length !== 1 ? 's' : ''}
                </button>

                {citationsOpen && (
                  <div className="mt-2 space-y-2">
                    {citations.map((c, i) => (
                      <div key={i} className="bg-gray-50 dark:bg-gray-900 rounded-lg px-3 py-2 border border-gray-100 dark:border-gray-700">
                        <p className="text-xs font-medium text-indigo-700 dark:text-indigo-400">{c.filename} — p.{c.page}</p>
                        {c.excerpt && (
                          <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5 italic">"{c.excerpt}…"</p>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
