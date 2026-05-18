import { cn } from '@/lib/utils'
import type { Message } from '@/hooks/useChat'

export default function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === 'user'

  return (
    <div className={cn('flex', isUser ? 'justify-end' : 'justify-start')}>
      <div className={cn(
        'max-w-prose rounded-2xl px-4 py-3 text-sm leading-relaxed',
        isUser
          ? 'bg-indigo-600 text-white rounded-br-sm'
          : 'bg-white border border-gray-200 text-gray-800 rounded-bl-sm shadow-sm',
      )}>
        <p className="whitespace-pre-wrap">{message.content}</p>

        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="mt-2 pt-2 border-t border-gray-100">
            <p className="text-xs text-gray-400 font-medium mb-1">Sources</p>
            <div className="flex flex-wrap gap-1">
              {message.sources.map((s, i) => (
                <span key={i} className="text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded-full">
                  {s}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
