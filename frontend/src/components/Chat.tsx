import { useRef, useEffect, useState } from 'react'
import { Send, Loader2, Trash2, FileDown } from 'lucide-react'
import { useChat } from '@/hooks/useChat'
import type { Message } from '@/hooks/useChat'
import MessageBubble from './MessageBubble'
import api from '@/lib/api'

const SUGGESTIONS = [
  'What are the GST filing deadlines for a startup?',
  'How do I register a private limited company?',
  'What are TDS deduction rules for salary?',
  'Explain Section 80C tax deductions.',
]

async function downloadPDF(msg: Message, query: string) {
  try {
    const response = await api.post(
      '/export/pdf',
      {
        query,
        summary: msg.content,
        citations: msg.citations ?? [],
        domain: msg.domain ?? null,
        metadata: msg.rawData?.metadata ?? {},
      },
      { responseType: 'blob' },
    )
    const url = URL.createObjectURL(response.data)
    const a = document.createElement('a')
    a.href = url
    a.download = `legal_advisory_${Date.now()}.pdf`
    a.click()
    URL.revokeObjectURL(url)
  } catch {
    alert('Failed to generate PDF. Please try again.')
  }
}

export default function Chat() {
  const { messages, loading, error, sendMessage, clearMessages } = useChat()
  const [input, setInput] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    const q = input.trim()
    if (!q || loading) return
    setInput('')
    sendMessage(q)
  }

  // Find the most recent user message for a given assistant message
  function getUserQuery(assistantIndex: number): string {
    for (let i = assistantIndex - 1; i >= 0; i--) {
      if (messages[i].role === 'user') return messages[i].content
    }
    return ''
  }

  return (
    <div className="flex flex-col h-full max-w-3xl mx-auto px-4">
      {/* Header */}
      <div className="flex items-center justify-between py-4">
        <div>
          <h1 className="text-lg font-semibold text-gray-900">Legal Assistant</h1>
          <p className="text-xs text-gray-500">Ask anything about Indian law</p>
        </div>
        {messages.length > 0 && (
          <button
            onClick={clearMessages}
            className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-red-500 transition-colors"
          >
            <Trash2 className="w-3.5 h-3.5" /> Clear
          </button>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto space-y-4 pb-4">
        {messages.length === 0 && (
          <div className="pt-8">
            <p className="text-center text-gray-400 text-sm mb-6">Try asking something:</p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => sendMessage(s)}
                  className="text-left text-sm bg-white border border-gray-200 rounded-xl px-4 py-3 text-gray-600 hover:border-indigo-300 hover:text-indigo-700 transition-colors shadow-sm"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, idx) => (
          <div key={m.id}>
            <MessageBubble message={m} />
            {m.role === 'assistant' && (
              <div className="flex justify-start mt-1 ml-1">
                <button
                  onClick={() => downloadPDF(m, getUserQuery(idx))}
                  className="flex items-center gap-1 text-xs text-gray-400 hover:text-indigo-600 transition-colors"
                >
                  <FileDown className="w-3.5 h-3.5" /> Download as PDF
                </button>
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-white border border-gray-200 rounded-2xl rounded-bl-sm px-4 py-3 shadow-sm">
              <Loader2 className="w-4 h-4 text-indigo-500 animate-spin" />
            </div>
          </div>
        )}

        {error && <p className="text-center text-xs text-red-500">{error}</p>}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="py-4 border-t border-gray-100">
        <div className="flex gap-2 bg-white border border-gray-200 rounded-2xl px-4 py-2 shadow-sm focus-within:border-indigo-400 transition-colors">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a legal question..."
            className="flex-1 text-sm outline-none bg-transparent text-gray-800 placeholder-gray-400"
            disabled={loading}
          />
          <button
            type="submit"
            disabled={!input.trim() || loading}
            className="p-1.5 rounded-xl bg-indigo-600 text-white disabled:opacity-40 hover:bg-indigo-700 transition-colors"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </form>
    </div>
  )
}
