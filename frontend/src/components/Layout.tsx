import { useState } from 'react'
import { NavLink } from 'react-router-dom'
import { UserButton } from '@clerk/clerk-react'
import { MessageSquare, FileText, Upload, Menu, X, Scale, Sun, Moon } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useTheme } from '@/context/ThemeContext'
import { useChatContext } from '@/context/ChatContext'
import ConversationSidebar from '@/components/ConversationSidebar'

const navItems = [
  { to: '/', label: 'Chat', icon: MessageSquare },
  { to: '/documents', label: 'Documents', icon: Upload },
  { to: '/contracts', label: 'Contracts', icon: FileText },
]

export default function Layout({ children }: { children: React.ReactNode }) {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const { theme, toggleTheme } = useTheme()
  const { conversationId, loadConversation, newChat } = useChatContext()

  return (
    <div className="flex h-screen bg-gray-50 dark:bg-gray-950 overflow-hidden">
      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/40 z-20 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside className={cn(
        'fixed lg:static inset-y-0 left-0 z-30 w-64 bg-white dark:bg-gray-900 border-r border-gray-200 dark:border-gray-700 flex flex-col transition-transform duration-200',
        sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0',
      )}>
        <div className="flex items-center gap-2 px-5 h-16 border-b border-gray-200 dark:border-gray-700">
          <Scale className="w-6 h-6 text-indigo-600" />
          <span className="font-semibold text-gray-900 dark:text-gray-100">Legal Advisor</span>
        </div>

        <nav className="flex-1 px-3 py-4 space-y-1">
          {navItems.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              onClick={() => setSidebarOpen(false)}
              className={({ isActive }) => cn(
                'flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                isActive
                  ? 'bg-indigo-50 dark:bg-indigo-950 text-indigo-700 dark:text-indigo-300'
                  : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 hover:text-gray-900 dark:hover:text-gray-100',
              )}
            >
              <Icon className="w-4 h-4" />
              {label}
            </NavLink>
          ))}
        </nav>

        <div className="flex-1 border-t border-gray-200 dark:border-gray-700 overflow-y-auto py-2">
          <ConversationSidebar
            activeConversationId={conversationId}
            onSelect={(id) => { loadConversation(id); setSidebarOpen(false) }}
            onNew={() => { newChat(); setSidebarOpen(false) }}
          />
        </div>

        <div className="px-5 py-4 border-t border-gray-200 dark:border-gray-700 flex items-center justify-between">
          <UserButton afterSignOutUrl="/login" />
          <button
            onClick={toggleTheme}
            className="p-1.5 rounded-lg text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
            aria-label="Toggle theme"
          >
            {theme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>
        </div>
      </aside>

      {/* Main */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Mobile topbar */}
        <header className="lg:hidden flex items-center gap-3 px-4 h-14 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-700">
          <button onClick={() => setSidebarOpen(true)} className="p-1 rounded-md text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200">
            {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
          <Scale className="w-5 h-5 text-indigo-600" />
          <span className="font-semibold text-gray-900 dark:text-gray-100">Legal Advisor</span>
        </header>

        <main className="flex-1 overflow-auto">
          {children}
        </main>
      </div>
    </div>
  )
}
