import { useState } from 'react'
import { NavLink } from 'react-router-dom'
import { UserButton } from '@clerk/clerk-react'
import { MessageSquare, FileText, Upload, Menu, X, Scale } from 'lucide-react'
import { cn } from '@/lib/utils'

const navItems = [
  { to: '/', label: 'Chat', icon: MessageSquare },
  { to: '/documents', label: 'Documents', icon: Upload },
  { to: '/contracts', label: 'Contracts', icon: FileText },
]

export default function Layout({ children }: { children: React.ReactNode }) {
  const [sidebarOpen, setSidebarOpen] = useState(false)

  return (
    <div className="flex h-screen bg-gray-50 overflow-hidden">
      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/40 z-20 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside className={cn(
        'fixed lg:static inset-y-0 left-0 z-30 w-64 bg-white border-r border-gray-200 flex flex-col transition-transform duration-200',
        sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0',
      )}>
        <div className="flex items-center gap-2 px-5 h-16 border-b border-gray-200">
          <Scale className="w-6 h-6 text-indigo-600" />
          <span className="font-semibold text-gray-900">Legal Advisor</span>
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
                  ? 'bg-indigo-50 text-indigo-700'
                  : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900',
              )}
            >
              <Icon className="w-4 h-4" />
              {label}
            </NavLink>
          ))}
        </nav>

        <div className="px-5 py-4 border-t border-gray-200">
          <UserButton afterSignOutUrl="/login" />
        </div>
      </aside>

      {/* Main */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Mobile topbar */}
        <header className="lg:hidden flex items-center gap-3 px-4 h-14 bg-white border-b border-gray-200">
          <button onClick={() => setSidebarOpen(true)} className="p-1 rounded-md text-gray-500 hover:text-gray-700">
            {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
          <Scale className="w-5 h-5 text-indigo-600" />
          <span className="font-semibold text-gray-900">Legal Advisor</span>
        </header>

        <main className="flex-1 overflow-auto">
          {children}
        </main>
      </div>
    </div>
  )
}
