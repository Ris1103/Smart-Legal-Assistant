import { SignIn } from '@clerk/clerk-react'
import { Scale } from 'lucide-react'

export default function LoginPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 to-white flex flex-col items-center justify-center px-4">
      <div className="flex items-center gap-2 mb-8">
        <Scale className="w-8 h-8 text-indigo-600" />
        <span className="text-2xl font-semibold text-gray-900">Legal Advisor</span>
      </div>
      <p className="text-sm text-gray-500 mb-6 text-center max-w-xs">
        AI-powered Indian legal assistant for GST, Income Tax, Company Law and more.
      </p>
      <SignIn routing="hash" afterSignInUrl="/" />
    </div>
  )
}
