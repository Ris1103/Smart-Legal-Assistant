import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { ClerkProvider, SignedIn, SignedOut, RedirectToSignIn } from '@clerk/clerk-react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useAuth } from '@clerk/clerk-react'
import { useEffect } from 'react'
import { setAuthToken } from '@/lib/api'
import Layout from '@/components/Layout'
import LoginPage from '@/pages/LoginPage'
import DashboardPage from '@/pages/DashboardPage'
import DocumentsPage from '@/pages/DocumentsPage'
import ContractsPage from '@/pages/ContractsPage'

const queryClient = new QueryClient()
const PUBLISHABLE_KEY = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY

function TokenSync() {
  const { getToken } = useAuth()
  useEffect(() => {
    getToken().then(setAuthToken)
  }, [getToken])
  return null
}

function ProtectedLayout() {
  return (
    <SignedIn>
      <TokenSync />
      <Layout>
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/documents" element={<DocumentsPage />} />
          <Route path="/contracts" element={<ContractsPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Layout>
    </SignedIn>
  )
}

export default function App() {
  return (
    <ClerkProvider publishableKey={PUBLISHABLE_KEY}>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <Routes>
            <Route path="/login" element={<LoginPage />} />
            <Route
              path="/*"
              element={
                <>
                  <SignedOut><RedirectToSignIn /></SignedOut>
                  <ProtectedLayout />
                </>
              }
            />
          </Routes>
        </BrowserRouter>
      </QueryClientProvider>
    </ClerkProvider>
  )
}
