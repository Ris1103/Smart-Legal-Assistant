import { useState } from 'react'
import { Loader2, FileSignature } from 'lucide-react'
import { useGenerateContract } from '@/hooks/useDocuments'
import ContractViewer from '@/components/ContractViewer'

const TEMPLATES = [
  { label: 'NDA', prompt: 'Draft an NDA between [Party A] and [Party B] for a software development project' },
  { label: 'Service Agreement', prompt: 'Draft a service agreement between [Client] and [Vendor] for IT consulting services' },
  { label: 'Employment Agreement', prompt: 'Draft an employment agreement for a software engineer joining [Company]' },
]

export default function ContractsPage() {
  const [query, setQuery] = useState('')
  const { loading, contract, error, generate } = useGenerateContract()

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    const q = query.trim()
    if (q) generate(q)
  }

  return (
    <div className="max-w-3xl mx-auto px-4 py-8 space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-gray-900">Contract Generator</h1>
        <p className="text-sm text-gray-500 mt-1">
          Describe the contract you need in plain English.
        </p>
      </div>

      {/* Quick templates */}
      <div className="flex flex-wrap gap-2">
        {TEMPLATES.map((t) => (
          <button
            key={t.label}
            onClick={() => setQuery(t.prompt)}
            className="text-xs bg-indigo-50 text-indigo-700 border border-indigo-200 px-3 py-1.5 rounded-full hover:bg-indigo-100 transition-colors"
          >
            {t.label}
          </button>
        ))}
      </div>

      <form onSubmit={handleSubmit} className="space-y-3">
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          rows={3}
          placeholder="e.g. Draft an NDA between Acme Corp and John Doe for a 6-month AI project"
          className="w-full text-sm border border-gray-200 rounded-xl px-4 py-3 outline-none focus:border-indigo-400 transition-colors resize-none text-gray-800 placeholder-gray-400"
        />
        <button
          type="submit"
          disabled={!query.trim() || loading}
          className="flex items-center gap-2 bg-indigo-600 text-white text-sm font-medium px-5 py-2.5 rounded-xl hover:bg-indigo-700 disabled:opacity-40 transition-colors"
        >
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <FileSignature className="w-4 h-4" />}
          Generate Contract
        </button>
      </form>

      {error && <p className="text-sm text-red-500">{error}</p>}

      {contract && (
        <ContractViewer
          contractType={contract.contract_type}
          renderedText={contract.rendered_contract}
        />
      )}
    </div>
  )
}
