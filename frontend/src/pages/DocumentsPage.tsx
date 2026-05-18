import DocumentUpload from '@/components/DocumentUpload'

export default function DocumentsPage() {
  return (
    <div className="max-w-2xl mx-auto px-4 py-8">
      <div className="mb-6">
        <h1 className="text-xl font-semibold text-gray-900">Documents</h1>
        <p className="text-sm text-gray-500 mt-1">
          Upload legal PDFs to expand the knowledge base. Supports GST acts, IT regulations, company law, etc.
        </p>
      </div>
      <DocumentUpload />
    </div>
  )
}
