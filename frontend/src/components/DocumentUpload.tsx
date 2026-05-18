import { useRef, useState } from 'react'
import { Upload, FileText, CheckCircle, AlertCircle, Loader2 } from 'lucide-react'
import { useIngestDocument } from '@/hooks/useDocuments'
import { cn } from '@/lib/utils'

export default function DocumentUpload() {
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const { mutate, isPending, isSuccess, isError, error, reset } = useIngestDocument()

  function handleFile(file: File) {
    if (!file.name.endsWith('.pdf')) return
    reset()
    mutate(file)
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  return (
    <div className="space-y-3">
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        className={cn(
          'border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-colors',
          dragging ? 'border-indigo-400 bg-indigo-50' : 'border-gray-200 hover:border-indigo-300 hover:bg-gray-50',
        )}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf"
          className="hidden"
          onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f) }}
        />

        {isPending ? (
          <div className="flex flex-col items-center gap-2 text-indigo-600">
            <Loader2 className="w-8 h-8 animate-spin" />
            <p className="text-sm font-medium">Uploading & indexing…</p>
          </div>
        ) : isSuccess ? (
          <div className="flex flex-col items-center gap-2 text-green-600">
            <CheckCircle className="w-8 h-8" />
            <p className="text-sm font-medium">Document ingested successfully</p>
            <p className="text-xs text-gray-400">Drop another PDF to add more</p>
          </div>
        ) : isError ? (
          <div className="flex flex-col items-center gap-2 text-red-500">
            <AlertCircle className="w-8 h-8" />
            <p className="text-sm font-medium">Upload failed</p>
            <p className="text-xs">{error?.message}</p>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-2 text-gray-400">
            <Upload className="w-8 h-8" />
            <p className="text-sm font-medium text-gray-600">Drop a PDF here or click to browse</p>
            <p className="text-xs">Max 50 MB · PDF only</p>
          </div>
        )}
      </div>

      <div className="flex items-center gap-2 text-xs text-gray-400">
        <FileText className="w-3.5 h-3.5" />
        Documents are chunked, embedded, and added to the knowledge base immediately.
      </div>
    </div>
  )
}
