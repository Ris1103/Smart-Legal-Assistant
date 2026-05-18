import { Download, FileText } from 'lucide-react'

interface Props {
  contractType: string
  renderedText: string
}

export default function ContractViewer({ contractType, renderedText }: Props) {
  function download() {
    const blob = new Blob([renderedText], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${contractType.toLowerCase().replace(/\s+/g, '_')}.txt`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="bg-white border border-gray-200 rounded-2xl overflow-hidden shadow-sm">
      <div className="flex items-center justify-between px-5 py-3 border-b border-gray-100 bg-gray-50">
        <div className="flex items-center gap-2 text-sm font-medium text-gray-700">
          <FileText className="w-4 h-4 text-indigo-500" />
          {contractType}
        </div>
        <button
          onClick={download}
          className="flex items-center gap-1.5 text-xs text-indigo-600 hover:text-indigo-800 transition-colors"
        >
          <Download className="w-3.5 h-3.5" /> Download
        </button>
      </div>
      <pre className="p-5 text-xs text-gray-700 whitespace-pre-wrap leading-relaxed max-h-[60vh] overflow-y-auto font-mono">
        {renderedText}
      </pre>
    </div>
  )
}
