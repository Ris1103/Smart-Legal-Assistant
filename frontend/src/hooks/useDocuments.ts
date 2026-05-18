import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import api from '@/lib/api'

export function useIngestDocument() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (file: File) => {
      const base64 = await toBase64(file)
      const { data } = await api.post('/ingest', {
        base64_text: base64,
        file_type: '.pdf',
        filename: file.name,
        metadata: {},
      })
      return data
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ['documents'] }),
  })
}

function toBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      const result = reader.result as string
      resolve(result.split(',')[1])
    }
    reader.onerror = reject
    reader.readAsDataURL(file)
  })
}

export function useGenerateContract() {
  const [loading, setLoading] = useState(false)
  const [contract, setContract] = useState<{ contract_type: string; rendered_contract: string } | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function generate(query: string) {
    setLoading(true)
    setError(null)
    try {
      const { data } = await api.post('/contracts/generate', { user_query: query })
      setContract(data)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to generate contract.')
    } finally {
      setLoading(false)
    }
  }

  return { loading, contract, error, generate }
}
