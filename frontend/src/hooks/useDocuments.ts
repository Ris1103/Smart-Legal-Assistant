import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import api from '@/lib/api'
import { getLogger } from '@/lib/logger'

const logger = getLogger('useDocuments')

export function useIngestDocument() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (file: File) => {
      logger.info('Ingesting document', { filename: file.name, sizeMb: (file.size / 1024 / 1024).toFixed(2) })
      const base64 = await toBase64(file)
      const { data } = await api.post('/ingest', {
        base64_text: base64,
        file_type: '.pdf',
        filename: file.name,
        metadata: {},
      })
      logger.info('Document ingested', { filename: file.name, status: data.status, chunksAdded: data.chunks_added })
      return data
    },
    onSuccess: (data) => {
      if (data.status === 'duplicate') {
        logger.warn('Document already exists in knowledge base', { filename: data.filename })
      }
      qc.invalidateQueries({ queryKey: ['documents'] })
    },
    onError: (err: unknown) => {
      const message = err instanceof Error ? err.message : 'Unknown error'
      logger.error('Document ingestion failed', { error: message })
    },
  })
}

function toBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      const result = reader.result as string
      resolve(result.split(',')[1])
    }
    reader.onerror = (e) => {
      logger.error('FileReader failed', { filename: file.name, error: String(e) })
      reject(e)
    }
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
    logger.info('Generating contract', { queryLength: query.length })

    try {
      const { data } = await api.post('/contracts/generate', { user_query: query })
      logger.info('Contract generated', { contractType: data.contract_type })
      setContract(data)
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to generate contract.'
      logger.error('Contract generation failed', { error: message })
      setError(message)
    } finally {
      setLoading(false)
    }
  }

  return { loading, contract, error, generate }
}
