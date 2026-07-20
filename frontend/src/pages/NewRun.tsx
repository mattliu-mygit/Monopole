import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import {
  advanceRun,
  createRun,
  getModels,
  getRubrics,
  setAutoRun,
  setRunConfig,
  setRunSelection,
} from '../api'
import PageHeader from '../components/PageHeader'
import RunSetup from '../features/runs/RunSetup'
import type { DataSelection, RunConfig } from '../types'

function ErrorNotice({ message }: { message: string }) {
  return (
    <div role="alert" className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
      {message}
    </div>
  )
}

export default function NewRun() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const modelsQuery = useQuery({ queryKey: ['models'], queryFn: getModels })
  const rubricsQuery = useQuery({ queryKey: ['rubrics'], queryFn: getRubrics })
  const startMutation = useMutation({
    mutationFn: async ({
      selection,
      config,
      autoRun,
    }: {
      selection: DataSelection
      config: RunConfig
      autoRun: boolean
    }) => {
      const created = await createRun()
      queryClient.setQueryData(['run', created.run_id], created)
      navigate(`/runs/${created.run_id}`)
      await setRunSelection(created.run_id, selection)
      await setRunConfig(created.run_id, config)
      await setAutoRun(created.run_id, autoRun)
      return advanceRun(created.run_id)
    },
    onSuccess: (run) => {
      queryClient.setQueryData(['run', run.run_id], run)
      void queryClient.invalidateQueries({ queryKey: ['runs'] })
    },
  })
  const catalogError = modelsQuery.error ?? rubricsQuery.error

  return (
    <div>
      <PageHeader title="New run" />
      {catalogError ? (
        <div className="space-y-3">
          <ErrorNotice message={(catalogError as Error).message} />
          <button
            type="button"
            className="text-sm font-semibold text-blue-700 hover:underline"
            onClick={() => {
              void modelsQuery.refetch()
              void rubricsQuery.refetch()
            }}
          >
            Retry configuration catalogs
          </button>
        </div>
      ) : modelsQuery.data && rubricsQuery.data ? (
        <RunSetup
          initialSelection={null}
          initialConfig={null}
          initialAutoRun={false}
          models={modelsQuery.data}
          rubrics={rubricsQuery.data}
          pending={startMutation.isPending}
          onStart={(selection, config, autoRun) =>
            startMutation.mutate({ selection, config, autoRun })}
        />
      ) : (
        <p role="status" className="text-sm text-gray-500">Loading configuration catalogs…</p>
      )}
      {startMutation.error && (
        <div className="mt-4">
          <ErrorNotice message={(startMutation.error as Error).message} />
        </div>
      )}
    </div>
  )
}
