import { useState } from "react";

interface JobProgressProps {
  job: {
    job_id: string;
    type: string;
    status: string;
    started_at: string;
    progress: Record<string, any>;
    result: any;
    error: string | null;
  };
}

function statusBadge(status: string) {
  const colors: Record<string, string> = {
    complete: "bg-green-100 text-green-800",
    running: "bg-yellow-100 text-yellow-800",
    failed: "bg-red-100 text-red-800",
  };
  return (
    <span
      className={`inline-block rounded-full px-2 py-0.5 text-xs font-medium ${colors[status] ?? "bg-gray-100 text-gray-800"}`}
    >
      {status}
    </span>
  );
}

function typeBadge(type: string) {
  const colors: Record<string, string> = {
    scoring: "bg-blue-100 text-blue-800",
    judging: "bg-purple-100 text-purple-800",
    analysis: "bg-indigo-100 text-indigo-800",
  };
  return (
    <span
      className={`inline-block rounded-full px-2 py-0.5 text-xs font-medium uppercase ${colors[type] ?? "bg-gray-100 text-gray-800"}`}
    >
      {type}
    </span>
  );
}

function progressPercent(progress: Record<string, any>): number | null {
  for (const v of Object.values(progress)) {
    if (typeof v === "number" && v >= 0 && v <= 1) return v;
  }
  // Check for done/total pattern
  if (typeof progress.done === "number" && typeof progress.total === "number" && progress.total > 0) {
    return progress.done / progress.total;
  }
  return null;
}

export default function JobProgress({ job }: JobProgressProps) {
  const [showResult, setShowResult] = useState(false);
  const pct = job.status === "running" ? progressPercent(job.progress) : null;

  return (
    <div className="rounded-lg border p-4 shadow-sm">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {typeBadge(job.type)}
          {statusBadge(job.status)}
        </div>
        <span className="font-mono text-xs text-gray-500">{job.job_id}</span>
      </div>

      <div className="mt-2 text-xs text-gray-500">
        Started {new Date(job.started_at).toLocaleString()}
      </div>

      {pct !== null && (
        <div className="mt-3">
          <div className="bg-gray-200 rounded-full h-2">
            <div
              className="bg-blue-500 rounded-full h-2 transition-all"
              style={{ width: `${Math.round(pct * 100)}%` }}
            />
          </div>
          <div className="text-xs text-gray-500 mt-1">
            {Math.round(pct * 100)}%
          </div>
        </div>
      )}

      {job.status === "failed" && job.error && (
        <div className="mt-3 text-red-600 text-sm">{job.error}</div>
      )}

      {job.status === "complete" && job.result && (
        <div className="mt-3">
          <button
            onClick={() => setShowResult((prev) => !prev)}
            className="text-sm text-blue-600 hover:text-blue-800"
          >
            {showResult ? "Hide result" : "View result"}
          </button>
          {showResult && (
            <pre className="mt-2 bg-gray-50 rounded-lg border p-3 text-xs font-mono overflow-auto max-h-64">
              {typeof job.result === "string"
                ? job.result
                : JSON.stringify(job.result, null, 2)}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}
