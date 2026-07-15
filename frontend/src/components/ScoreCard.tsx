import type { TrendEntry } from "../types";

type Trend = Pick<TrendEntry, "direction" | "delta">;

interface ScoreCardProps {
  scorer: string;
  mean: number;
  count: number;
  ci: [number, number];
  passRate: number | null;
  trend?: Trend;
  confident: boolean;
}

function bgColor(mean: number): string {
  if (mean >= 0.7) return "bg-green-50";
  if (mean >= 0.4) return "bg-yellow-50";
  return "bg-red-50";
}

function trendIndicator(trend: Trend) {
  const delta = (trend.delta * 100).toFixed(1);
  if (trend.direction === "regression") {
    return (
      <span role="img" aria-label="Regression" className="text-red-600">
        ↓ {delta}%
      </span>
    );
  }
  return (
    <span role="img" aria-label="Improvement" className="text-green-600">
      ↑ +{delta}%
    </span>
  );
}

export default function ScoreCard({
  scorer,
  mean,
  count,
  ci,
  passRate,
  trend,
  confident,
}: ScoreCardProps) {
  const displayValue = passRate !== null ? passRate : mean;
  const formatted = `${(displayValue * 100).toFixed(1)}%`;

  return (
    <div className={`rounded-lg border p-4 shadow-sm ${bgColor(mean)}`}>
      <div className="font-medium text-sm text-gray-700">{scorer}</div>
      <div className="text-3xl font-bold mt-2">{formatted}</div>
      <div className="flex items-center justify-between mt-3">
        <span className="text-xs text-gray-500">n={count}</span>
        <span className="text-xs text-gray-400">
          [{ci[0].toFixed(2)}, {ci[1].toFixed(2)}]
        </span>
        {trend && <span className="text-xs">{trendIndicator(trend)}</span>}
      </div>
      {!confident && (
        <span className="inline-block mt-2 text-xs text-amber-600">
          ⚠ low n
        </span>
      )}
    </div>
  );
}
