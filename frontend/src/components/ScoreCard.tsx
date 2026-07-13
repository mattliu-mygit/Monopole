interface ScoreCardProps {
  scorer: string;
  mean: number;
  count: number;
  ci: [number, number];
  passRate: number | null;
  trend?: { direction: string; delta: number };
  confident: boolean;
}

function bgColor(mean: number): string {
  if (mean >= 0.7) return "bg-green-50";
  if (mean >= 0.4) return "bg-yellow-50";
  return "bg-red-50";
}

function trendIndicator(trend: { direction: string; delta: number }) {
  const delta = (trend.delta * 100).toFixed(1);
  if (trend.direction === "up") {
    return <span className="text-green-600">↑ +{delta}%</span>;
  }
  if (trend.direction === "down") {
    return <span className="text-red-600">↓ {delta}%</span>;
  }
  return <span className="text-gray-500">→ {delta}%</span>;
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
