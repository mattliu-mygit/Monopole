interface ScoreBadgeProps {
  scorer: string;
  value: number;
}

function badgeColors(value: number): string {
  if (value >= 0.7) return "bg-green-100 text-green-800";
  if (value >= 0.4) return "bg-yellow-100 text-yellow-800";
  return "bg-red-100 text-red-800";
}

function abbreviate(scorer: string): string {
  return scorer
    .replace(/^weave_agent_signals\./, "")
    .replace(/^(outcome\.|judge\.)/, "");
}

export default function ScoreBadge({ scorer, value }: ScoreBadgeProps) {
  return (
    <span
      className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${badgeColors(value)}`}
    >
      {abbreviate(scorer)} {(value ?? 0).toFixed(2)}
    </span>
  );
}
