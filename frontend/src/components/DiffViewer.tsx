interface DiffViewerProps {
  diff: string;
}

function lineClass(line: string): string {
  if (line.startsWith("@@")) return "bg-blue-50 text-blue-600 font-medium";
  if (line.startsWith("+")) return "bg-green-100 text-green-800";
  if (line.startsWith("-")) return "bg-red-100 text-red-800";
  return "";
}

export default function DiffViewer({ diff }: DiffViewerProps) {
  const lines = diff.split("\n");

  return (
    <pre className="font-mono text-xs overflow-x-auto rounded-lg border p-4">
      <code>
        {lines.map((line, i) => (
          <div key={i} className={lineClass(line)}>
            {line}
          </div>
        ))}
      </code>
    </pre>
  );
}
