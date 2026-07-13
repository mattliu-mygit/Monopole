interface CodeViewerProps {
  content: string;
  filename?: string;
}

export default function CodeViewer({ content, filename }: CodeViewerProps) {
  return (
    <div>
      {filename && (
        <div className="bg-gray-100 px-4 py-2 text-sm font-mono text-gray-600 rounded-t-lg border border-b-0">
          {filename}
        </div>
      )}
      <pre
        className={`font-mono text-xs overflow-auto p-4 bg-gray-50 border max-h-96 ${
          filename ? "rounded-b-lg" : "rounded-lg"
        }`}
      >
        <code>{content}</code>
      </pre>
    </div>
  );
}
