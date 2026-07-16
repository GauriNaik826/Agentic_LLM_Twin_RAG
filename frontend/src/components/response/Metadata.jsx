export default function Metadata({ metadata, checks }) {
  const entries = Object.entries(checks || {}).filter(([, value]) => typeof value === "boolean");

  return (
    <article className="block">
      <h2>Metadata</h2>
      <p className="metadata-route">Route: {metadata.selected_route || metadata.executed_agent || "unknown"}</p>
      {entries.length > 0 && (
        <ul className="checks-list">
          {entries.map(([key, value]) => (
            <li key={key}>
              <span>{key}</span>
              <strong>{value ? "pass" : "fail"}</strong>
            </li>
          ))}
        </ul>
      )}
    </article>
  );
}
