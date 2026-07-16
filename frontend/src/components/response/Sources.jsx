function hostLabel(value) {
  try {
    return new URL(value).hostname.replace("www.", "");
  } catch {
    return value;
  }
}

export default function Sources({ sources }) {
  return (
    <article className="block">
      <h2>Sources</h2>
      {sources.length === 0 ? (
        <p className="muted">No explicit source URLs provided for this response.</p>
      ) : (
        <ul className="sources-list">
          {sources.map((source) => (
            <li key={source}>
              <a href={source} target="_blank" rel="noreferrer">
                {hostLabel(source)}
              </a>
            </li>
          ))}
        </ul>
      )}
    </article>
  );
}
