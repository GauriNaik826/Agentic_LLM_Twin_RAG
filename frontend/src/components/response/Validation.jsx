export default function Validation({ grounded, confidence, latencySec }) {
  return (
    <article className="block">
      <h2>Validation</h2>
      <ul className="validation-list">
        <li>
          <span>Grounded</span>
          <strong>{grounded ? "Yes" : "No"}</strong>
        </li>
        <li>
          <span>Confidence</span>
          <strong>{Math.round(confidence * 100)}%</strong>
        </li>
        <li>
          <span>Latency</span>
          <strong>{latencySec.toFixed(2)} sec</strong>
        </li>
      </ul>
    </article>
  );
}
