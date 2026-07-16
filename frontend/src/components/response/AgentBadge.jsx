function agentEmoji(agent) {
  if (agent === "Advanced RAG") return "🟢";
  if (agent === "Web Agent") return "🔵";
  if (agent === "Twin Writer") return "🟠";
  return "⚪";
}

export default function AgentBadge({ agent }) {
  return (
    <article className="block">
      <h2>Agent Used</h2>
      <p className="agent-badge">
        <span>{agentEmoji(agent)}</span>
        <span>{agent || "Unknown"}</span>
      </p>
    </article>
  );
}
