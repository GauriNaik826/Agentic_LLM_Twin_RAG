import AgentBadge from "./response/AgentBadge";
import Answer from "./response/Answer";
import Sources from "./response/Sources";
import Validation from "./response/Validation";
import Metadata from "./response/Metadata";

export default function ResponseCard({ data }) {
  const validationChecks = data?.metadata?.validation_checks || {};

  return (
    <section className="response-card">
      <AgentBadge agent={data.agent_used} />
      <Answer answer={data.answer} />
      <Sources sources={data.sources || []} />
      <Validation
        grounded={Boolean(data.grounded)}
        confidence={Number(data.confidence || 0)}
        latencySec={Number(data.latency_sec || 0)}
      />
      <Metadata metadata={data.metadata || {}} checks={validationChecks} />
    </section>
  );
}
