# What: Import Opik instrumentation decorators.
# Why: Tracks Twin Writer executions for observability and debugging.
import opik
# What: Import Python regex utilities.
# Why: Used to detect invalid instruction patterns safely.
import re

# What: Import circuit breaker primitives.
# Why: Protect SageMaker dependency from repeated failures.
from llm_engineering.application.guardrails import CircuitBreaker, CircuitBreakerConfig
# What: Import settings singleton.
# Why: Provides endpoint config and environment-driven values.
from llm_engineering import settings
# What: Import inference utilities.
# Why: Handles request construction/execution against SageMaker endpoint.
from llm_engineering.model.inference import InferenceExecutor, LLMInferenceSagemakerEndpoint


# What: Twin Writer agent for style-focused generation.
# Why: Produces author-style responses without retrieval.
class TwinWriterAgent:
    """Style-focused writer that directly calls the fine-tuned SageMaker endpoint."""

    # What: Default style prompt when caller does not provide one.
    # Why: Ensures generation still has style constraints.
    DEFAULT_STYLE_PROMPT = (
        "Write in the author's style: clear, practical, and technical. "
        "Use short paragraphs, concrete examples, and an engaging professional tone."
    )
    # What: Hard cap for user query size.
    # Why: Prevents oversized prompts that can break or degrade generation.
    MAX_QUERY_LENGTH = 4000
    # What: Hard cap for style prompt size.
    # Why: Keeps style instruction concise and model-friendly.
    MAX_STYLE_PROMPT_LENGTH = 1000

    # What: Regex patterns for instruction-override / prompt-injection style requests.
    # Why: Rejects unsafe control-hijacking instructions before model call.
    _INVALID_INSTRUCTION_PATTERNS = [
        re.compile(r"\bignore\s+(all\s+)?(previous|prior)\s+instructions\b", re.IGNORECASE),
        re.compile(r"\breveal\s+(the\s+)?system\s+prompt\b", re.IGNORECASE),
        re.compile(r"\bforget\s+everything\b", re.IGNORECASE),
    ]
    # What: Circuit breaker guarding SageMaker endpoint calls.
    # Why: Opens after repeated failures and recovers via half-open probe.
    _sagemaker_cb = CircuitBreaker(
        name="twin_writer_sagemaker",
        config=CircuitBreakerConfig(failure_threshold=5, recovery_timeout_seconds=30, half_open_max_calls=1),
    )

    # What: Checks whether text matches unsafe instruction patterns.
    # Why: Blocks known prompt-injection phrases early.
    @staticmethod
    def _is_invalid_instruction(text: str) -> bool:
        # What: Returns True if any compiled regex matches.
        # Why: Single helper keeps validation logic reusable and clear.
        return any(pattern.search(text) for pattern in TwinWriterAgent._INVALID_INSTRUCTION_PATTERNS)

    # What: Main Twin Writer execution method.
    # Why: Validates input, then generates style-conditioned output.
    @staticmethod
    # What: Opik trace wrapper.
    # Why: Captures runtime telemetry for this agent call.
    @opik.track(name="twin_writer")
    def invoke(query: str, style_prompt: str | None = None) -> str:
        # What: Reject empty/blank queries.
        # Why: Model cannot generate meaningful output without user intent.
        if len(query.strip()) == 0:
            return "I need a prompt to generate a response."

        # What: Enforce maximum query length.
        # Why: Prevent prompt overflow and unstable latency/cost.
        if len(query) > TwinWriterAgent.MAX_QUERY_LENGTH:
            return "Your prompt is too long. Please provide a shorter request."

        # What: Resolve style prompt with default fallback.
        # Why: Guarantees a non-empty style policy for generation.
        effective_style_prompt = (style_prompt or TwinWriterAgent.DEFAULT_STYLE_PROMPT).strip()
        # What: Validate style prompt presence.
        # Why: Missing style guidance defeats Twin Writer purpose.
        if len(effective_style_prompt) == 0:
            return "A valid style prompt is required before generation."

        # What: Enforce style prompt length bound.
        # Why: Avoid overly verbose style instructions.
        if len(effective_style_prompt) > TwinWriterAgent.MAX_STYLE_PROMPT_LENGTH:
            return "Style prompt is too long. Please provide a shorter style instruction."

        # What: Block prompt-injection attempts in both query and style prompt.
        # Why: Prevent user instructions from overriding safe behavior.
        if TwinWriterAgent._is_invalid_instruction(query) or TwinWriterAgent._is_invalid_instruction(
            effective_style_prompt
        ):
            return "I cannot follow instruction-override requests."

        # What: Template that combines user request with style instructions.
        # Why: Shapes model output toward desired author voice.
        prompt = (
            "You are a fine-tuned TwinLlama writing assistant.\n"
            "User Prompt: {query}\n"
            "Style Prompt: {style_prompt}\n"
            "Generate the final response now."
        )

        # What: Local callable for guarded SageMaker invocation.
        # Why: Circuit breaker requires a callable to wrap execution.
        def _call() -> str:
            # What: Build SageMaker LLM client.
            # Why: Connects to configured inference endpoint.
            llm = LLMInferenceSagemakerEndpoint(
                endpoint_name=settings.SAGEMAKER_ENDPOINT_INFERENCE,
                inference_component_name=None,
            )
            # What: Execute inference with formatted prompt.
            # Why: Produces final generated response text.
            return InferenceExecutor(
                llm=llm,
                query=query,
                context="",
                prompt=prompt.format(
                    # What: Preserve `{query}` placeholder for InferenceExecutor formatting.
                    # Why: Executor injects real query at execute-time.
                    query="{query}",
                    # What: Insert validated style prompt now.
                    # Why: Locks in style instructions before execution.
                    style_prompt=effective_style_prompt,
                ),
            ).execute()

        # What: Run model call through circuit breaker.
        # Why: Fails fast when dependency is unhealthy.
        answer = TwinWriterAgent._sagemaker_cb.call(_call)

        # What: Return generated answer.
        # Why: Final output for supervisor/API response.
        return answer