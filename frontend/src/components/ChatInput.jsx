export default function ChatInput({ value, onChange, onSubmit, disabled, canSubmit }) {
  return (
    <div className="chat-input">
      <label htmlFor="query">Ask a question...</label>
      <textarea
        id="query"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Type your query for the agentic system"
        rows={4}
        disabled={disabled}
      />
      <button type="button" onClick={onSubmit} disabled={!canSubmit}>
        Send
      </button>
    </div>
  );
}
