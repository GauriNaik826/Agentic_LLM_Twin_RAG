import { useMemo, useState } from "react";
import Header from "./components/Header";
import ChatInput from "./components/ChatInput";
import LoadingSpinner from "./components/LoadingSpinner";
import ResponseCard from "./components/ResponseCard";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export default function App() {
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [error, setError] = useState("");

  const canSubmit = useMemo(() => query.trim().length > 0 && !isLoading, [query, isLoading]);

  async function handleSend() {
    if (!canSubmit) return;

    setIsLoading(true);
    setError("");

    try {
      const res = await fetch(`${API_BASE_URL}/rag/details`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: query.trim() }),
      });

      if (!res.ok) {
        throw new Error(`Request failed with status ${res.status}`);
      }

      const data = await res.json();
      setResponse(data);
    } catch (err) {
      setResponse(null);
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="page-shell">
      <main className="panel">
        <Header />

        <section className="input-zone">
          <ChatInput
            value={query}
            onChange={setQuery}
            onSubmit={handleSend}
            disabled={isLoading}
            canSubmit={canSubmit}
          />
        </section>

        {isLoading && <LoadingSpinner />}

        {error && (
          <section className="error-box" role="alert">
            {error}
          </section>
        )}

        {response && !isLoading && <ResponseCard data={response} />}
      </main>
    </div>
  );
}
