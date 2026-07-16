export default function LoadingSpinner() {
  return (
    <section className="loading" aria-live="polite">
      <div className="spinner" />
      <p>Running supervisor and agents...</p>
    </section>
  );
}
