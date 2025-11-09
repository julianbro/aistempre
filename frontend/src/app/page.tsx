export default function Home() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-zinc-50 dark:bg-zinc-900">
      <main className="flex flex-col items-center justify-center gap-8 px-8 py-16">
        <h1 className="text-4xl font-bold tracking-tight text-zinc-900 dark:text-zinc-50 sm:text-6xl">
          AI Trading Platform
        </h1>
        <p className="max-w-2xl text-center text-lg text-zinc-600 dark:text-zinc-400">
          Multi-input, multi-horizon, probabilistic Transformer for financial time-series prediction
        </p>
        <div className="flex gap-4">
          <a
            className="rounded-md bg-zinc-900 px-6 py-3 text-sm font-semibold text-white shadow-sm hover:bg-zinc-700 dark:bg-zinc-50 dark:text-zinc-900 dark:hover:bg-zinc-200"
            href="/docs"
          >
            Get Started
          </a>
          <a
            className="rounded-md border border-zinc-300 px-6 py-3 text-sm font-semibold text-zinc-900 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-50 dark:hover:bg-zinc-800"
            href="https://github.com/julianbro/aistempre"
            target="_blank"
            rel="noopener noreferrer"
          >
            View on GitHub
          </a>
        </div>
      </main>
    </div>
  );
}
