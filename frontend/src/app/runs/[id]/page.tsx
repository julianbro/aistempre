'use client';

import { useState, useEffect, useRef } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';

interface RunMetrics {
  epoch?: number;
  step?: number;
  train_loss?: number;
  val_loss?: number;
  val_da?: number;
  val_f1?: number;
  val_rmse?: number;
  learning_rate?: number;
  gpu_memory_mb?: number;
  gpu_utilization?: number;
  eta_minutes?: number;
}

interface CheckpointInfo {
  filename: string;
  epoch: number;
  val_loss?: number;
  val_score?: number;
  size_mb?: number;
  created_at: string;
}

interface TrainingRun {
  id: string;
  status: string;
  config: any;
  metrics?: RunMetrics;
  checkpoints: CheckpointInfo[];
  created_at: string;
  started_at?: string;
  completed_at?: string;
  error?: string;
}

export default function RunDashboard() {
  const params = useParams();
  const runId = params?.id as string;

  const [run, setRun] = useState<TrainingRun | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [wsConnected, setWsConnected] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const logsEndRef = useRef<HTMLDivElement>(null);

  // Fetch run details
  useEffect(() => {
    if (!runId) return;

    const fetchRun = async () => {
      try {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
        const response = await fetch(`${apiUrl}/runs/${runId}`);
        if (!response.ok) {
          throw new Error('Failed to fetch run details');
        }
        const data = await response.json();
        setRun(data);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load run');
      } finally {
        setLoading(false);
      }
    };

    fetchRun();
    const interval = setInterval(fetchRun, 5000); // Refresh every 5 seconds

    return () => clearInterval(interval);
  }, [runId]);

  // WebSocket connection for live updates
  useEffect(() => {
    if (!runId) return;

    const wsUrl =
      process.env.NEXT_PUBLIC_WS_URL ||
      `ws://localhost:8000/runs/${runId}/stream`;

    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      setWsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.event_type === 'metrics' && data.data) {
          setRun((prev) =>
            prev ? { ...prev, metrics: data.data } : prev
          );
        } else if (data.event_type === 'log' && data.data?.message) {
          setLogs((prev) => [...prev, data.data.message]);
        } else if (data.event_type === 'status' && data.data?.status) {
          setRun((prev) =>
            prev ? { ...prev, status: data.data.status } : prev
          );
        } else if (data.event_type === 'checkpoint' && data.data?.checkpoints) {
          setRun((prev) =>
            prev ? { ...prev, checkpoints: data.data.checkpoints } : prev
          );
        }
      } catch (err) {
        console.error('WebSocket message error:', err);
      }
    };

    ws.onerror = () => {
      setWsConnected(false);
    };

    ws.onclose = () => {
      setWsConnected(false);
    };

    wsRef.current = ws;

    return () => {
      ws.close();
    };
  }, [runId]);

  // Auto-scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const handleCancel = async () => {
    if (!runId) return;

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/runs/${runId}/cancel`, {
        method: 'POST',
      });
      if (!response.ok) {
        throw new Error('Failed to cancel run');
      }
      // Refresh run details
      const runResponse = await fetch(`${apiUrl}/runs/${runId}`);
      const data = await runResponse.json();
      setRun(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to cancel run');
    }
  };

  const downloadArtifact = (filename: string) => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    window.open(`${apiUrl}/runs/${runId}/artifacts/${filename}`, '_blank');
  };

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="text-lg">Loading...</div>
      </div>
    );
  }

  if (error || !run) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="text-lg text-red-600">{error || 'Run not found'}</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-zinc-900">
      {/* Header */}
      <header className="border-b border-zinc-200 bg-white px-6 py-4 dark:border-zinc-800 dark:bg-zinc-950">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-zinc-900 dark:text-zinc-50">
              Training Run: {runId}
            </h1>
            <div className="mt-1 flex items-center gap-4">
              <span
                className={`inline-block rounded-full px-3 py-1 text-xs font-semibold ${
                  run.status === 'running'
                    ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-200'
                    : run.status === 'completed'
                      ? 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-200'
                      : run.status === 'failed'
                        ? 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-200'
                        : run.status === 'cancelled'
                          ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-200'
                          : 'bg-zinc-100 text-zinc-800 dark:bg-zinc-800 dark:text-zinc-200'
                }`}
              >
                {run.status.toUpperCase()}
              </span>
              <span
                className={`text-xs ${wsConnected ? 'text-green-600' : 'text-red-600'}`}
              >
                {wsConnected ? '● Live' : '○ Disconnected'}
              </span>
            </div>
          </div>
          <div className="flex gap-2">
            {run.status === 'running' && (
              <button
                onClick={handleCancel}
                className="rounded-md bg-red-600 px-4 py-2 text-sm font-semibold text-white hover:bg-red-700"
              >
                Cancel
              </button>
            )}
            <Link
              href="/"
              className="rounded-md border border-zinc-300 px-4 py-2 text-sm font-semibold text-zinc-900 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-50 dark:hover:bg-zinc-800"
            >
              Back to Home
            </Link>
          </div>
        </div>
      </header>

      <div className="p-6">
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          {/* Metrics Panel */}
          <div className="rounded-lg bg-white p-6 shadow dark:bg-zinc-950">
            <h2 className="mb-4 text-xl font-semibold text-zinc-900 dark:text-zinc-50">
              Training Metrics
            </h2>
            {run.metrics ? (
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-sm text-zinc-600 dark:text-zinc-400">Epoch</div>
                  <div className="text-2xl font-bold">{run.metrics.epoch || 'N/A'}</div>
                </div>
                <div>
                  <div className="text-sm text-zinc-600 dark:text-zinc-400">Step</div>
                  <div className="text-2xl font-bold">{run.metrics.step || 'N/A'}</div>
                </div>
                <div>
                  <div className="text-sm text-zinc-600 dark:text-zinc-400">
                    Train Loss
                  </div>
                  <div className="text-2xl font-bold">
                    {run.metrics.train_loss?.toFixed(4) || 'N/A'}
                  </div>
                </div>
                <div>
                  <div className="text-sm text-zinc-600 dark:text-zinc-400">Val Loss</div>
                  <div className="text-2xl font-bold">
                    {run.metrics.val_loss?.toFixed(4) || 'N/A'}
                  </div>
                </div>
                <div>
                  <div className="text-sm text-zinc-600 dark:text-zinc-400">
                    Val DA (%)
                  </div>
                  <div className="text-2xl font-bold">
                    {run.metrics.val_da
                      ? (run.metrics.val_da * 100).toFixed(2)
                      : 'N/A'}
                  </div>
                </div>
                <div>
                  <div className="text-sm text-zinc-600 dark:text-zinc-400">Val F1</div>
                  <div className="text-2xl font-bold">
                    {run.metrics.val_f1?.toFixed(4) || 'N/A'}
                  </div>
                </div>
                <div>
                  <div className="text-sm text-zinc-600 dark:text-zinc-400">Val RMSE</div>
                  <div className="text-2xl font-bold">
                    {run.metrics.val_rmse?.toFixed(4) || 'N/A'}
                  </div>
                </div>
                <div>
                  <div className="text-sm text-zinc-600 dark:text-zinc-400">
                    Learning Rate
                  </div>
                  <div className="text-2xl font-bold">
                    {run.metrics.learning_rate?.toExponential(2) || 'N/A'}
                  </div>
                </div>
              </div>
            ) : (
              <p className="text-sm text-zinc-600 dark:text-zinc-400">
                No metrics available yet
              </p>
            )}
          </div>

          {/* GPU Panel */}
          <div className="rounded-lg bg-white p-6 shadow dark:bg-zinc-950">
            <h2 className="mb-4 text-xl font-semibold text-zinc-900 dark:text-zinc-50">
              System Info
            </h2>
            {run.metrics ? (
              <div className="space-y-4">
                <div>
                  <div className="text-sm text-zinc-600 dark:text-zinc-400">
                    GPU Memory
                  </div>
                  <div className="text-2xl font-bold">
                    {run.metrics.gpu_memory_mb?.toFixed(0) || 'N/A'} MB
                  </div>
                </div>
                <div>
                  <div className="text-sm text-zinc-600 dark:text-zinc-400">
                    GPU Utilization
                  </div>
                  <div className="text-2xl font-bold">
                    {run.metrics.gpu_utilization?.toFixed(1) || 'N/A'}%
                  </div>
                </div>
                <div>
                  <div className="text-sm text-zinc-600 dark:text-zinc-400">ETA</div>
                  <div className="text-2xl font-bold">
                    {run.metrics.eta_minutes
                      ? `${run.metrics.eta_minutes.toFixed(1)} min`
                      : 'N/A'}
                  </div>
                </div>
              </div>
            ) : (
              <p className="text-sm text-zinc-600 dark:text-zinc-400">
                No system info available yet
              </p>
            )}
          </div>

          {/* Checkpoints Panel */}
          <div className="rounded-lg bg-white p-6 shadow dark:bg-zinc-950">
            <h2 className="mb-4 text-xl font-semibold text-zinc-900 dark:text-zinc-50">
              Checkpoints
            </h2>
            {run.checkpoints.length > 0 ? (
              <div className="space-y-2">
                {run.checkpoints.map((checkpoint, idx) => (
                  <div
                    key={idx}
                    className="flex items-center justify-between rounded-md border border-zinc-200 p-3 dark:border-zinc-800"
                  >
                    <div>
                      <div className="font-semibold">{checkpoint.filename}</div>
                      <div className="text-xs text-zinc-600 dark:text-zinc-400">
                        Epoch {checkpoint.epoch} • Val Loss:{' '}
                        {checkpoint.val_loss?.toFixed(4) || 'N/A'}
                      </div>
                    </div>
                    <button
                      onClick={() => downloadArtifact(checkpoint.filename)}
                      className="rounded-md border border-zinc-300 px-3 py-1 text-sm hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-800"
                    >
                      Download
                    </button>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-zinc-600 dark:text-zinc-400">
                No checkpoints available yet
              </p>
            )}
          </div>

          {/* Console Logs */}
          <div className="rounded-lg bg-white p-6 shadow dark:bg-zinc-950">
            <h2 className="mb-4 text-xl font-semibold text-zinc-900 dark:text-zinc-50">
              Console Logs
            </h2>
            <div className="h-64 overflow-y-auto rounded-md bg-zinc-900 p-3 font-mono text-xs text-zinc-100">
              {logs.length > 0 ? (
                <>
                  {logs.map((log, idx) => (
                    <div key={idx}>{log}</div>
                  ))}
                  <div ref={logsEndRef} />
                </>
              ) : (
                <div className="text-zinc-500">No logs available yet</div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
