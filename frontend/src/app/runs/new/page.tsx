'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';

interface RunConfig {
  data_source: string;
  timeframes: string[];
  start_date?: string;
  end_date?: string;
  variant: 'base' | 'medium' | 'large';
  features?: string[];
  enable_technical_indicators: boolean;
  enable_calendar_features: boolean;
  max_epochs: number;
  batch_size: number;
  learning_rate: number;
  weight_decay: number;
  loss_weights?: Record<string, number>;
  regression_loss: string;
  next_horizon: number;
  short_horizon: string;
  long_horizon: string;
}

const PRESETS = {
  base: {
    variant: 'base' as const,
    max_epochs: 100,
    batch_size: 32,
    learning_rate: 0.0002,
  },
  medium: {
    variant: 'medium' as const,
    max_epochs: 150,
    batch_size: 64,
    learning_rate: 0.0001,
  },
  large: {
    variant: 'large' as const,
    max_epochs: 200,
    batch_size: 128,
    learning_rate: 0.00005,
  },
};

const AVAILABLE_TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'];
const TECHNICAL_INDICATORS = ['RSI', 'MACD', 'ATR', 'BB', 'EMA', 'SMA', 'OBV'];

export default function NewRun() {
  const router = useRouter();
  const [step, setStep] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [config, setConfig] = useState<RunConfig>({
    data_source: './data/example.csv',
    timeframes: ['1m', '15m', '1h'],
    variant: 'base',
    enable_technical_indicators: true,
    enable_calendar_features: true,
    max_epochs: 100,
    batch_size: 32,
    learning_rate: 0.0002,
    weight_decay: 0.05,
    regression_loss: 'gaussian_nll',
    next_horizon: 1,
    short_horizon: '30m',
    long_horizon: '1w',
  });

  const totalSteps = 6;

  const applyPreset = (preset: keyof typeof PRESETS) => {
    setConfig({ ...config, ...PRESETS[preset] });
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/runs`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ config }),
      });

      if (!response.ok) {
        throw new Error('Failed to create training run');
      }

      const data = await response.json();
      router.push(`/runs/${data.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create run');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-zinc-900">
      {/* Header */}
      <header className="border-b border-zinc-200 bg-white px-6 py-4 dark:border-zinc-800 dark:bg-zinc-950">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-zinc-900 dark:text-zinc-50">
              New Training Run
            </h1>
            <p className="text-sm text-zinc-600 dark:text-zinc-400">
              Configure your model and training parameters
            </p>
          </div>
          <Link
            href="/"
            className="rounded-md border border-zinc-300 px-4 py-2 text-sm font-semibold text-zinc-900 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-50 dark:hover:bg-zinc-800"
          >
            Cancel
          </Link>
        </div>
      </header>

      <div className="mx-auto max-w-4xl p-6">
        {/* Progress Indicator */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            {Array.from({ length: totalSteps }, (_, i) => i + 1).map((s) => (
              <div
                key={s}
                className={`flex h-10 w-10 items-center justify-center rounded-full text-sm font-semibold ${
                  s === step
                    ? 'bg-blue-600 text-white'
                    : s < step
                      ? 'bg-green-600 text-white'
                      : 'bg-zinc-200 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400'
                }`}
              >
                {s}
              </div>
            ))}
          </div>
          <div className="mt-2 flex justify-between text-xs text-zinc-600 dark:text-zinc-400">
            <span>Data</span>
            <span>Features</span>
            <span>Targets</span>
            <span>Model</span>
            <span>Loss</span>
            <span>Train</span>
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-4 rounded-md bg-red-50 p-4 dark:bg-red-900/20">
            <p className="text-sm text-red-800 dark:text-red-200">{error}</p>
          </div>
        )}

        {/* Step Content */}
        <div className="rounded-lg bg-white p-6 shadow dark:bg-zinc-950">
          {/* Step 1: Data Selection */}
          {step === 1 && (
            <div>
              <h2 className="mb-4 text-xl font-semibold text-zinc-900 dark:text-zinc-50">
                Data Configuration
              </h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300">
                    Data Source
                  </label>
                  <input
                    type="text"
                    value={config.data_source}
                    onChange={(e) =>
                      setConfig({ ...config, data_source: e.target.value })
                    }
                    className="mt-1 block w-full rounded-md border border-zinc-300 px-3 py-2 dark:border-zinc-700 dark:bg-zinc-900"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300">
                    Timeframes
                  </label>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {AVAILABLE_TIMEFRAMES.map((tf) => (
                      <label key={tf} className="flex items-center">
                        <input
                          type="checkbox"
                          checked={config.timeframes.includes(tf)}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setConfig({
                                ...config,
                                timeframes: [...config.timeframes, tf],
                              });
                            } else {
                              setConfig({
                                ...config,
                                timeframes: config.timeframes.filter((t) => t !== tf),
                              });
                            }
                          }}
                          className="mr-2"
                        />
                        <span className="text-sm">{tf}</span>
                      </label>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Step 2: Features */}
          {step === 2 && (
            <div>
              <h2 className="mb-4 text-xl font-semibold text-zinc-900 dark:text-zinc-50">
                Feature Configuration
              </h2>
              <div className="space-y-4">
                <div className="flex items-center">
                  <input
                    type="checkbox"
                    checked={config.enable_technical_indicators}
                    onChange={(e) =>
                      setConfig({
                        ...config,
                        enable_technical_indicators: e.target.checked,
                      })
                    }
                    className="mr-2"
                  />
                  <label className="text-sm font-medium">
                    Enable Technical Indicators
                  </label>
                </div>
                <div className="flex items-center">
                  <input
                    type="checkbox"
                    checked={config.enable_calendar_features}
                    onChange={(e) =>
                      setConfig({
                        ...config,
                        enable_calendar_features: e.target.checked,
                      })
                    }
                    className="mr-2"
                  />
                  <label className="text-sm font-medium">
                    Enable Calendar Features
                  </label>
                </div>
              </div>
            </div>
          )}

          {/* Step 3: Targets/Horizons */}
          {step === 3 && (
            <div>
              <h2 className="mb-4 text-xl font-semibold text-zinc-900 dark:text-zinc-50">
                Prediction Horizons
              </h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300">
                    Next Horizon (bars)
                  </label>
                  <input
                    type="number"
                    value={config.next_horizon}
                    onChange={(e) =>
                      setConfig({ ...config, next_horizon: parseInt(e.target.value) })
                    }
                    className="mt-1 block w-full rounded-md border border-zinc-300 px-3 py-2 dark:border-zinc-700 dark:bg-zinc-900"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300">
                    Short-term Horizon
                  </label>
                  <input
                    type="text"
                    value={config.short_horizon}
                    onChange={(e) =>
                      setConfig({ ...config, short_horizon: e.target.value })
                    }
                    className="mt-1 block w-full rounded-md border border-zinc-300 px-3 py-2 dark:border-zinc-700 dark:bg-zinc-900"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300">
                    Long-term Horizon
                  </label>
                  <input
                    type="text"
                    value={config.long_horizon}
                    onChange={(e) =>
                      setConfig({ ...config, long_horizon: e.target.value })
                    }
                    className="mt-1 block w-full rounded-md border border-zinc-300 px-3 py-2 dark:border-zinc-700 dark:bg-zinc-900"
                  />
                </div>
              </div>
            </div>
          )}

          {/* Step 4: Model Configuration */}
          {step === 4 && (
            <div>
              <h2 className="mb-4 text-xl font-semibold text-zinc-900 dark:text-zinc-50">
                Model Configuration
              </h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300">
                    Model Variant
                  </label>
                  <div className="mt-2 grid grid-cols-3 gap-4">
                    {Object.keys(PRESETS).map((preset) => (
                      <button
                        key={preset}
                        onClick={() => applyPreset(preset as keyof typeof PRESETS)}
                        className={`rounded-md border p-4 text-center ${
                          config.variant === preset
                            ? 'border-blue-600 bg-blue-50 dark:bg-blue-900/20'
                            : 'border-zinc-300 dark:border-zinc-700'
                        }`}
                      >
                        <div className="font-semibold capitalize">{preset}</div>
                        <div className="mt-1 text-xs text-zinc-600 dark:text-zinc-400">
                          {preset === 'base' && 'd_model: 256, 2 layers'}
                          {preset === 'medium' && 'd_model: 512, 3 layers'}
                          {preset === 'large' && 'd_model: 768, 4 layers'}
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Step 5: Loss Configuration */}
          {step === 5 && (
            <div>
              <h2 className="mb-4 text-xl font-semibold text-zinc-900 dark:text-zinc-50">
                Loss Configuration
              </h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300">
                    Regression Loss
                  </label>
                  <select
                    value={config.regression_loss}
                    onChange={(e) =>
                      setConfig({ ...config, regression_loss: e.target.value })
                    }
                    className="mt-1 block w-full rounded-md border border-zinc-300 px-3 py-2 dark:border-zinc-700 dark:bg-zinc-900"
                  >
                    <option value="gaussian_nll">Gaussian NLL</option>
                    <option value="student_t">Student-t</option>
                    <option value="quantile">Quantile</option>
                    <option value="deterministic">Deterministic (MSE)</option>
                  </select>
                </div>
              </div>
            </div>
          )}

          {/* Step 6: Training Parameters */}
          {step === 6 && (
            <div>
              <h2 className="mb-4 text-xl font-semibold text-zinc-900 dark:text-zinc-50">
                Training Parameters
              </h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300">
                    Max Epochs
                  </label>
                  <input
                    type="number"
                    value={config.max_epochs}
                    onChange={(e) =>
                      setConfig({ ...config, max_epochs: parseInt(e.target.value) })
                    }
                    className="mt-1 block w-full rounded-md border border-zinc-300 px-3 py-2 dark:border-zinc-700 dark:bg-zinc-900"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300">
                    Batch Size
                  </label>
                  <input
                    type="number"
                    value={config.batch_size}
                    onChange={(e) =>
                      setConfig({ ...config, batch_size: parseInt(e.target.value) })
                    }
                    className="mt-1 block w-full rounded-md border border-zinc-300 px-3 py-2 dark:border-zinc-700 dark:bg-zinc-900"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300">
                    Learning Rate
                  </label>
                  <input
                    type="number"
                    step="0.00001"
                    value={config.learning_rate}
                    onChange={(e) =>
                      setConfig({ ...config, learning_rate: parseFloat(e.target.value) })
                    }
                    className="mt-1 block w-full rounded-md border border-zinc-300 px-3 py-2 dark:border-zinc-700 dark:bg-zinc-900"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300">
                    Weight Decay
                  </label>
                  <input
                    type="number"
                    step="0.01"
                    value={config.weight_decay}
                    onChange={(e) =>
                      setConfig({ ...config, weight_decay: parseFloat(e.target.value) })
                    }
                    className="mt-1 block w-full rounded-md border border-zinc-300 px-3 py-2 dark:border-zinc-700 dark:bg-zinc-900"
                  />
                </div>
              </div>
            </div>
          )}

          {/* Navigation Buttons */}
          <div className="mt-6 flex justify-between">
            <button
              onClick={() => setStep(Math.max(1, step - 1))}
              disabled={step === 1}
              className="rounded-md border border-zinc-300 px-4 py-2 text-sm font-semibold disabled:opacity-50 dark:border-zinc-700"
            >
              Previous
            </button>
            {step < totalSteps ? (
              <button
                onClick={() => setStep(Math.min(totalSteps, step + 1))}
                className="rounded-md bg-blue-600 px-4 py-2 text-sm font-semibold text-white hover:bg-blue-700"
              >
                Next
              </button>
            ) : (
              <button
                onClick={handleSubmit}
                disabled={loading}
                className="rounded-md bg-green-600 px-4 py-2 text-sm font-semibold text-white hover:bg-green-700 disabled:opacity-50"
              >
                {loading ? 'Starting...' : 'Start Training'}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
