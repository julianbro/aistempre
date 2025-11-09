'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import DatasetList from '@/components/DatasetList';
import DatasetChart from '@/components/DatasetChart';
import ValidationModal from '@/components/ValidationModal';

interface Dataset {
  symbol: string;
  timeframe: string;
  filename: string;
  row_count: number;
  date_start?: string;
  date_end?: string;
  columns: string[];
}

interface ValidationReport {
  filename: string;
  is_valid: boolean;
  total_rows: number;
  issues: Array<{
    severity: string;
    message: string;
    row_number?: number;
  }>;
  schema_valid: boolean;
  timezone_valid: boolean;
  has_gaps: boolean;
  gap_count: number;
  inferred_dtypes: Record<string, string>;
  date_range?: {
    start: string;
    end: string;
  };
}

export default function DataExplorer() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
  const [validationReport, setValidationReport] = useState<ValidationReport | null>(null);
  const [showValidationModal, setShowValidationModal] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch datasets on mount
  useEffect(() => {
    fetchDatasets();
  }, []);

  const fetchDatasets = async () => {
    try {
      setLoading(true);
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/datasets`);
      if (!response.ok) {
        throw new Error('Failed to fetch datasets');
      }
      const data = await response.json();
      setDatasets(data.datasets);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load datasets');
    } finally {
      setLoading(false);
    }
  };

  const handleValidateDataset = async (filename: string) => {
    try {
      setLoading(true);
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/datasets/validate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ filename }),
      });
      if (!response.ok) {
        throw new Error('Validation failed');
      }
      const report = await response.json();
      setValidationReport(report);
      setShowValidationModal(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Validation failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen flex-col bg-zinc-50 dark:bg-zinc-900">
      {/* Header */}
      <header className="border-b border-zinc-200 bg-white px-6 py-4 dark:border-zinc-800 dark:bg-zinc-950">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-zinc-900 dark:text-zinc-50">
              Data Explorer
            </h1>
            <p className="text-sm text-zinc-600 dark:text-zinc-400">
              Browse and validate CSV datasets
            </p>
          </div>
          <Link
            href="/"
            className="rounded-md border border-zinc-300 px-4 py-2 text-sm font-semibold text-zinc-900 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-50 dark:hover:bg-zinc-800"
          >
            Back to Home
          </Link>
        </div>
      </header>

      {/* Main content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar - Dataset List */}
        <aside className="w-80 overflow-y-auto border-r border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-950">
          <div className="p-4">
            <h2 className="mb-4 text-lg font-semibold text-zinc-900 dark:text-zinc-50">
              Datasets
            </h2>
            {loading && datasets.length === 0 && (
              <p className="text-sm text-zinc-600 dark:text-zinc-400">Loading...</p>
            )}
            {error && (
              <div className="rounded-md bg-red-50 p-3 dark:bg-red-900/20">
                <p className="text-sm text-red-800 dark:text-red-200">{error}</p>
              </div>
            )}
            <DatasetList
              datasets={datasets}
              selectedDataset={selectedDataset}
              onSelectDataset={setSelectedDataset}
              onValidateDataset={handleValidateDataset}
            />
          </div>
        </aside>

        {/* Main panel - Chart */}
        <main className="flex-1 overflow-y-auto p-6">
          {selectedDataset ? (
            <DatasetChart filename={selectedDataset} />
          ) : (
            <div className="flex h-full items-center justify-center">
              <div className="text-center">
                <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50">
                  No dataset selected
                </h3>
                <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
                  Select a dataset from the sidebar to view its chart and details
                </p>
              </div>
            </div>
          )}
        </main>
      </div>

      {/* Validation Modal */}
      {showValidationModal && validationReport && (
        <ValidationModal
          report={validationReport}
          onClose={() => setShowValidationModal(false)}
        />
      )}
    </div>
  );
}
