'use client';

interface Dataset {
  symbol: string;
  timeframe: string;
  filename: string;
  row_count: number;
  date_start?: string;
  date_end?: string;
  columns: string[];
}

interface DatasetListProps {
  datasets: Dataset[];
  selectedDataset: string | null;
  onSelectDataset: (filename: string) => void;
  onValidateDataset: (filename: string) => void;
}

export default function DatasetList({
  datasets,
  selectedDataset,
  onSelectDataset,
  onValidateDataset,
}: DatasetListProps) {
  // Group datasets by symbol
  const groupedDatasets = datasets.reduce(
    (acc, dataset) => {
      if (!acc[dataset.symbol]) {
        acc[dataset.symbol] = [];
      }
      acc[dataset.symbol].push(dataset);
      return acc;
    },
    {} as Record<string, Dataset[]>,
  );

  return (
    <div className="space-y-4">
      {Object.entries(groupedDatasets).map(([symbol, symbolDatasets]) => (
        <div key={symbol} className="rounded-lg border border-zinc-200 dark:border-zinc-800">
          <div className="border-b border-zinc-200 bg-zinc-50 px-3 py-2 dark:border-zinc-800 dark:bg-zinc-900">
            <h3 className="font-semibold text-zinc-900 dark:text-zinc-50">{symbol}</h3>
          </div>
          <div className="divide-y divide-zinc-200 dark:divide-zinc-800">
            {symbolDatasets.map((dataset) => (
              <div
                key={dataset.filename}
                className={`cursor-pointer p-3 transition-colors hover:bg-zinc-50 dark:hover:bg-zinc-900 ${
                  selectedDataset === dataset.filename
                    ? 'bg-zinc-100 dark:bg-zinc-800'
                    : ''
                }`}
                onClick={() => onSelectDataset(dataset.filename)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="rounded bg-zinc-200 px-2 py-0.5 text-xs font-medium text-zinc-700 dark:bg-zinc-700 dark:text-zinc-300">
                        {dataset.timeframe}
                      </span>
                      <span className="text-sm text-zinc-600 dark:text-zinc-400">
                        {dataset.row_count.toLocaleString()} rows
                      </span>
                    </div>
                    {dataset.date_start && dataset.date_end && (
                      <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-500">
                        {new Date(dataset.date_start).toLocaleDateString()} -{' '}
                        {new Date(dataset.date_end).toLocaleDateString()}
                      </p>
                    )}
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onValidateDataset(dataset.filename);
                    }}
                    className="rounded bg-blue-600 px-2 py-1 text-xs font-medium text-white hover:bg-blue-700"
                  >
                    Validate
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      ))}
      {datasets.length === 0 && (
        <div className="rounded-lg border border-zinc-200 p-4 text-center dark:border-zinc-800">
          <p className="text-sm text-zinc-600 dark:text-zinc-400">No datasets found</p>
          <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-500">
            Add CSV files to the data directory
          </p>
        </div>
      )}
    </div>
  );
}
