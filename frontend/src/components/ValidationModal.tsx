'use client';

interface ValidationIssue {
  severity: string;
  message: string;
  row_number?: number;
}

interface ValidationReport {
  filename: string;
  is_valid: boolean;
  total_rows: number;
  issues: ValidationIssue[];
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

interface ValidationModalProps {
  report: ValidationReport;
  onClose: () => void;
}

export default function ValidationModal({ report, onClose }: ValidationModalProps) {
  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'error':
        return 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-200';
      case 'warning':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-200';
      case 'info':
        return 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-200';
      default:
        return 'bg-zinc-100 text-zinc-800 dark:bg-zinc-900/20 dark:text-zinc-200';
    }
  };

  const getBadgeColor = (valid: boolean) => {
    return valid
      ? 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-200'
      : 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-200';
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="max-h-[90vh] w-full max-w-4xl overflow-y-auto rounded-lg bg-white shadow-xl dark:bg-zinc-900">
        {/* Header */}
        <div className="border-b border-zinc-200 px-6 py-4 dark:border-zinc-800">
          <div className="flex items-start justify-between">
            <div>
              <h2 className="text-2xl font-bold text-zinc-900 dark:text-zinc-50">
                Validation Report
              </h2>
              <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
                {report.filename}
              </p>
            </div>
            <button
              onClick={onClose}
              className="rounded-md p-2 text-zinc-400 hover:bg-zinc-100 hover:text-zinc-600 dark:hover:bg-zinc-800 dark:hover:text-zinc-300"
            >
              <svg
                className="h-6 w-6"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="space-y-6 p-6">
          {/* Overall Status */}
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50">
                Overall Status
              </h3>
              <span
                className={`rounded-full px-3 py-1 text-sm font-medium ${getBadgeColor(report.is_valid)}`}
              >
                {report.is_valid ? '✓ Valid' : '✗ Invalid'}
              </span>
            </div>
            <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
              Total rows: {report.total_rows.toLocaleString()}
            </p>
            {report.date_range && (
              <p className="text-sm text-zinc-600 dark:text-zinc-400">
                Date range: {new Date(report.date_range.start).toLocaleString()} -{' '}
                {new Date(report.date_range.end).toLocaleString()}
              </p>
            )}
          </div>

          {/* Validation Checks */}
          <div>
            <h3 className="mb-3 text-lg font-semibold text-zinc-900 dark:text-zinc-50">
              Validation Checks
            </h3>
            <div className="space-y-2">
              <div className="flex items-center justify-between rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
                <span className="text-sm text-zinc-700 dark:text-zinc-300">Schema (OHLCV)</span>
                <span className={`rounded px-2 py-1 text-xs font-medium ${getBadgeColor(report.schema_valid)}`}>
                  {report.schema_valid ? 'Pass' : 'Fail'}
                </span>
              </div>
              <div className="flex items-center justify-between rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
                <span className="text-sm text-zinc-700 dark:text-zinc-300">Timezone (UTC)</span>
                <span className={`rounded px-2 py-1 text-xs font-medium ${getBadgeColor(report.timezone_valid)}`}>
                  {report.timezone_valid ? 'Pass' : 'Warning'}
                </span>
              </div>
              <div className="flex items-center justify-between rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
                <span className="text-sm text-zinc-700 dark:text-zinc-300">Data Gaps</span>
                <span className={`rounded px-2 py-1 text-xs font-medium ${getBadgeColor(!report.has_gaps)}`}>
                  {report.has_gaps ? `${report.gap_count} gaps found` : 'No gaps'}
                </span>
              </div>
            </div>
          </div>

          {/* Inferred Data Types */}
          <div>
            <h3 className="mb-3 text-lg font-semibold text-zinc-900 dark:text-zinc-50">
              Inferred Data Types
            </h3>
            <div className="rounded-lg border border-zinc-200 dark:border-zinc-800">
              <table className="w-full">
                <thead className="border-b border-zinc-200 bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-900">
                  <tr>
                    <th className="px-4 py-2 text-left text-sm font-semibold text-zinc-900 dark:text-zinc-50">
                      Column
                    </th>
                    <th className="px-4 py-2 text-left text-sm font-semibold text-zinc-900 dark:text-zinc-50">
                      Data Type
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-200 dark:divide-zinc-800">
                  {Object.entries(report.inferred_dtypes).map(([col, dtype]) => (
                    <tr key={col}>
                      <td className="px-4 py-2 text-sm text-zinc-700 dark:text-zinc-300">
                        {col}
                      </td>
                      <td className="px-4 py-2 text-sm text-zinc-600 dark:text-zinc-400">
                        {dtype}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Issues */}
          {report.issues.length > 0 && (
            <div>
              <h3 className="mb-3 text-lg font-semibold text-zinc-900 dark:text-zinc-50">
                Issues ({report.issues.length})
              </h3>
              <div className="space-y-2">
                {report.issues.map((issue, index) => (
                  <div
                    key={index}
                    className={`rounded-lg p-3 ${getSeverityColor(issue.severity)}`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <p className="text-sm font-medium">{issue.message}</p>
                        {issue.row_number && (
                          <p className="mt-1 text-xs opacity-75">Row: {issue.row_number}</p>
                        )}
                      </div>
                      <span className="ml-2 text-xs font-semibold uppercase">
                        {issue.severity}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-zinc-200 px-6 py-4 dark:border-zinc-800">
          <button
            onClick={onClose}
            className="rounded-md bg-zinc-900 px-4 py-2 text-sm font-semibold text-white hover:bg-zinc-700 dark:bg-zinc-50 dark:text-zinc-900 dark:hover:bg-zinc-200"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
