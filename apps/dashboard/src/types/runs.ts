export type RunSummary = {
  runId: string;
  ticker: string;
  tag: string | null;
  fileName: string;
  filePath?: string;
  createdAt: string;
  rows: number;
  best?: {
    objective: number | null;
    holdoutSharpe: number | null;
    holdoutReturn: number | null;
    holdoutMaxDrawdown: number | null;
    holdoutTrades: number | null;
    status: string | null;
    entryMinEv: number | null;
    retrainEvery: number | null;
    lookback: number | null;
  };
};

export type RunDetail = {
  run: RunSummary;
  columns: string[];
  rows: Array<Record<string, any>>;
};

/**
 * Top-K artifacts (produced by sweep_backtests.py with --save-top-k).
 * Used by the RunDetailPage "Top-K artifacts" section.
 */
export type RunTopKItem = {
  rank: number;          // 1..K
  hasTrades: boolean;    // trades.csv exists
  hasEquity: boolean;    // equity.csv exists
  tradesPath?: string;   // optional: relative/abs path for debugging
  equityPath?: string;   // optional: relative/abs path for debugging
};

/** One data point from equity.csv */
export type EquityPoint = {
  date: string;
  equity: number;
  action: string | null;
};
