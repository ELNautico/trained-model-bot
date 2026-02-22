export type RunBest = {
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

export type RunMeta = {
  kind?: string;
  runId?: string;
  ticker?: string;
  tag?: string | null;
  cli?: Record<string, any>;
};

export type RunSummary = {
  runId: string;
  ticker: string;
  tag: string | null;
  kind?: string | null;
  fileName: string;
  createdAt: string;
  rows: number;
  best?: RunBest;
  layout?: 'structured' | 'legacy';
};

export type RunDetail = {
  run: RunSummary;
  meta: RunMeta | null;
  columns: string[];
  rows: Array<Record<string, any>>;
};

export type RunTopKItem = {
  rank: number;
  folder: string;
  hasTrades: boolean;
  hasEquity: boolean;
};

export type EquityPoint = {
  date: string;
  equity: number;
  action: string | null;
};
