export interface PaperPosition {
  id: number;
  ticker: string;
  entry_ts: string;
  entry_px: number;
  shares: number;
  stop_px: number;
  target_px: number;
  account_balance: number;
}

export interface PaperTrade {
  id: number;
  ticker: string;
  entry_ts: string;
  exit_ts: string;
  entry_px: number;
  exit_px: number;
  shares: number;
  pnl: number;
  return_pct: number;
  exit_reason: string;
}

export interface PaperSummary {
  totalPnl: number;
  winRate: number;
  totalTrades: number;
  avgReturnPct: number;
  bestTradePnl: number | null;
  worstTradePnl: number | null;
  avgHoldDays: number | null;
  openPositions: number;
  exitReasons: Record<string, number>;
}

export interface PaperEquityPoint {
  date: string;
  equity: number;
  action: null;
}
