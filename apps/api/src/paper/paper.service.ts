import { Injectable } from '@nestjs/common';
import * as path from 'node:path';
import * as fs from 'node:fs';
import type {
  PaperPosition,
  PaperTrade,
  PaperSummary,
  EquityPoint,
  TickerStat,
} from './paper.types';

// eslint-disable-next-line @typescript-eslint/no-require-imports
const Database = require('better-sqlite3');

const INITIAL_EQUITY = 100_000;

@Injectable()
export class PaperService {
  private resolveDbPath(): string | null {
    const candidates = [
      process.env.BOTDB_PATH,
      path.resolve(process.cwd(), 'core/bot.db'),
      path.resolve(process.cwd(), '../../core/bot.db'),
      path.resolve(__dirname, '../../../../core/bot.db'),
    ].filter(Boolean) as string[];

    for (const c of candidates) {
      if (fs.existsSync(c)) return c;
    }
    return null;
  }

  private tableExists(db: InstanceType<typeof Database>, name: string): boolean {
    const row = db
      .prepare(`SELECT name FROM sqlite_master WHERE type='table' AND name=?`)
      .get(name) as { name: string } | undefined;
    return !!row;
  }

  getPositions(): PaperPosition[] {
    const dbPath = this.resolveDbPath();
    if (!dbPath) return [];

    const db = new Database(dbPath, { readonly: true });
    try {
      if (!this.tableExists(db, 'paper_positions')) return [];
      return db
        .prepare('SELECT * FROM paper_positions ORDER BY entry_ts ASC')
        .all() as PaperPosition[];
    } finally {
      db.close();
    }
  }

  getTrades(): PaperTrade[] {
    const dbPath = this.resolveDbPath();
    if (!dbPath) return [];

    const db = new Database(dbPath, { readonly: true });
    try {
      if (!this.tableExists(db, 'paper_trades')) return [];
      return db
        .prepare('SELECT * FROM paper_trades ORDER BY exit_ts ASC')
        .all() as PaperTrade[];
    } finally {
      db.close();
    }
  }

  getSummary(): PaperSummary {
    const trades = this.getTrades();
    const positions = this.getPositions();

    const totalTrades = trades.length;
    const totalPnl = trades.reduce((s, t) => s + (t.pnl ?? 0), 0);
    const wins = trades.filter((t) => (t.pnl ?? 0) > 0).length;
    const winRate = totalTrades > 0 ? wins / totalTrades : 0;

    const avgReturnPct =
      totalTrades > 0
        ? trades.reduce((s, t) => s + (t.return_pct ?? 0), 0) / totalTrades
        : 0;

    const pnls = trades.map((t) => t.pnl ?? 0);
    const bestTradePnl = pnls.length > 0 ? Math.max(...pnls) : null;
    const worstTradePnl = pnls.length > 0 ? Math.min(...pnls) : null;

    let avgHoldDays: number | null = null;
    if (totalTrades > 0) {
      const totalMs = trades.reduce((s, t) => {
        const entry = new Date(t.entry_ts).getTime();
        const exit = new Date(t.exit_ts).getTime();
        return s + (exit - entry);
      }, 0);
      avgHoldDays = totalMs / totalTrades / (1000 * 60 * 60 * 24);
    }

    const exitReasons: Record<string, number> = {};
    for (const t of trades) {
      const r = t.exit_reason ?? 'unknown';
      exitReasons[r] = (exitReasons[r] ?? 0) + 1;
    }

    // ── Profit Factor ────────────────────────────────────────────────────────
    const grossWins = trades
      .filter((t) => (t.pnl ?? 0) > 0)
      .reduce((s, t) => s + (t.pnl ?? 0), 0);
    const grossLosses = Math.abs(
      trades
        .filter((t) => (t.pnl ?? 0) < 0)
        .reduce((s, t) => s + (t.pnl ?? 0), 0),
    );
    const profitFactor: number | null = grossLosses > 0 ? grossWins / grossLosses : null;

    // ── Expectancy (avg $ per trade) ─────────────────────────────────────────
    const expectancy = totalTrades > 0 ? totalPnl / totalTrades : 0;

    // ── Sharpe (trade-level, annualised) ─────────────────────────────────────
    let sharpeRatio: number | null = null;
    if (totalTrades >= 3 && avgHoldDays) {
      const returns = trades.map((t) => (t.return_pct ?? 0) / 100);
      const meanR = returns.reduce((a, b) => a + b, 0) / returns.length;
      const variance =
        returns.reduce((s, r) => s + Math.pow(r - meanR, 2), 0) / returns.length;
      const stdR = Math.sqrt(variance);
      if (stdR > 0) {
        const tradesPerYear = 252 / avgHoldDays;
        sharpeRatio = (meanR / stdR) * Math.sqrt(tradesPerYear);
      }
    }

    // ── CAGR ─────────────────────────────────────────────────────────────────
    let cagr: number | null = null;
    if (totalTrades > 0 && trades[0]?.entry_ts && trades[totalTrades - 1]?.exit_ts) {
      const firstEntry = new Date(trades[0].entry_ts).getTime();
      const lastExit = new Date(trades[totalTrades - 1].exit_ts).getTime();
      const years = (lastExit - firstEntry) / (1000 * 60 * 60 * 24 * 365.25);
      const finalEquity = INITIAL_EQUITY + totalPnl;
      if (years >= 0.05 && finalEquity > 0) {
        cagr = Math.pow(finalEquity / INITIAL_EQUITY, 1 / years) - 1;
      }
    }

    // ── Max Drawdown ─────────────────────────────────────────────────────────
    let maxDrawdownPct = 0;
    {
      let equity = INITIAL_EQUITY;
      let peak = INITIAL_EQUITY;
      for (const t of trades) {
        equity += t.pnl ?? 0;
        if (equity > peak) peak = equity;
        const dd = Math.abs(((equity - peak) / peak) * 100);
        if (dd > maxDrawdownPct) maxDrawdownPct = dd;
      }
    }

    // ── Max Consecutive Losses ───────────────────────────────────────────────
    let maxConsecLosses = 0;
    let curConsec = 0;
    for (const t of trades) {
      if ((t.pnl ?? 0) <= 0) {
        curConsec++;
        if (curConsec > maxConsecLosses) maxConsecLosses = curConsec;
      } else {
        curConsec = 0;
      }
    }

    // ── Per-Ticker Stats ─────────────────────────────────────────────────────
    const tickerMap = new Map<string, TickerStat>();
    for (const t of trades) {
      if (!tickerMap.has(t.ticker)) {
        tickerMap.set(t.ticker, {
          ticker: t.ticker,
          trades: 0,
          wins: 0,
          winRate: 0,
          totalPnl: 0,
        });
      }
      const stat = tickerMap.get(t.ticker)!;
      stat.trades++;
      if ((t.pnl ?? 0) > 0) stat.wins++;
      stat.totalPnl += t.pnl ?? 0;
    }
    for (const stat of tickerMap.values()) {
      stat.winRate = stat.trades > 0 ? stat.wins / stat.trades : 0;
    }
    const tickerStats = [...tickerMap.values()].sort((a, b) => b.totalPnl - a.totalPnl);

    return {
      totalPnl,
      winRate,
      totalTrades,
      avgReturnPct,
      bestTradePnl,
      worstTradePnl,
      avgHoldDays,
      openPositions: positions.length,
      exitReasons,
      maxDrawdownPct,
      sharpeRatio,
      profitFactor,
      expectancy,
      cagr,
      maxConsecLosses,
      tickerStats,
    };
  }

  getEquity(): EquityPoint[] {
    const trades = this.getTrades();
    if (trades.length === 0) return [];

    let equity = INITIAL_EQUITY;
    let peak = INITIAL_EQUITY;

    return trades.map((t) => {
      equity += t.pnl ?? 0;
      if (equity > peak) peak = equity;
      const drawdown = peak > 0 ? ((equity - peak) / peak) * 100 : 0;
      const date = (t.exit_ts ?? '').slice(0, 10);
      return { date, equity, action: null, drawdown };
    });
  }
}
