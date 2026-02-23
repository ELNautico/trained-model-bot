import { Injectable } from '@nestjs/common';
import * as path from 'node:path';
import * as fs from 'node:fs';
import type { PaperPosition, PaperTrade, PaperSummary, EquityPoint } from './paper.types';

// eslint-disable-next-line @typescript-eslint/no-require-imports
const Database = require('better-sqlite3');

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
      return db.prepare('SELECT * FROM paper_positions ORDER BY entry_ts ASC').all() as PaperPosition[];
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
      return db.prepare('SELECT * FROM paper_trades ORDER BY exit_ts ASC').all() as PaperTrade[];
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
    };
  }

  getEquity(): EquityPoint[] {
    const trades = this.getTrades();
    if (trades.length === 0) return [];

    const INITIAL = 100_000;
    let equity = INITIAL;
    return trades.map((t) => {
      equity += t.pnl ?? 0;
      // Use date portion of exit_ts as the label
      const date = (t.exit_ts ?? '').slice(0, 10);
      return { date, equity, action: null };
    });
  }
}
