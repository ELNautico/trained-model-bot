import { Injectable, NotFoundException, BadRequestException } from '@nestjs/common';
import { parse } from 'csv-parse/sync';
import * as fs from 'node:fs';
import * as fsp from 'node:fs/promises';
import * as path from 'node:path';
import { RunDetail, RunSummary, RunTopKItem, RunMeta } from './runs.types';

function toNumberOrNull(v: unknown): number | null {
  if (v === null || v === undefined) return null;
  const s = String(v).trim();
  if (!s) return null;
  const n = Number(s);
  return Number.isFinite(n) ? n : null;
}

function isSafeId(s: string): boolean {
  // allow the runId you generated: 20260113_015351Z__TSLA__SMOKE__sweep
  return /^[A-Za-z0-9._-]+(?:__[A-Za-z0-9._-]+)*$/.test(s);
}

function parseRunId(runId: string): { ticker?: string; tag?: string | null; kind?: string | null; createdAt?: string | null } {
  const parts = runId.split('__').filter(Boolean);
  // Expected:
  //  - with tag: [ts, ticker, tag, kind]
  //  - no tag : [ts, ticker, kind]
  if (parts.length < 3) return {};
  const ts = parts[0];
  const ticker = (parts[1] ?? '').toUpperCase();

  let tag: string | null = null;
  let kind: string | null = null;

  if (parts.length === 3) {
    kind = parts[2] ?? null;
  } else {
    tag = parts[2] ?? null;
    kind = parts[3] ?? null;
  }

  // ts format: YYYYMMDD_HHMMSSZ
  let createdAt: string | null = null;
  const m = ts.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})Z$/);
  if (m) {
    const [, Y, Mo, D, h, mi, s] = m;
    const d = new Date(Date.UTC(Number(Y), Number(Mo) - 1, Number(D), Number(h), Number(mi), Number(s)));
    createdAt = d.toISOString();
  }

  return { ticker: ticker || undefined, tag, kind, createdAt };
}

@Injectable()
export class RunsService {
  /**
   * Resolve the backtests directory.
   * - override: BACKTESTS_DIR env var
   * - otherwise: try cwd/backtests, else ../../backtests (when running from apps/api)
   */
  private resolveBacktestsDir(): string {
    const fromEnv = process.env.BACKTESTS_DIR?.trim();
    if (fromEnv) return path.resolve(fromEnv);

    const p1 = path.resolve(process.cwd(), 'backtests');
    if (fs.existsSync(p1)) return p1;

    const p2 = path.resolve(process.cwd(), '..', '..', 'backtests');
    return p2;
  }

  private resolveRunsRoot(): string {
    return path.resolve(this.resolveBacktestsDir(), 'runs');
  }

  // ---------- legacy flat files support ----------
  private isLegacySweepResultsFile(fileName: string): boolean {
    return /_sweep_results.*\.csv$/i.test(fileName);
  }

  private extractLegacyTickerAndTag(fileName: string): { ticker: string; tag: string | null; runId: string } {
    const base = fileName.replace(/\.csv$/i, '');
    const m = base.match(/^([A-Z0-9\.\-_]+)_sweep_results(?:_(.+))?$/i);
    if (!m) return { ticker: 'UNKNOWN', tag: null, runId: base };
    const ticker = (m[1] ?? 'UNKNOWN').toUpperCase();
    const tag = m[2] ? String(m[2]) : null;
    return { ticker, tag, runId: base };
  }

  private safeResolveLegacyCsv(runId: string): string {
    if (!/^[A-Za-z0-9._-]+$/.test(runId)) throw new BadRequestException('Invalid runId');
    const dir = this.resolveBacktestsDir();
    const candidate = path.resolve(dir, `${runId}.csv`);
    const rel = path.relative(dir, candidate);
    if (rel.startsWith('..') || path.isAbsolute(rel)) throw new BadRequestException('Invalid path');
    return candidate;
  }

  // ---------- structured runs support ----------
  private safeResolveRunDir(runId: string): string {
    if (!isSafeId(runId)) throw new BadRequestException('Invalid runId');
    const root = this.resolveRunsRoot();
    const candidate = path.resolve(root, runId);
    const rel = path.relative(root, candidate);
    if (rel.startsWith('..') || path.isAbsolute(rel)) throw new BadRequestException('Invalid path');
    return candidate;
  }

  private async readJsonIfExists<T>(absPath: string): Promise<T | null> {
    try {
      const txt = await fsp.readFile(absPath, 'utf-8');
      return JSON.parse(txt) as T;
    } catch {
      return null;
    }
  }

  private parseCsv(content: string): { columns: string[]; rows: Record<string, string>[] } {
    const records = parse(content, {
      columns: true,
      skip_empty_lines: true,
      relax_quotes: true,
      relax_column_count: true,
      trim: true,
    }) as Record<string, string>[];

    const columns = records.length > 0 ? Object.keys(records[0]) : [];
    return { columns, rows: records };
  }

  private computeBestRow(rows: Record<string, string>[]) {
    let best: Record<string, string> | null = null;
    let bestScore = -Infinity;

    for (const r of rows) {
      const status = (r['status'] ?? '').toString().toLowerCase();
      if (status && status !== 'ok') continue;

      const objective = toNumberOrNull(r['objective']);
      const sharpe = toNumberOrNull(r['holdout_sharpe']);
      const score = objective ?? sharpe ?? -1e12;

      if (score > bestScore) {
        bestScore = score;
        best = r;
      }
    }

    if (!best && rows.length > 0) best = rows[0];
    if (!best) return undefined;

    return {
      objective: toNumberOrNull(best['objective']),
      holdoutSharpe: toNumberOrNull(best['holdout_sharpe']),
      holdoutReturn: toNumberOrNull(best['holdout_total_return']),
      holdoutMaxDrawdown: toNumberOrNull(best['holdout_max_drawdown']),
      holdoutTrades: toNumberOrNull(best['holdout_trades']),
      status: best['status'] ?? null,
      entryMinEv: toNumberOrNull(best['entry_min_ev']),
      retrainEvery: toNumberOrNull(best['retrain_every']),
      lookback: toNumberOrNull(best['lookback']),
    };
  }

  private async buildStructuredSummary(runId: string): Promise<RunSummary | null> {
    const runDir = this.safeResolveRunDir(runId);
    if (!fs.existsSync(runDir)) return null;

    const sweepPath = path.resolve(runDir, 'sweep_results.csv');
    if (!fs.existsSync(sweepPath)) return null;

    const metaPath = path.resolve(runDir, 'meta.json');
    const meta = await this.readJsonIfExists<RunMeta>(metaPath);

    const stat = await fsp.stat(sweepPath);
    const parsedFromId = parseRunId(runId);

    const content = await fsp.readFile(sweepPath, 'utf-8');
    const parsed = this.parseCsv(content);

    const ticker = (meta?.ticker ?? parsedFromId.ticker ?? 'UNKNOWN').toUpperCase();
    const tag = meta?.tag ?? parsedFromId.tag ?? null;
    const kind = meta?.kind ?? parsedFromId.kind ?? null;

    // createdAt priority:
    // 1) parsed timestamp in runId (deterministic)
    // 2) file mtime
    const createdAt = parsedFromId.createdAt ?? stat.mtime.toISOString();

    return {
      runId,
      ticker,
      tag,
      kind,
      fileName: 'sweep_results.csv',
      createdAt,
      rows: parsed.rows.length,
      best: this.computeBestRow(parsed.rows),
      layout: 'structured',
    };
  }

  private async buildLegacySummary(fileName: string): Promise<RunSummary> {
    const dir = this.resolveBacktestsDir();
    const abs = path.resolve(dir, fileName);
    const stat = await fsp.stat(abs);

    const { ticker, tag, runId } = this.extractLegacyTickerAndTag(fileName);
    const content = await fsp.readFile(abs, 'utf-8');
    const parsed = this.parseCsv(content);

    return {
      runId,
      ticker,
      tag,
      kind: 'sweep',
      fileName,
      createdAt: stat.mtime.toISOString(),
      rows: parsed.rows.length,
      best: this.computeBestRow(parsed.rows),
      layout: 'legacy',
    };
  }

  async listRuns(): Promise<RunSummary[]> {
    const out: RunSummary[] = [];

    // 1) structured runs
    const runsRoot = this.resolveRunsRoot();
    if (fs.existsSync(runsRoot)) {
      const entries = await fsp.readdir(runsRoot, { withFileTypes: true });
      const dirs = entries.filter((e) => e.isDirectory()).map((e) => e.name);

      for (const runId of dirs) {
        const s = await this.buildStructuredSummary(runId);
        if (s) out.push(s);
      }
    }

    // 2) legacy flat sweep files (optional compatibility)
    const backtestsDir = this.resolveBacktestsDir();
    if (fs.existsSync(backtestsDir)) {
      const files = await fsp.readdir(backtestsDir);
      const legacy = files.filter((f) => this.isLegacySweepResultsFile(f));

      for (const fileName of legacy) {
        // Avoid duplicates if someone made a runId that matches legacy base (unlikely)
        out.push(await this.buildLegacySummary(fileName));
      }
    }

    // newest first
    out.sort((a, b) => (a.createdAt < b.createdAt ? 1 : -1));
    return out;
  }

  async getRun(runId: string): Promise<RunDetail> {
    // Prefer structured
    const runDir = this.safeResolveRunDir(runId);
    const sweepPath = path.resolve(runDir, 'sweep_results.csv');
    const metaPath = path.resolve(runDir, 'meta.json');

    if (fs.existsSync(runDir) && fs.existsSync(sweepPath)) {
      const stat = await fsp.stat(sweepPath);
      const meta = await this.readJsonIfExists<RunMeta>(metaPath);
      const parsedFromId = parseRunId(runId);

      const content = await fsp.readFile(sweepPath, 'utf-8');
      const parsed = this.parseCsv(content);

      const rows = parsed.rows.map((r) => {
        const obj: Record<string, string | number | null> = {};
        for (const [k, v] of Object.entries(r)) {
          if (k === 'status' || k === 'error' || k === 'ticker') obj[k] = v ?? null;
          else {
            const n = toNumberOrNull(v);
            obj[k] = n !== null ? n : (v ?? null);
          }
        }
        return obj;
      });

      const ticker = (meta?.ticker ?? parsedFromId.ticker ?? 'UNKNOWN').toUpperCase();
      const tag = meta?.tag ?? parsedFromId.tag ?? null;
      const kind = meta?.kind ?? parsedFromId.kind ?? null;
      const createdAt = parsedFromId.createdAt ?? stat.mtime.toISOString();

      const best = this.computeBestRow(parsed.rows);

      return {
        run: {
          runId,
          ticker,
          tag,
          kind,
          fileName: 'sweep_results.csv',
          createdAt,
          rows: parsed.rows.length,
          best,
          layout: 'structured',
        },
        meta: meta ?? null,
        columns: parsed.columns,
        rows,
      };
    }

    // Fallback: legacy
    const abs = this.safeResolveLegacyCsv(runId);
    if (!fs.existsSync(abs)) throw new NotFoundException('Run not found');

    const stat = await fsp.stat(abs);
    const fileName = path.basename(abs);
    const { ticker, tag } = this.extractLegacyTickerAndTag(fileName);

    const content = await fsp.readFile(abs, 'utf-8');
    const parsed = this.parseCsv(content);

    const rows = parsed.rows.map((r) => {
      const obj: Record<string, string | number | null> = {};
      for (const [k, v] of Object.entries(r)) {
        if (k === 'status' || k === 'error' || k === 'ticker') obj[k] = v ?? null;
        else {
          const n = toNumberOrNull(v);
          obj[k] = n !== null ? n : (v ?? null);
        }
      }
      return obj;
    });

    const best = this.computeBestRow(parsed.rows);

    return {
      run: {
        runId,
        ticker,
        tag,
        kind: 'sweep',
        fileName,
        createdAt: stat.mtime.toISOString(),
        rows: parsed.rows.length,
        best,
        layout: 'legacy',
      },
      meta: null,
      columns: parsed.columns,
      rows,
    };
  }

  async streamRunCsv(runId: string): Promise<{ absPath: string; fileName: string }> {
    // structured first
    const runDir = this.safeResolveRunDir(runId);
    const sweepPath = path.resolve(runDir, 'sweep_results.csv');
    if (fs.existsSync(runDir) && fs.existsSync(sweepPath)) {
      return { absPath: sweepPath, fileName: `${runId}__sweep_results.csv` };
    }

    // legacy
    const abs = this.safeResolveLegacyCsv(runId);
    if (!fs.existsSync(abs)) throw new NotFoundException('Run not found');
    return { absPath: abs, fileName: path.basename(abs) };
  }

  // -------------------- Top-K --------------------
  async listTopK(runId: string): Promise<RunTopKItem[]> {
    const runDir = this.safeResolveRunDir(runId);
    const topkDir = path.resolve(runDir, 'topk');

    if (!fs.existsSync(topkDir)) return [];

    const entries = await fsp.readdir(topkDir, { withFileTypes: true });
    const dirs = entries.filter((e) => e.isDirectory()).map((e) => e.name);

    const items: RunTopKItem[] = [];
    for (const d of dirs) {
      // expected: rank01__ev0p160__...__lb1500
      const m = d.match(/^rank(\d{2})__/i);
      if (!m) continue;

      const rank = Number(m[1]);
      const absDir = path.resolve(topkDir, d);

      const tradesPath = path.resolve(absDir, 'trades.csv');
      const equityPath = path.resolve(absDir, 'equity.csv');

      items.push({
        rank,
        folder: d,
        hasTrades: fs.existsSync(tradesPath),
        hasEquity: fs.existsSync(equityPath),
      });
    }

    items.sort((a, b) => a.rank - b.rank);
    return items;
  }

  async streamTopKCsv(
    runId: string,
    rank: string,
    which: string,
  ): Promise<{ absPath: string; fileName: string }> {
    if (!/^\d+$/.test(rank)) throw new BadRequestException('Invalid rank');
    if (which !== 'trades' && which !== 'equity') throw new BadRequestException('Invalid artifact');

    const runDir = this.safeResolveRunDir(runId);
    const topkDir = path.resolve(runDir, 'topk');
    if (!fs.existsSync(topkDir)) throw new NotFoundException('TopK not found');

    const entries = await fsp.readdir(topkDir, { withFileTypes: true });
    const dirs = entries.filter((e) => e.isDirectory()).map((e) => e.name);

    const want = String(rank).padStart(2, '0');
    const match = dirs.find((d) => d.toLowerCase().startsWith(`rank${want}__`));
    if (!match) throw new NotFoundException('Rank not found');

    const absDir = path.resolve(topkDir, match);
    const absPath = path.resolve(absDir, `${which}.csv`);
    if (!fs.existsSync(absPath)) throw new NotFoundException(`${which}.csv not found`);

    return {
      absPath,
      fileName: `${runId}__rank${want}__${which}.csv`,
    };
  }
}
