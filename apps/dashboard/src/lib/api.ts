import { notifications } from '@mantine/notifications';
import type { RunDetail, RunSummary, RunTopKItem, EquityPoint } from '../types/runs';
import type { PaperSummary, PaperPosition, PaperTrade, PaperEquityPoint } from '../types/paper';

const API_BASE = import.meta.env.VITE_API_BASE ?? '/api';

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);

  if (!res.ok) {
    const text = await res.text().catch(() => '');
    const msg = `API ${res.status} ${res.statusText}${text ? ` — ${text}` : ''}`;
    throw new Error(msg);
  }

  return res.json() as Promise<T>;
}

export function apiUrl(p: string) {
  return `${API_BASE}${p}`;
}

export async function safe<T>(p: Promise<T>, title = 'Request failed'): Promise<T | null> {
  try {
    return await p;
  } catch (e: any) {
    notifications.show({
      title,
      message: e?.message ?? String(e),
      color: 'red',
    });
    return null;
  }
}

export const api = {
  listRuns: () => fetchJson<RunSummary[]>(apiUrl('/runs')),
  getRun: (runId: string) => fetchJson<RunDetail>(apiUrl(`/runs/${encodeURIComponent(runId)}`)),

  downloadRunCsvUrl: (runId: string) => apiUrl(`/runs/${encodeURIComponent(runId)}/download`),

  // Equity curves (JSON)
  getEquityCurve: (runId: string) =>
    fetchJson<EquityPoint[]>(apiUrl(`/runs/${encodeURIComponent(runId)}/equity-json`)),
  getTopKEquityCurve: (runId: string, rank: number) =>
    fetchJson<EquityPoint[]>(apiUrl(`/runs/${encodeURIComponent(runId)}/topk/${encodeURIComponent(String(rank))}/equity-json`)),

  // Top-K
  listTopK: (runId: string) => fetchJson<RunTopKItem[]>(apiUrl(`/runs/${encodeURIComponent(runId)}/topk`)),
  downloadTopKCsvUrl: (runId: string, rank: number, which: 'trades' | 'equity') =>
    apiUrl(`/runs/${encodeURIComponent(runId)}/topk/${encodeURIComponent(String(rank))}/download/${encodeURIComponent(which)}`),

  // Paper trading
  getPaperSummary: () => fetchJson<PaperSummary>(apiUrl('/paper/summary')),
  getPaperPositions: () => fetchJson<PaperPosition[]>(apiUrl('/paper/positions')),
  getPaperTrades: () => fetchJson<PaperTrade[]>(apiUrl('/paper/trades')),
  getPaperEquity: () => fetchJson<PaperEquityPoint[]>(apiUrl('/paper/equity')),
};
