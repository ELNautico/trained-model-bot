import dayjs from 'dayjs';

export function fmtPct(x: unknown) {
  const n = typeof x === 'number' ? x : Number(x);
  if (!Number.isFinite(n)) return '—';
  return `${(n * 100).toFixed(2)}%`;
}

export function fmtNum(x: unknown, digits = 2) {
  const n = typeof x === 'number' ? x : Number(x);
  if (!Number.isFinite(n)) return '—';
  return n.toFixed(digits);
}

export function fmtInt(x: unknown) {
  const n = typeof x === 'number' ? x : Number(x);
  if (!Number.isFinite(n)) return '—';
  return String(Math.trunc(n));
}

export function fmtDate(iso: string) {
  if (!iso) return '—';
  const d = dayjs(iso);
  return d.isValid() ? d.format('YYYY-MM-DD HH:mm') : iso;
}
